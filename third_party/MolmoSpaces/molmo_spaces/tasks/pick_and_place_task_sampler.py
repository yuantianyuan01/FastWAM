import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.env.object_manager import ObjectManager
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_and_place_object_target_task_sampler import (
    AbstractPickAndPlaceObjectTargetTaskSampler,
)
from molmo_spaces.tasks.task_sampler_errors import ObjectPlacementError
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
from molmo_spaces.utils.mujoco_scene_utils import place_object_near
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.synset_utils import get_valid_receptacle_uids

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig


log = logging.getLogger(__name__)

MAX_BOTTOM_Z_DIFFERENCE = 0.05  # 5cm

_VALID_RECEPTACLE_CACHE: dict[str, dict] | None = None


def _get_cached_valid_receptacles() -> dict[str, dict]:
    """Get cached valid receptacle UIDs filtered by synset rules."""
    global _VALID_RECEPTACLE_CACHE
    if _VALID_RECEPTACLE_CACHE is None:
        _VALID_RECEPTACLE_CACHE = get_valid_receptacle_uids()
    return _VALID_RECEPTACLE_CACHE


class PickAndPlaceReceptacleTaskSampler(AbstractPickAndPlaceObjectTargetTaskSampler):
    """Pick-and-place where external receptacle objects are added to the scene."""

    def __init__(self, config: "PickAndPlaceDataGenConfig") -> None:
        super().__init__(config)
        self._receptacle_cache = {}
        self._receptacle_names: list[str] = []
        self._receptacle_uids: list[str] = []
        self._current_receptacle_index: int = 0
        self._episodes_with_current_receptacle: int = 0
        self._receptacle_multiplier = 1
        self._receptacle_staging_poses = {}
        self.added_objects = {}

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        super().add_auxiliary_objects(spec)
        self._add_receptacles_to_scene(spec)

    @staticmethod
    def receptacle_material_callback(object_spec: mujoco.MjsBody) -> None:
        """Override in subclasses to modify materials on added receptacles."""
        pass

    def _add_receptacles_to_scene(self, spec: MjSpec) -> None:
        """Add receptacle objects to scene for place-in-receptacle tasks."""

        max_size = np.array([0.5, 0.5, 0.15])
        min_size = np.array([0.17, 0.17, -1])

        def valid_receptacle(anno):
            xyz = [anno["boundingBox"][x] for x in "xyz"]
            return (
                anno["receptacle"]
                and anno["primaryProperty"] == "CanPickup"
                and max_size[2] >= xyz[2]
                and max_size[0] >= xyz[0] > min_size[0]
                and max_size[1] >= xyz[1] > min_size[1]
            )

        task_sampler_config = self.config.task_sampler_config

        if not task_sampler_config.place_receptacle_types:
            cache_key = "synset_receptacles"
            if cache_key not in self._receptacle_cache:
                all_valid = _get_cached_valid_receptacles()
                valid_uids = sorted(
                    [uid for uid, anno in all_valid.items() if valid_receptacle(anno)]
                )
                self._receptacle_cache[cache_key] = {uid: all_valid[uid] for uid in valid_uids}
            valid_uids = sorted(self._receptacle_cache[cache_key].keys())
        else:
            recep_types = list(task_sampler_config.place_receptacle_types)
            np.random.shuffle(recep_types)

            cache_key: str = None
            for it in range(len(recep_types)):
                cache_key = recep_types[it].lower()

                if cache_key not in self._receptacle_cache:
                    uid_to_anno = ObjectManager.uid_to_annotation_for_type(cache_key)
                    valid_uids = sorted(
                        [uid for uid, anno in uid_to_anno.items() if valid_receptacle(anno)]
                    )

                    valid_uids = ObjectManager.prefilter_with_clip(cache_key, valid_uids)

                    self._receptacle_cache[cache_key] = {
                        uid: uid_to_anno[uid] for uid in valid_uids
                    }

                    if len(valid_uids):
                        break
                else:
                    valid_uids = sorted(self._receptacle_cache[cache_key].keys())
                    if len(valid_uids):
                        break
            else:
                valid_uids = []

        if len(valid_uids) == 0:
            raise ValueError("No valid receptacle assets found")

        num_receptacles = getattr(task_sampler_config, "num_place_receptacles", 2)
        num_receptacles = min(num_receptacles, len(valid_uids))

        selected_uids = list(np.random.choice(valid_uids, size=num_receptacles, replace=False))

        multiplier = self._receptacle_multiplier

        self._receptacle_names = []
        self._receptacle_uids = []
        self._current_receptacle_index = 0
        name_to_meta = {}

        staging_size = np.array([num_receptacles, multiplier, 1]) / 2
        staging_center = np.array([5, 5, 35])
        staging_start = staging_center + np.array(
            [0.5 - staging_size[0], 0.5 - staging_size[1], staging_size[2]]
        )

        mocap_body = spec.worldbody.add_body(
            name="receptacle_staging_floor",
            mocap=True,
            pos=staging_center,
        )

        mocap_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=staging_size,
            contype=8,
            conaffinity=15,
            group=4,
        )

        self._receptacle_staging_poses = {}

        for i, uid in enumerate(selected_uids):
            receptacle_xml = install_uid(uid)
            for j in range(multiplier):
                receptacle_spec = MjSpec.from_file(str(receptacle_xml))
                if len(receptacle_spec.worldbody.bodies) != 1:
                    log.warning(
                        f"{receptacle_xml} has {len(receptacle_spec.worldbody.bodies)} bodies, expected 1."
                    )
                receptacle_obj: mujoco.MjsBody = receptacle_spec.worldbody.bodies[0]
                self.receptacle_material_callback(receptacle_obj)

                if not receptacle_obj.first_joint():
                    receptacle_obj.add_joint(
                        name=f"{uid}_copy{j}_jntfree",
                        type=mujoco.mjtJoint.mjJNT_FREE,
                        damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
                    )

                z_shift = self._receptacle_cache[cache_key][uid]["boundingBox"]["z"] / 2 + 0.01

                position = staging_start + np.array([i, j, z_shift])
                quat = R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)

                attach_frame = spec.worldbody.add_frame(
                    pos=position,
                    quat=quat,
                )
                namespace = f"{task_sampler_config.place_receptacle_namespace}{i}_{j}/"
                attach_frame.attach_body(receptacle_obj, namespace, "")

                self._receptacle_names.append(receptacle_obj.name)
                self._receptacle_uids.append(uid)

                self._receptacle_staging_poses[receptacle_obj.name] = np.concatenate(
                    (position, quat)
                )

                xml_path_rel = receptacle_xml.relative_to(ASSETS_DIR)
                self.added_objects[receptacle_obj.name] = xml_path_rel

                uid_anno = self._receptacle_cache[cache_key][uid]
                name_to_meta[receptacle_obj.name] = {
                    "asset_id": uid,
                    "category": uid_anno["category"],
                    "object_enum": "temp_object",
                    "is_static": False,
                    "boundingBox": uid_anno.get("boundingBox", {}),
                }

        self.place_receptacle_name = self._receptacle_names[0]
        log.info(
            f"Added {num_receptacles} (x {multiplier}) receptacles to scene: {self._receptacle_uids}"
        )

        self._metadata_adder.update(name_to_meta)

    # -- Receptacle management --

    def advance_to_next_receptacle(self, env: CPUMujocoEnv) -> bool:
        multiplier = self._receptacle_multiplier

        if self._current_receptacle_index + multiplier < len(self._receptacle_names):
            self._current_receptacle_index += multiplier
            self.place_receptacle_name = self._receptacle_names[self._current_receptacle_index]
            self._episodes_with_current_receptacle = 1
            log.info(
                f"Advanced to receptacle {self._current_receptacle_index // multiplier + 1}/{len(self._receptacle_names) // multiplier}: "
                f"{self._receptacle_uids[self._current_receptacle_index]}"
            )
            return True
        else:
            log.info("No more receptacles available to try")
            return False

    @property
    def current_receptacle_uid(self) -> str | None:
        if self._receptacle_uids and self._current_receptacle_index < len(self._receptacle_uids):
            return self._receptacle_uids[self._current_receptacle_index]
        return None

    @property
    def has_more_receptacles(self) -> bool:
        multiplier = self._receptacle_multiplier
        return self._current_receptacle_index + multiplier < len(self._receptacle_names)

    @property
    def active_receptacle_names(self):
        multiplier = self._receptacle_multiplier
        return self._receptacle_names[
            self._current_receptacle_index : self._current_receptacle_index + multiplier
        ]

    # -- Hook implementations --

    def _get_place_target_candidates(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        supporting_geom_id: int,
    ) -> list[str]:
        return self._receptacle_names

    def _prepare_place_target(
        self,
        env: CPUMujocoEnv,
        place_target_name: str,
        pickup_obj_name: str,
        pickup_obj_pos: np.ndarray,
        supporting_geom_id: int,
    ) -> bool:
        """Position all active receptacles near the pickup object."""
        task_sampler_config = self.config.task_sampler_config

        om = env.object_managers[env.current_batch_index]
        data = env.current_data

        for receptacle_name in self.active_receptacle_names:
            if not self._filter_place_target(env, pickup_obj_name, receptacle_name):
                log.info(f"Place receptacle {receptacle_name} fails filter size")
                if self.config.task_sampler_config.added_pickup_objects:
                    self._advance_to_next_added_pickupable(env)
                return False

            receptacle_id = om.get_object_body_id(receptacle_name)

            try:
                place_object_near(
                    data=env.current_data,
                    object_id=receptacle_id,
                    placement_point=pickup_obj_pos,
                    min_dist=task_sampler_config.min_object_to_receptacle_dist,
                    max_dist=task_sampler_config.max_object_to_receptacle_dist,
                    max_tries=task_sampler_config.max_place_receptacle_sampling_attempts,
                    max_dist_to_reference=task_sampler_config.max_robot_to_place_receptacle_dist,
                    supporting_geom_id=supporting_geom_id,
                    z_eps=0.003,
                )
            except ObjectPlacementError:
                log.info(f"Failed to place receptacle {receptacle_name} near pickup object")
                return False

            r_obj = om.get_object(receptacle_name)
            r_base_pos = body_base_pos(data, r_obj.body_id)

            if abs(r_base_pos[2] - pickup_obj_pos[2]) > MAX_BOTTOM_Z_DIFFERENCE:
                raise ValueError(
                    f"Failed to place receptacle {receptacle_name} at same height as pickup object"
                )

        return True

    def _filter_place_target(
        self,
        env: CPUMujocoEnv,
        pickup_obj_name: str,
        place_target_name: str,
    ) -> bool:
        """Ensure pickup object is smaller than receptacle."""
        pickup_asset_id = (
            env.current_scene_metadata["objects"].get(pickup_obj_name, {}).get("asset_id")
        )
        if pickup_asset_id is None:
            log.info(f"Failed to get asset_id for {pickup_obj_name}")
            return False

        pickup_anno = ObjectMeta.annotation(pickup_asset_id)
        if pickup_anno is None:
            log.info(f"Failed to get annotation for {pickup_obj_name}")
            return False

        place_target_meta = env.current_scene_metadata["objects"].get(place_target_name)
        if place_target_meta is None or "boundingBox" not in place_target_meta:
            log.info(f"Failed to get bounding box for {place_target_name}")
            return False

        max_diag = float(np.linalg.norm([place_target_meta["boundingBox"][x] for x in "xyz"]))
        pickup_diag = float(np.linalg.norm([pickup_anno["boundingBox"][x] for x in "xyz"]))

        if pickup_diag > max_diag:
            log.info(
                f"Excluded pickup object {pickup_obj_name} with diag {pickup_diag:.3f} "
                f"larger than place target's {max_diag:.3f}"
            )
            return False

        return True

    # -- Overrides --

    def _select_pickup_object(self, env: CPUMujocoEnv) -> int:
        """Receptacle auto-advance and metadata injection before parent's retry loop."""
        episodes_per_receptacle = getattr(
            self.config.task_sampler_config, "episodes_per_receptacle", 0
        )
        if episodes_per_receptacle > 0:
            self._episodes_with_current_receptacle += 1
            if self._episodes_with_current_receptacle > episodes_per_receptacle:
                if self.has_more_receptacles:
                    self.advance_to_next_receptacle(env)
                else:
                    self._current_receptacle_index = 0
                    if self._receptacle_names:
                        self.place_receptacle_name = self._receptacle_names[0]
                    self._episodes_with_current_receptacle = 1
                    log.info("Wrapped around to first receptacle")

        return super()._select_pickup_object(env)

    def _build_context_objects(self, env, om, pickup_obj_name, supporting_geom_id):
        """Remove non-active receptacles from context (e.g. distractor receptacles)."""
        context_objects = super()._build_context_objects(
            env, om, pickup_obj_name, supporting_geom_id
        )

        if self._receptacle_multiplier > 1:
            remove_from_context = {
                recep_name
                for recep_name in self.active_receptacle_names
                if recep_name != self.place_receptacle_name
            }
            context_objects = [
                obj for obj in context_objects if obj.name not in remove_from_context
            ]

        return context_objects

    def _finalize_task_config(self, env: CPUMujocoEnv):
        """Merge receptacle added_objects into task config."""
        super()._finalize_task_config(env)
        receptacle_added_objects = {
            active: self.added_objects[active]
            for active in self.active_receptacle_names
            if active in self.added_objects
        }
        if receptacle_added_objects:
            added = self.config.task_config.added_objects or {}
            added.update(receptacle_added_objects)
            self.config.task_config.added_objects = added

    def _sample_task(self, env: CPUMujocoEnv):
        from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask

        self._configure_pick_and_place(env)
        return PickAndPlaceTask(env, self.config)


class PickAndPlaceTaskSampler(PickAndPlaceReceptacleTaskSampler):
    """Historical alias for PickAndPlaceReceptacleTaskSampler."""

    pass


import copy

from molmo_spaces.tasks.llm_task_utils import generate_llm_task_from_summaries
from molmo_spaces.tasks.multi_task import MultiTask
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask


class PickAndPlaceMultiTaskSampler(PickAndPlaceTaskSampler):
    def _sample_task(self, env: CPUMujocoEnv) -> MultiTask:
        from molmo_spaces.configs.task_configs import OpeningTaskConfig, PickTaskConfig
        from molmo_spaces.configs.task_sampler_configs import OpenTaskSamplerConfig
        from molmo_spaces.tasks.opening_task_samplers import OpenTaskSampler
        from molmo_spaces.tasks.opening_tasks import OpeningTask
        from molmo_spaces.tasks.pick_task import PickTask
        from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

        om = env.object_managers[env.current_batch_index]

        summaries = om.summarize_top_level_bodies(receptacle_types=[], limit=1024)

        task_prompt, actions = generate_llm_task_from_summaries(summaries)

        print("Generated multi-step task prompt:", task_prompt)
        print("With actions:", actions)

        original_config = self.config
        pick_place_tasks = []
        pick_tasks = []
        open_tasks = []
        first_robot_pose = None

        for action in actions:
            action_type = action[0]

            if action_type == "pick_place":
                pick_id = action[1]
                place_id = action[2]
                self.config = copy.deepcopy(original_config)

                if first_robot_pose is not None:
                    self.config.task_config.robot_base_pose = first_robot_pose

                llm_task = (task_prompt, pick_id, place_id)
                task = super()._sample_task(env, llm_task=llm_task)
                pick_place_tasks.append(task)

                if first_robot_pose is None:
                    first_robot_pose = task.config.task_config.robot_base_pose

        for action in actions:
            action_type = action[0]

            if action_type == "pick_place":
                pick_id = action[1]

                try:
                    from molmo_spaces.utils.pose import pose_mat_to_7d

                    pick_obj = om.get_object_by_name(pick_id)

                    pickup_obj_start_pose = pose_mat_to_7d(pick_obj.pose)

                    pickup_obj_goal_pose = pose_mat_to_7d(pick_obj.pose)
                    pickup_obj_goal_pose[2] += 0.03

                    pick_config = copy.deepcopy(original_config)
                    pick_config.task_type = "pick"

                    pick_config.task_config = PickTaskConfig(
                        task_cls=PickTask,
                        pickup_obj_name=pick_id,
                        robot_base_pose=first_robot_pose,
                        pickup_obj_start_pose=pickup_obj_start_pose.tolist(),
                        pickup_obj_goal_pose=pickup_obj_goal_pose.tolist(),
                    )

                    pick_sampler = PickTaskSampler(pick_config)
                    pick_sampler.candidate_objects = [pick_obj]
                    pick_sampler._task_counter = 0

                    task = pick_sampler._sample_task(env, skip_robot_placement=True)
                    pick_tasks.append(task)
                    log.info(f"Created pick task for {pick_id}")
                except Exception as e:
                    log.warning(f"Failed to create pick task for {pick_id}: {e}")
                    continue

        for action in actions:
            action_type = action[0]

            if action_type == "open":
                object_id = action[1]
                log.info(f"Creating open task for object: {object_id}")

                try:
                    obj = om.get_object_by_name(object_id)

                    open_config = copy.deepcopy(original_config)
                    open_config.task_type = "open"

                    open_config.task_sampler_config = OpenTaskSamplerConfig(
                        task_sampler_class=OpenTaskSampler,
                        target_initial_state_open_percentage=0,
                    )

                    open_config.task_config = OpeningTaskConfig(
                        task_cls=OpeningTask,
                        pickup_obj_name=object_id,
                        joint_index=0,
                        any_inst_of_category=True,
                        robot_base_pose=first_robot_pose,  # Set robot pose from first task
                    )

                    open_sampler = OpenTaskSampler(open_config)
                    open_sampler.candidate_objects = [obj]
                    open_sampler._task_counter = 0

                    task = open_sampler._sample_task(env, skip_robot_placement=True)
                    open_tasks.append(task)
                    log.info(f"Created open task for {object_id}")
                except Exception as e:
                    log.warning(f"Failed to create open task for {object_id}: {e}")
                    continue

        tasks = pick_place_tasks + pick_tasks + open_tasks

        self.config = original_config

        if len(tasks) == 0:
            raise HouseInvalidForTask("Failed to create any valid tasks from LLM actions")

        return MultiTask(tasks, task_prompt)
