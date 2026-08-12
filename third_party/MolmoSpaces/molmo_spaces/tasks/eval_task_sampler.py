import logging
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.env.data_views import (
    MlSpacesArticulationObject,
    MlSpacesObject,
    create_mlspaces_body,
)
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_task import PickTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.utils.constants.object_constants import (
    THOR_PICKUP_OBJECTS_LOWERCASE,
)
from molmo_spaces.utils.constants.simulation_constants import (
    OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
)
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.pose import pos_quat_to_pose_mat, pose_mat_to_7d

if TYPE_CHECKING:
    from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.tasks.task_sampler import (
    BaseMujocoTaskSampler,
)

log = logging.getLogger(__name__)


class EvalTaskSampler(PickTaskSampler):
    """
    Default task sampler for Franka pick-and-place tasks with house iteration control.
    House order (`house_inds`) and samples per house are provided via config.
    """

    def __init__(self, config: "PickBaseConfig") -> None:
        super().__init__(config)
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety

        # TODO(max): 90% sure this can be removed.
        if not config.task_sampler_config.pickup_types:
            config.task_sampler_config.pickup_types = THOR_PICKUP_OBJECTS_LOWERCASE

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Use this function to put task specific assets into the scene."""
        for (
            object_name,
            object_xml_rel,
        ) in self.config.task_config.added_objects.items():
            object_xml = ASSETS_DIR / object_xml_rel
            if not object_xml.is_file():
                objct_uid = object_xml_rel.stem
                object_xml_installed = install_uid(objct_uid)
                if object_xml != object_xml_installed:
                    raise ValueError(
                        f"Asset {object_xml} not found, can't be automatically installed."
                    )

            object_spec = MjSpec.from_file(str(object_xml))
            if len(object_spec.worldbody.bodies) != 1:
                log.warning(
                    f"{object_xml} has {len(object_spec.worldbody.bodies)} bodies, expected 1. Using first one."
                )
            obj_body: mujoco.MjsBody = object_spec.worldbody.bodies[0]

            if not obj_body.first_joint():
                obj_body.add_joint(
                    name="XYZ_jntfree",
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
                )

            pos = self.config.task_config.object_poses[object_name][0:3]
            quat = self.config.task_config.object_poses[object_name][3:7]
            attach_frame = spec.worldbody.add_frame(pos=pos, quat=quat)
            # object name is e.g. 'place_receptacle/Bowl_25'
            name_parts = object_name.split("/")
            # assert len(name_parts) == 2, f"Invalid name {name_parts}"
            assert name_parts[-1] == obj_body.name, (
                f"Name mismatch {name_parts[-1]} vs {obj_body.name}"
            )
            # attach_frame.attach_body(obj_body, name_parts[0] + "/", "")
            attach_frame.attach_body(obj_body, "/".join(name_parts[:-1]) + "/", "")

            self.place_receptacle_name = object_name
            # with open("test_eval.xml", "w") as fhandle:
            #    fhandle.write(spec.to_xml())
            log.info(f"Adding body to scene: {object_name}")

        super().add_auxiliary_objects(spec)

    def set_joint_values(self, env: CPUMujocoEnv) -> None:
        # set the pickup object joint positions
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.config.task_config.pickup_obj_name)
        from molmo_spaces.utils.grasps import get_joint_grasp_path

        if not isinstance(pickup_obj, MlSpacesArticulationObject):
            return
        # only do MlSpacesArticulationObject

        # initialize the task target state
        joint_names = pickup_obj.joint_names
        joint_names_with_grasp_file = []
        for joint_name in joint_names:
            thor_object_name = (
                env.current_scene_metadata.get("objects", {})
                .get(pickup_obj.name, {})
                .get("asset_id", None)
            )
            thor_joint_name = (
                env.current_scene_metadata.get("objects", {})
                .get(pickup_obj.name, {})
                .get("name_map", {})
                .get("joints", {})
                .get(joint_name, None)
            )
            if get_joint_grasp_path(thor_object_name, thor_joint_name) is not None:
                joint_names_with_grasp_file.append(joint_name)
        if len(joint_names_with_grasp_file) == 0:
            raise ValueError(f"No joints with grasp file found for {pickup_obj.name}")

        target_joint_name = None
        try:
            target_joint_name = self.config.task_config.joint_name
        except AttributeError:
            log.warning(f"Not setting joint of {pickup_obj}")
            return

        target_joint_index = list(joint_names).index(target_joint_name)
        try:
            joint_start_position = self.config.task_config.joint_start_position
        except AttributeError as e:
            log.warning("Not setting joint.")
            raise e

        pickup_obj.set_joint_position(target_joint_index, joint_start_position)

    def randomize_scene(self, env: CPUMujocoEnv, robot_view) -> None:
        """Setup scene state: robot joints, texture randomization, cameras."""
        # randomize scene here
        super().randomize_scene(env, robot_view)

        model = env.current_model
        data = env.current_data
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)  # update the xpos poses

        config_poses = self.config.task_config.object_poses
        bodies_in_config = config_poses.keys()

        for body_id in range(model.nbody):
            sim_name = model.body(body_id).name
            if sim_name not in bodies_in_config:
                continue

            sim_body = create_mlspaces_body(data, sim_name)
            pos_close = np.allclose(sim_body.position, config_poses[sim_name][0:3], atol=1e-3)
            orn_diff = R.from_quat(sim_body.quat).inv() * R.from_quat(config_poses[sim_name][3:7])
            orn_close = orn_diff.magnitude() < 1e-2
            if not pos_close or not orn_close:
                log.info(f"Re-setting body: {sim_name}")
            if not pos_close:
                log.info(f"Position difference: {sim_body.position - config_poses[sim_name][0:3]}")
            if not orn_close:
                log.info(f"Orientation difference: {orn_diff.magnitude()}")

            sim_body.position, sim_body.quat = (
                config_poses[sim_name][0:3],
                config_poses[sim_name][3:7],
            )
        # Now that we have done this, set object poses to None
        self.config.task_config.object_poses = None

        # this only sets the joint position for the pickup object for now
        self.set_joint_values(env)

        mujoco.mj_forward(model, data)  # update the xpos poses

        for group_name, qpos in self.config.robot_config.init_qpos.items():
            robot_view.get_move_group(group_name).joint_pos = np.array(qpos)
        assert self.config.robot_config.init_qpos_noise_range is None

        # Specific coloring step for PickAndPlaceColor tasks
        if hasattr(self.config.task_config, "object_colors"):
            for (
                object_name,
                color_rgba,
            ) in self.config.task_config.object_colors.items():
                obj = env.get_object_by_name(object_name)
                model = env.current_model
                for geom_id in obj.geom_ids:
                    model.geom_rgba[geom_id] = color_rgba
                log.info(f"✓ Successfully colored object {object_name}")

        log.info("Scene setup completed.\n")

    def _sample_task(self, env: CPUMujocoEnv) -> PickTask:
        """Sample a pick-and-place task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Sample pickup object
        assert self.config.task_config.pickup_obj_name is not None
        log.info(
            f"✅ Attempting object {self.config.task_config.pickup_obj_name} of {len(self.candidate_objects)}"
        )
        self._task_counter += 1  # update counter, so we don't re-try same object

        # Provisionally setup cameras so visibility resolver is available during robot placement
        self.setup_cameras(env, deterministic_only=True)

        #  supporting receptacle, and place robot accordingly
        self._sample_and_place_robot(env)

        # Ensure robot is in final position before camera setup
        mujoco.mj_forward(env.current_model, env.current_data)

        # Setup cameras after pickup object and robot placement
        # This allows cameras to use task-specific info (pickup object, workspace center)
        self.setup_cameras(env)

        task = self.config.task_config.task_cls(env, self.config)
        return task

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a pickup object and receptacle, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, receptacle_name, placement_region, robot_base_pose
        """
        task_cfg = self.config.task_config
        pickup_obj = env.object_managers[env.current_batch_index].get_object_by_name(
            task_cfg.pickup_obj_name
        )
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        log.info(f"Selected pickup object: {self.config.task_config.pickup_obj_name}")
        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        # randomize pickup object
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            self.texture_randomizer.randomize_object(pickup_obj)

        robot_view = env.current_robot.robot_view
        if not isinstance(pickup_obj, MlSpacesObject):
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        assert task_cfg.robot_base_pose is not None
        robot_pose = task_cfg.robot_base_pose
        robot_pose_m = pos_quat_to_pose_mat(robot_pose[0:3], robot_pose[3:7])
        robot_view.base.pose = robot_pose_m

        task_type = self.config.task_type

        if task_type in ("pick", "open", "close"):
            pickup_obj_goal_pose = pose_mat_to_7d(pickup_obj.pose.copy())
            pickup_obj_goal_pose[2] += 0.05  # 5 cm
            task_cfg.pickup_obj_goal_pose = pickup_obj_goal_pose.tolist()

        elif (
            task_type == "pick_and_place"
            or task_type == "nav_to_obj"
            or task_type == "pick_and_place_next_to"
        ):
            pass

        else:
            raise ValueError(f"Invalid action {task_type}")

        if self.config.task_config.receptacle_name is not None:
            log.info(f"Supporting receptacle: {self.config.task_config.receptacle_name}")


class DefaulEvalTaskSampler(BaseMujocoTaskSampler):
    def set_joint_values(self, env: CPUMujocoEnv) -> None:
        return

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        return

    def _sample_task(self, env: CPUMujocoEnv) -> PickTask:
        """Sample a pick-and-place task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0

        # self._task_counter += 1  # update counter, so we don't re-try same object

        # Provisionally setup cameras so visibility resolver is available during robot placement
        self.setup_cameras(env, deterministic_only=True)

        #  supporting receptacle, and place robot accordingly
        self._sample_and_place_robot(env)

        # Ensure robot is in final position before camera setup
        mujoco.mj_forward(env.current_model, env.current_data)

        # Setup cameras after pickup object and robot placement
        # This allows cameras to use task-specific info (pickup object, workspace center)
        self.setup_cameras(env)

        task = self.config.task_config.task_cls(env, self.config)
        return task

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        task_cfg = self.config.task_config

        robot_view = env.current_robot.robot_view

        assert task_cfg.robot_base_pose is not None
        robot_pose = task_cfg.robot_base_pose
        robot_pose_m = pos_quat_to_pose_mat(robot_pose[0:3], robot_pose[3:7])
        robot_view.base.pose = robot_pose_m
