"""
JSON-based eval task sampler for benchmark evaluation.

This task sampler loads episode specifications from JSON benchmark files and
creates tasks accordingly. Unlike the pickle-based frozen config approach,
JSON specs are fully self-contained and human-readable.

The task sampler:
- Loads scene modifications (added_objects, object_poses)
- Configures robot (init_qpos from spec, robot_base_pose from task)
- Sets up cameras from the camera specs
- Creates task instances using dynamic class loading from task_cls

Key design principle: The EpisodeSpec is authoritative. All fields needed to
recreate the episode are in the JSON, and if the field is present it strictly overrides the existing config.
"""

import importlib
import logging
import types
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import (
    CameraSystemConfig,
    FixedExocentricCameraConfig,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.task_configs import (
    BaseMujocoTaskConfig,
    DoorOpeningTaskConfig,
    NavToObjTaskConfig,
    OpeningTaskConfig,
    PickAndPlaceColorTaskConfig,
    PickAndPlaceNextToTaskConfig,
    PickAndPlaceTaskConfig,
    PickTaskConfig,
)
from molmo_spaces.env.data_views import (
    MlSpacesArticulationObject,
    create_mlspaces_body,
)
from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.evaluation.benchmark_schema import (
    BaseTaskSpec,
    DoorOpeningTaskSpec,
    EpisodeSpec,
    ExocentricCameraSpec,
    NavToObjTaskSpec,
    OpenCloseTaskSpec,
    PickAndPlaceColorTaskSpec,
    PickAndPlaceNextToTaskSpec,
    PickAndPlaceTaskSpec,
    PickTaskSpec,
    RobotMountedCameraSpec,
    get_task_spec_field_names,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler
from molmo_spaces.utils.constants.simulation_constants import OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING
from molmo_spaces.utils.lazy_loading_utils import install_uid
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms
from molmo_spaces.utils.object_metadata import ObjectMeta
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)

# Mapping from task class names to their corresponding task config classes.
# When loading from JSON benchmark, we need to create the proper task config type
# so that task-specific fields (like pickup_obj_name) are available.
TASK_CLASS_TO_CONFIG_CLASS: dict[str, type[BaseMujocoTaskConfig]] = {
    "PickTask": PickTaskConfig,
    "PickAndPlaceTask": PickAndPlaceTaskConfig,
    "PickAndPlaceNextToTask": PickAndPlaceNextToTaskConfig,
    "PickAndPlaceColorTask": PickAndPlaceColorTaskConfig,
    "OpeningTask": OpeningTaskConfig,
    "DoorOpeningTask": DoorOpeningTaskConfig,
    "NavToObjTask": NavToObjTaskConfig,
}

# Mapping from task class names to their benchmark schema spec classes.
# Used to validate the raw task dict from JSON and fill in schema defaults
# for fields not explicitly stored in the benchmark file.
TASK_CLASS_TO_SPEC_CLASS: dict[str, type[BaseTaskSpec]] = {
    "PickTask": PickTaskSpec,
    "PickAndPlaceTask": PickAndPlaceTaskSpec,
    "PickAndPlaceNextToTask": PickAndPlaceNextToTaskSpec,
    "PickAndPlaceColorTask": PickAndPlaceColorTaskSpec,
    "OpeningTask": OpenCloseTaskSpec,
    "DoorOpeningTask": DoorOpeningTaskSpec,
    "NavToObjTask": NavToObjTaskSpec,
}


def import_class_from_string(class_path: str) -> type:
    """
    Dynamically import a class from its fully qualified name.

    Args:
        class_path: Fully qualified class name, e.g. "molmo_spaces.tasks.pick_task.PickTask"

    Returns:
        The imported class

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
    """
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid class path: {class_path}. Expected 'module.ClassName' format.")

    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def camera_spec_to_config(
    spec: RobotMountedCameraSpec | ExocentricCameraSpec,
) -> RobotMountedCameraConfig | FixedExocentricCameraConfig:
    """
    Convert a benchmark camera spec to a camera config object.

    The benchmark schema uses simpler camera specs for JSON serialization.
    This converts them to the internal camera config classes used by the
    camera manager.
    """
    if isinstance(spec, RobotMountedCameraSpec) or (
        isinstance(spec, dict) and spec.get("type") == "robot_mounted"
    ):
        if isinstance(spec, dict):
            return RobotMountedCameraConfig(
                name=spec["name"],
                reference_body_names=spec["reference_body_names"],
                camera_offset=spec["camera_offset"],
                lookat_offset=spec["lookat_offset"],
                camera_quaternion=spec.get("camera_quaternion"),
                fov=spec.get("fov"),
                record_depth=spec.get("record_depth", False),
            )
        return RobotMountedCameraConfig(
            name=spec.name,
            reference_body_names=spec.reference_body_names,
            camera_offset=spec.camera_offset,
            lookat_offset=spec.lookat_offset,
            camera_quaternion=spec.camera_quaternion,
            fov=spec.fov,
            record_depth=spec.record_depth,
        )
    elif isinstance(spec, ExocentricCameraSpec) or (
        isinstance(spec, dict) and spec.get("type") == "exocentric"
    ):
        if isinstance(spec, dict):
            return FixedExocentricCameraConfig(
                name=spec["name"],
                pos=spec["pos"],
                up=spec["up"],
                forward=spec["forward"],
                fov=spec.get("fov"),
                record_depth=spec.get("record_depth", False),
            )
        return FixedExocentricCameraConfig(
            name=spec.name,
            pos=spec.pos,
            up=spec.up,
            forward=spec.forward,
            fov=spec.fov,
            record_depth=spec.record_depth,
        )
    else:
        raise ValueError(f"Unknown camera spec type: {type(spec)}")


class JsonEvalTaskSampler(BaseMujocoTaskSampler):
    """
    Task sampler that loads episode configuration from JSON benchmark specs.

    This sampler takes a fully self-contained EpisodeSpec and configures the
    environment accordingly. It handles:
    - Adding auxiliary objects from scene_modifications.added_objects
    - Setting object poses from scene_modifications.object_poses
    - Configuring robot joint positions from robot.init_qpos
    - Setting robot base pose from task.robot_base_pose
    - Setting up cameras from the cameras list
    - Creating task instances using dynamic class loading from task.task_cls

    The key difference from other task samplers is that JsonEvalTaskSampler
    does NOT sample anything - all parameters come from the EpisodeSpec.
    """

    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        episode_spec: EpisodeSpec,
    ) -> None:
        """
        Initialize the JSON eval task sampler.

        The episode_spec is authoritative - all required fields must be present.
        Missing fields will raise errors, not fall back to defaults.

        If ``exp_config.camera_config`` is a ``FrankaEvalCameraSystem`` (set by
        the CLI via ``--use_eval_cameras``), the eval camera pipeline is
        activated: central values are loaded per-episode, cameras are placed
        with visibility-aware fallback, and a deterministic seed ensures
        repeatability.  Otherwise the JSON-recorded cameras are replayed as-is.

        Args:
            exp_config: Base experiment config (provides robot_config, etc.)
            episode_spec: The episode specification from JSON benchmark
        """
        # Validate required fields upfront - fail fast on missing data
        self._validate_episode_spec(episode_spec)

        # Store the episode spec before calling super().__init__
        self.episode_spec = episode_spec

        # Override house_inds to only include the house from the episode spec
        exp_config.task_sampler_config.house_inds = [episode_spec.house_index]

        # Build the JSON-recorded camera config (used when eval cameras are not active).
        self._recorded_camera_config: CameraSystemConfig = self._build_camera_config_from_spec(
            episode_spec
        )

        # Detect whether the caller injected a FrankaEvalCameraSystem
        from molmo_spaces.configs.camera_configs import FrankaEvalCameraSystem

        if isinstance(exp_config.camera_config, FrankaEvalCameraSystem):
            self._eval_camera_system: FrankaEvalCameraSystem | None = exp_config.camera_config
            # Keep the eval system on exp_config.camera_config so the sensor
            # suite (image resolution, camera names) is built from it.
        else:
            self._eval_camera_system = None
            # No eval system — use recorded cameras as-is
            exp_config.camera_config = self._recorded_camera_config

        # Override depth if the policy requires it
        if exp_config.policy_config.force_enable_depth:
            log.info("Force enabling depth for all cameras")
            for camera in exp_config.camera_config.cameras:
                camera.record_depth = True

        # Override exp_config.task_type to match the episode spec's task
        # The task class's judge_success() method checks config.task_type, so it must match.
        exp_config.task_type = self._infer_task_type(episode_spec)

        # Override scene_dataset and data_split from episode spec
        # These must be set BEFORE super().__init__() because the base class uses them
        # to build the scene index map for house lookups
        exp_config.scene_dataset = episode_spec.scene_dataset
        exp_config.data_split = episode_spec.data_split

        # Disable action noise for deterministic evaluation by default
        # TODO(RMH): Add input arg for noise level (high, low, medium) to support noisy eval
        if exp_config.robot_config.action_noise_config is not None:
            exp_config.robot_config.action_noise_config.enabled = False

        # Apply robot-specific evaluation overrides. This is a little hacky, so use sparingly.
        if exp_config.eval_runtime_params and exp_config.eval_runtime_params.robot_override_fn:
            exp_config.eval_runtime_params.robot_override_fn(episode_spec, exp_config)

        super().__init__(exp_config)

        # Cache the task class for _sample_task
        self._task_cls: type | None = None

    def _validate_episode_spec(self, spec: EpisodeSpec) -> None:
        """Validate that all required fields are present in the episode spec."""
        # Validate scene identification
        if not spec.scene_dataset:
            raise ValueError(
                f"Episode spec missing required 'scene_dataset' field. "
                f"house_index={spec.house_index}"
            )

        if not spec.data_split:
            raise ValueError(
                f"Episode spec missing required 'data_split' field. house_index={spec.house_index}"
            )

        # Validate task_cls
        if not spec.task.get("task_cls"):
            raise ValueError(
                f"Episode spec missing required 'task.task_cls' field. "
                f"house_index={spec.house_index}"
            )

        # Validate robot_base_pose
        if not spec.task.get("robot_base_pose"):
            raise ValueError(
                f"Episode spec missing required 'task.robot_base_pose' field. "
                f"house_index={spec.house_index}"
            )

        # Validate cameras
        if not spec.cameras:
            raise ValueError(
                f"Episode spec has empty 'cameras' list. house_index={spec.house_index}"
            )

        # Validate img_resolution
        if not spec.img_resolution or len(spec.img_resolution) != 2:
            raise ValueError(
                f"Episode spec missing or invalid 'img_resolution'. "
                f"Expected (width, height) tuple, got {spec.img_resolution}. "
                f"house_index={spec.house_index}"
            )

        # Validate robot init_qpos
        if not spec.robot.init_qpos:
            raise ValueError(
                f"Episode spec missing 'robot.init_qpos'. house_index={spec.house_index}"
            )

        # Validate language
        if not spec.language.task_description:
            raise ValueError(
                f"Episode spec missing 'language.task_description'. house_index={spec.house_index}"
            )

    def _infer_task_type(self, spec: EpisodeSpec) -> str:
        """Infer task_type from episode spec.

        Uses task_type from spec if available, otherwise infers from task_cls.
        """
        # Use explicit task_type if provided
        task_type = spec.task.get("task_type")
        if task_type:
            return task_type

        # Infer from task_cls
        task_cls = spec.get_task_cls()
        task_cls_to_type = {
            "molmo_spaces.tasks.pick_task.PickTask": "pick",
            "molmo_spaces.tasks.opening_tasks.OpeningTask": "open",
            "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask": "pick_and_place",
            "molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask": "pick_and_place_next_to",
            "molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask": "pick_and_place_color",
            "molmo_spaces.tasks.opening_tasks.DoorOpeningTask": "door_opening",
            "molmo_spaces.tasks.nav_task.NavToObjTask": "nav_to_obj",
        }

        if task_cls in task_cls_to_type:
            return task_cls_to_type[task_cls]

        raise ValueError(
            f"Cannot infer task_type from task_cls: {task_cls}. "
            f"'task_type' should be specified explicitly in the episode spec."
        )

    def set_joint_values(self, env: CPUMujocoEnv) -> None:
        # set the pickup object joint positions
        if "pickup_obj_name" not in self.episode_spec.task:
            return

        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(self.episode_spec.task["pickup_obj_name"])
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
            target_joint_name = self.episode_spec.task["joint_name"]
        except (AttributeError, KeyError):
            log.warning(f"Not setting joint of {pickup_obj}")
            return

        target_joint_index = list(joint_names).index(target_joint_name)
        try:
            joint_start_position = self.episode_spec.task["joint_start_position"][0]
        except (AttributeError, KeyError) as e:
            log.warning("Not setting joint.")
            raise e

        pickup_obj.set_joint_position(target_joint_index, joint_start_position)

    def _build_camera_config_from_spec(self, episode_spec: EpisodeSpec) -> CameraSystemConfig:
        """Build a CameraSystemConfig from the episode spec's camera list.

        This method is called before self.episode_spec is set, so it takes
        the episode_spec as a parameter.
        """
        # cameras validated in _validate_episode_spec
        camera_configs = []
        for cam_spec in episode_spec.cameras:
            # Handle both pydantic models and dicts
            if isinstance(cam_spec, dict):
                cam_type = cam_spec.get("type")
                if cam_type == "robot_mounted":
                    cam_config = RobotMountedCameraConfig(
                        name=cam_spec["name"],
                        reference_body_names=cam_spec["reference_body_names"],
                        camera_offset=cam_spec["camera_offset"],
                        lookat_offset=cam_spec["lookat_offset"],
                        camera_quaternion=cam_spec.get("camera_quaternion"),
                        fov=cam_spec.get("fov"),
                        record_depth=cam_spec.get("record_depth", False),
                    )
                elif cam_type == "exocentric":
                    cam_config = FixedExocentricCameraConfig(
                        name=cam_spec["name"],
                        pos=cam_spec["pos"],
                        up=cam_spec["up"],
                        forward=cam_spec["forward"],
                        fov=cam_spec.get("fov"),
                        record_depth=cam_spec.get("record_depth", False),
                    )
                else:
                    raise ValueError(f"Unknown camera type in spec: {cam_type}")
            else:
                cam_config = camera_spec_to_config(cam_spec)
            camera_configs.append(cam_config)

        return CameraSystemConfig(
            img_resolution=episode_spec.img_resolution,
            cameras=camera_configs,
        )

    def add_auxiliary_objects(self, spec: MjSpec) -> None:
        """Add objects from episode spec's scene_modifications.added_objects.

        Also removes objects specified in scene_modifications.removed_objects from
        the base scene spec.
        """
        # First, remove objects from the base scene if specified
        removed_objects = self.episode_spec.scene_modifications.removed_objects
        if removed_objects:
            log.info(f"Removing {len(removed_objects)} objects from base scene: {removed_objects}")
            for obj_name in removed_objects:
                # Extract the base name (last part after "/") for matching
                obj_base_name = obj_name.split("/")[-1]

                # Try to find the body in the spec
                body_to_remove = None
                current_body = spec.worldbody.first_body()
                while current_body is not None:
                    # Check multiple matching strategies:
                    # 1. Exact match with full name
                    # 2. Body name ends with "/{obj_name}"
                    # 3. Body name matches the base name (last part)
                    # 4. Body name ends with "/{obj_base_name}"
                    if (
                        current_body.name == obj_name
                        or current_body.name.endswith(f"/{obj_name}")
                        or current_body.name == obj_base_name
                        or current_body.name.endswith(f"/{obj_base_name}")
                    ):
                        body_to_remove = current_body
                        break
                    current_body = spec.worldbody.next_body(current_body)

                if body_to_remove is not None:
                    try:
                        spec.delete(body_to_remove)
                        log.info(
                            f"Removed object '{obj_name}' (body: {body_to_remove.name}) from scene spec"
                        )
                    except Exception as e:
                        log.warning(
                            f"Failed to remove object '{obj_name}' from scene spec: {e}. "
                            f"It may not exist in the base scene or may have already been removed."
                        )
                else:
                    log.debug(
                        f"Object '{obj_name}' not found in base scene spec. "
                        f"It may not exist in the base scene or may be an added object."
                    )

        added_objects = self.episode_spec.scene_modifications.added_objects
        object_poses = self.episode_spec.scene_modifications.object_poses

        name_to_meta = {}

        for object_name, object_xml_rel in added_objects.items():
            object_xml = ASSETS_DIR / object_xml_rel
            object_uid = Path(object_xml_rel).stem
            if not object_xml.is_file():
                # Try to install the asset
                object_xml_installed = install_uid(object_uid)
                if object_xml != object_xml_installed:
                    raise ValueError(
                        f"Asset {object_xml} not found, cannot be automatically installed."
                    )

            object_spec = MjSpec.from_file(str(object_xml))
            if len(object_spec.worldbody.bodies) != 1:
                log.warning(
                    f"{object_xml} has {len(object_spec.worldbody.bodies)} bodies, expected 1. Using first one."
                )
            obj_body: mujoco.MjsBody = object_spec.worldbody.bodies[0]

            # Parse object name for proper prefixing
            name_parts = object_name.split("/")
            expected_body_name = name_parts[-1]

            # If body name is empty or doesn't match, set it to match the expected name
            # This handles cases where the XML body has no name or a different name
            if not obj_body.name or obj_body.name.strip() == "":
                obj_body.name = expected_body_name
                log.debug(f"Set empty body name to '{expected_body_name}' from object name")
            elif obj_body.name != expected_body_name:
                log.warning(
                    f"Body name '{obj_body.name}' doesn't match expected '{expected_body_name}' "
                    f"from object name '{object_name}'. Using expected name."
                )
                obj_body.name = expected_body_name

            # Add free joint if not present
            if not obj_body.first_joint():
                obj_body.add_joint(
                    name="XYZ_jntfree",
                    type=mujoco.mjtJoint.mjJNT_FREE,
                    damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
                )

            # Get pose from object_poses if available
            if object_name in object_poses:
                pose = object_poses[object_name]
                pos = pose[0:3]
                quat = pose[3:7]
            else:
                pos = [0, 0, 0]
                quat = [1, 0, 0, 0]

            attach_frame = spec.worldbody.add_frame(pos=pos, quat=quat)

            # Parse object name for proper prefixing
            name_parts = object_name.split("/")
            assert name_parts[-1] == obj_body.name, (
                f"Name mismatch {name_parts[-1]} vs {obj_body.name}"
            )
            attach_frame.attach_body(obj_body, "/".join(name_parts[:-1]) + "/", "")

            annotation = ObjectMeta.annotation(object_uid)
            name_to_meta[object_name] = {
                "asset_id": object_uid,
                "category": annotation["category"],
                "object_enum": "temp_object",
                "is_static": False,
                "boundingBox": annotation.get("boundingBox", {}),
            }

            log.info(f"Added body to scene: {object_name}")

        self._metadata_adder.update(name_to_meta)

    def randomize_scene(self, env: CPUMujocoEnv, robot_view) -> None:
        """
        Set up scene state from episode spec.

        Unlike other task samplers, this doesn't randomize - it applies
        the exact configuration from the episode spec:
        - Robot joint positions from robot.init_qpos
        - Object poses from scene_modifications.object_poses
        """
        # Log episode spec details for debugging
        log.info(
            f"randomize_scene: episode_spec says: "
            f"house_index={self.episode_spec.house_index}, "
            f"scene_dataset={self.episode_spec.scene_dataset}, "
            f"data_split={self.episode_spec.data_split}"
        )
        log.info(
            f"randomize_scene: config says: "
            f"scene_dataset={self.config.scene_dataset}, "
            f"data_split={self.config.data_split}"
        )

        # Call parent for lighting/texture randomization if configured
        # TODO open question from RMH: hardcode disable this? Feel like yes, leaving in place as a comment.
        # super().randomize_scene(env, robot_view)

        model = env.current_model
        data = env.current_data
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Set object poses from episode spec
        object_poses = self.episode_spec.scene_modifications.object_poses
        if object_poses:
            log.info(f"randomize_scene: Setting poses for {len(object_poses)} objects")
            for body_name, pose in object_poses.items():
                try:
                    body = create_mlspaces_body(data, body_name)
                except KeyError:
                    log.warning(
                        f"Body '{body_name}' from object_poses not found in scene — skipping. "
                        f"Episode spec: house_index={self.episode_spec.house_index}, "
                        f"scene_dataset={self.episode_spec.scene_dataset}, "
                        f"data_split={self.episode_spec.data_split}."
                    )
                    continue
                pos_close = np.allclose(body.position, pose[0:3], atol=1e-3)
                orn_diff = R.from_quat(body.quat).inv() * R.from_quat(pose[3:7])
                orn_close = orn_diff.magnitude() < 1e-2

                if not pos_close or not orn_close:
                    log.info(f"Setting pose for body: {body_name}")
                    body.position = pose[0:3]
                    body.quat = pose[3:7]

        self._randomize_colors(env)

        mujoco.mj_forward(model, data)
        self.set_joint_values(env)

        # Set robot joint positions from episode spec
        for group_name, qpos in self.episode_spec.robot.init_qpos.items():
            robot_view.get_move_group(group_name).joint_pos = np.array(qpos)
        mujoco.mj_forward(model, data)

        for robot in env.robots:
            for controller in robot.controllers.values():
                controller.reset()
            if "head" in robot.robot_view.move_group_ids():
                head_mg = robot.robot_view.get_move_group("head")
                head_mg.ctrl = head_mg.noop_ctrl
            # recompute control
            robot.set_stationary()
            robot.compute_control()
        log.info("Scene setup from episode spec completed.")

    def _randomize_colors(self, env: CPUMujocoEnv) -> None:
        # TODO(wilbert): we're testing this on beaker rn to figure out how to make it work
        model = env.current_model
        om = env.object_managers[env.current_batch_index]

        # object_colors = getattr(self.config.task_config, "object_colors", None)
        if "object_colors" not in self.episode_spec.task:
            return

        object_colors = self.episode_spec.task["object_colors"]
        if object_colors:
            model = env.current_model
            om = env.object_managers[env.current_batch_index]
            for object_name, color_rgba in object_colors.items():
                body_id = om.get_object_body_id(object_name)
                for geom_id in descendant_geoms(model, body_id, visible_only=True):
                    model.geom_matid[geom_id] = -1
                    model.geom_rgba[geom_id] = color_rgba
                log.info(f"Colored object {object_name}")

    def setup_cameras(self, env: CPUMujocoEnv, deterministic_only: bool = False) -> None:
        """Set up cameras for this eval episode.

        Two modes:

        1. **Default** (no eval camera system): replay the JSON-recorded cameras
           deterministically.
        2. **Eval camera system** (``--use_eval_cameras``): compute workspace
           center, resolve the exo reference pose from the shoulder mount,
           and place the exo camera via spherical perturbation with visibility
           checks.  Raises ``CameraPlacementError`` on failure.
        """
        if self._eval_camera_system is None:
            env.camera_manager.setup_cameras(env, self._recorded_camera_config)
            return

        from molmo_spaces.utils.eval_camera_randomization_utils import (
            derive_episode_camera_seed,
            setup_eval_cameras,
        )
        from molmo_spaces.utils.task_relevant_objects_and_workspace_utils import (
            compute_workspace_center_from_object_poses,
            get_task_relevant_objects,
        )

        task_relevant_objects = (
            self.episode_spec.task_relevant_objects
            if hasattr(self.episode_spec, "task_relevant_objects")
            and self.episode_spec.task_relevant_objects
            else get_task_relevant_objects(self.config.task_config)
        )
        log.info(f"Task relevant objects: {task_relevant_objects}")

        # Resolve gripper info (used for workspace center and visibility)
        gripper_pos = None
        gripper_body_name = None
        try:
            robot_view = env.current_robot.robot_view
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            gripper_mg = robot_view.get_move_group(gripper_mg_id)
            ee_pose = robot_view.base.pose @ gripper_mg.leaf_frame_to_robot
            gripper_pos = ee_pose[:3, 3]
            gripper_body_name = robot_view.mj_model.body(gripper_mg.root_body_id).name
        except Exception:
            pass

        workspace_center = compute_workspace_center_from_object_poses(
            object_names=task_relevant_objects,
            object_poses=self.episode_spec.scene_modifications.object_poses,
            gripper_pos=gripper_pos,
        )

        # Build the full visibility body list: task objects + gripper
        visibility_bodies = list(task_relevant_objects or [])
        if gripper_body_name and gripper_body_name not in visibility_bodies:
            visibility_bodies.append(gripper_body_name)

        log.info(f"Visibility bodies: {visibility_bodies}")

        rng_seed = derive_episode_camera_seed(self.episode_spec)

        setup_eval_cameras(
            env=env,
            eval_system=self._eval_camera_system,
            task_relevant_bodies=visibility_bodies,
            workspace_center=workspace_center,
            rng_seed=rng_seed,
        )

    def _get_task_class(self) -> type:
        """Get the task class from the episode spec's task_cls field."""
        if self._task_cls is not None:
            return self._task_cls

        task_cls_str = self.episode_spec.get_task_cls()

        # Benchmarks will be using molmo_spaces as the root module, but for legacy compatibility use
        # molmo_spaces
        task_cls_str = task_cls_str.replace(
            "molmo_spaces", "molmo_spaces"
        )  # TODO(rose): forking branch

        self._task_cls = import_class_from_string(task_cls_str)
        return self._task_cls

    def _apply_task_config(self) -> None:
        """
        Apply task-specific fields from episode spec to the config's task_config.

        Creates the appropriate task config class based on the task type from
        the episode spec, then copies fields from episode_spec.task into it.
        This ensures task-specific fields (like pickup_obj_name) are available.
        """
        task_dict = self.episode_spec.task

        # Get the task class name and find the corresponding config class
        task_cls_str = self.episode_spec.get_task_cls()
        task_class_name = task_cls_str.rsplit(".", 1)[-1]  # e.g., "PickAndPlaceTask"

        config_cls = TASK_CLASS_TO_CONFIG_CLASS.get(task_class_name)
        if config_cls is None:
            raise ValueError(
                f"Unknown task class '{task_class_name}' - no mapping to task config class. "
                f"Add it to TASK_CLASS_TO_CONFIG_CLASS in json_eval_task_sampler.py. "
                f"Available mappings: {list(TASK_CLASS_TO_CONFIG_CLASS.keys())}"
            )

        # Validate task_dict through the benchmark schema's TaskSpec model.
        # This fills in schema defaults for any fields not explicitly stored in
        # the JSON, making benchmark_schema.py the single source of truth for
        # eval defaults (not task_configs.py).
        spec_cls = TASK_CLASS_TO_SPEC_CLASS.get(task_class_name)
        if spec_cls is not None:
            validated_spec = spec_cls.model_validate(task_dict)
            task_dict = validated_spec.model_dump()

        # Create new task config of the proper type with task_cls set
        task_cls = self._get_task_class()
        task_config = config_cls(task_cls=task_cls)

        # Copy task fields from episode spec to task_config.
        # Field names are derived from TaskSpec Pydantic models in benchmark_schema.py
        # to stay in sync automatically when new fields are added.
        for field in get_task_spec_field_names():
            if field in task_dict and hasattr(task_config, field):
                setattr(task_config, field, task_dict[field])

        # Convert list fields that should be numpy arrays (JSON doesn't preserve numpy types)
        numpy_array_fields = [
            "articulated_joint_range",
            "articulated_joint_reset_state",
            "robot_base_pose",
        ]
        for field in numpy_array_fields:
            if hasattr(task_config, field):
                val = getattr(task_config, field)
                if val is not None and isinstance(val, list):
                    setattr(task_config, field, np.array(val))

        # Set added_objects and object_poses from scene_modifications
        task_config.added_objects = {
            name: Path(path)
            for name, path in self.episode_spec.scene_modifications.added_objects.items()
        }
        task_config.object_poses = self.episode_spec.scene_modifications.object_poses

        # Set referral expressions from language spec
        task_config.referral_expressions = self.episode_spec.language.referral_expressions
        task_config.referral_expressions_priority = (
            self.episode_spec.language.referral_expressions_priority
        )

        # The JSON benchmark is authoritative for displacement thresholds.
        # Assert they match the expected defaults rather than overriding.
        if isinstance(task_config, PickAndPlaceTaskConfig):
            assert task_config.max_place_receptacle_pos_displacement == 0.15, (
                f"Expected max_place_receptacle_pos_displacement=0.15, "
                f"got {task_config.max_place_receptacle_pos_displacement}"
            )
            assert np.isclose(task_config.max_place_receptacle_rot_displacement, np.radians(60)), (
                f"Expected max_place_receptacle_rot_displacement=radians(60), "
                f"got {task_config.max_place_receptacle_rot_displacement}"
            )

        # Replace the stub task_config with the properly typed one
        self.config.task_config = task_config

    def _sample_task(self, env: CPUMujocoEnv) -> BaseMujocoTask:
        """
        Create the task from episode spec.

        Unlike other task samplers, this doesn't sample - it creates the task
        using the exact configuration from the episode spec.
        """
        assert env.current_batch_index == 0

        # Apply task config from episode spec
        self._apply_task_config()

        # Set robot base pose from task dict (validated in __init__)
        robot_base_pose = self.episode_spec.task["robot_base_pose"]
        robot_view = env.current_robot.robot_view
        robot_pose_m = pos_quat_to_pose_mat(robot_base_pose[0:3], robot_base_pose[3:7])
        robot_view.base.pose = robot_pose_m

        # Reset controllers
        for controller in env.current_robot.controllers.values():
            controller.reset()
        env.current_robot.set_stationary()
        env.current_robot.compute_control()

        # Forward to update positions
        mujoco.mj_forward(env.current_model, env.current_data)

        # Apply object colors (PickAndPlaceColor tasks store per-object rgba).
        object_colors = getattr(self.config.task_config, "object_colors", None)
        if object_colors:
            model = env.current_model
            om = env.object_managers[env.current_batch_index]
            for object_name, color_rgba in object_colors.items():
                body_id = om.get_object_body_id(object_name)
                for geom_id in descendant_geoms(model, body_id, visible_only=True):
                    model.geom_matid[geom_id] = -1
                    model.geom_rgba[geom_id] = color_rgba
                log.info(f"Colored object {object_name}")

        # only one camera setup is needed. cameras and robot placement are not randomized.
        self.setup_cameras(env)

        # Get task class and instantiate
        task_cls = self._get_task_class()
        task = task_cls(env, self.config)

        task_description = self.episode_spec.language.task_description

        def get_task_description(self, _td=task_description) -> str:
            return _td

        task.get_task_description = types.MethodType(get_task_description, task)

        return task
