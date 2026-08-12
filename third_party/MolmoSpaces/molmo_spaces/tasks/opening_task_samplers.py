import logging

import mujoco
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.env.data_views import Door, MlSpacesArticulationObject, MlSpacesObject
from molmo_spaces.env.env import BaseMujocoEnv, CPUMujocoEnv
from molmo_spaces.env.object_manager import Context
from molmo_spaces.tasks.opening_tasks import OpeningTask
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler
from molmo_spaces.tasks.task_sampler import (
    BaseMujocoTaskSampler,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.constants.object_constants import (
    EXTENDED_ARTICULATION_TYPES_THOR,
)
from molmo_spaces.utils.grasps import has_joint_grasp_path
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.scene_maps import ProcTHORMap

log = logging.getLogger(__name__)


class OpenTaskSampler(PickTaskSampler):
    def __init__(self, config: MlSpacesExpConfig) -> None:
        self.candidate_objects: None | list[MlSpacesObject] = None
        self._task_counter = None  # Track tasks within the same house for variety

        if config.task_sampler_config.pickup_types is None:
            config.task_sampler_config.pickup_types = EXTENDED_ARTICULATION_TYPES_THOR
        super().__init__(config)

    def _has_grasps(self, pickup_obj: MlSpacesObject, asset_uid: str):
        assert isinstance(pickup_obj, MlSpacesArticulationObject)
        for joint_name in pickup_obj.joint_names:
            thor_joint_name = (
                self.env.current_scene_metadata.get("objects", {})
                .get(pickup_obj.name, {})
                .get("name_map", {})
                .get("joints", {})
                .get(joint_name, None)
            )
            if has_joint_grasp_path(
                asset_uid,
                thor_joint_name,
                grasp_libraries=self.config.task_sampler_config.grasp_libraries,
            ):
                return True

        log.info(f"Skipping {pickup_obj.name} (uid={asset_uid}) - no grasp file available")
        return False

    def _sample_task(self, env: CPUMujocoEnv, skip_robot_placement: bool = False) -> OpeningTask:
        """Sample an opening or closing task configuration and create the task."""
        # Set current batch index to 0 (most common case for single-batch environments)
        # TODO(rose) at some point: handle multi-batch environments properly
        assert env.current_batch_index == 0
        assert self.candidate_objects is not None and len(self.candidate_objects) > 0

        # Filter object that are not articulable
        self.candidate_objects = [
            obj for obj in self.candidate_objects if isinstance(obj, MlSpacesArticulationObject)
        ]
        if len(self.candidate_objects) == 0:
            raise ValueError("No articulable objects found in the scene")
        log.info(
            f"Found {len(self.candidate_objects)} articulable objects in the scene for opening task sampling"
        )

        # Sample pickup object
        if self.config.task_config.pickup_obj_name is None:
            object_index = self._task_counter % len(self.candidate_objects)
            pickup_obj_name = self.candidate_objects[object_index].name
            self.config.task_config.pickup_obj_name = pickup_obj_name
            log.debug(
                f"✅ Attempting object {pickup_obj_name} {object_index}/{len(self.candidate_objects)}"
            )
        else:
            pickup_obj_name = self.config.task_config.pickup_obj_name
            log.debug(f"✅ Attempting object {pickup_obj_name} of {len(self.candidate_objects)}")

        self._task_counter += 1  # update counter, so we don't re-try same object

        # make sure they're available for the visibility resolver during robot placement
        self.setup_cameras(env, deterministic_only=True)

        # Note: choosing a referral expression via sampling
        om = env.object_managers[env.current_batch_index]
        context_objects = om.get_context_objects(pickup_obj_name, Context.OBJECT)
        try:
            expression_priority = om.referral_expression_priority(pickup_obj_name, context_objects)
        except ValueError:
            expression_priority = [(1.0, 1.0, om.fallback_expression(pickup_obj_name))]
        self.config.task_config.referral_expressions["pickup_obj_name"] = om.sample_expression(
            expression_priority
        )
        self.config.task_config.referral_expressions_priority["pickup_obj_name"] = (
            expression_priority
        )

        #  supporting receptacle, and place robot accordingly
        if not skip_robot_placement:
            self._sample_and_place_robot(env)

            # Ensure robot is in final position before camera setup
            mujoco.mj_forward(env.current_model, env.current_data)

        # Setup cameras after pickup object and robot placement
        # This allows cameras to use task-specific info (pickup object, workspace center)
        self.setup_cameras(env)

        task = OpeningTask(env, self.config)
        return task

    def _sample_and_place_robot(self, env: CPUMujocoEnv) -> None:
        """Sample a pickup object and open/close the joint, place robot using occupancy map, and return sampled params.

        Returns:
            dict with keys: pickup_obj_name, joint_index, joint_name, joint_start_position, robot_base_pose
        """
        task_cfg = self.config.task_config
        om = env.object_managers[env.current_batch_index]
        pickup_obj = om.get_object_by_name(task_cfg.pickup_obj_name)
        task_cfg.pickup_obj_start_pose = pose_mat_to_7d(pickup_obj.pose).tolist()
        log.debug(f"Selected pickup object: {self.config.task_config.pickup_obj_name}")

        # randomize pickup object
        if (
            self.texture_randomizer is not None
            and self.config.task_sampler_config.randomize_textures
        ):
            self.texture_randomizer.randomize_object(pickup_obj)

        # initialize the task target state
        joint_names = pickup_obj.joint_names
        # randomly sample a joint that has grasp file
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
            if has_joint_grasp_path(
                thor_object_name,
                thor_joint_name,
                grasp_libraries=self.config.task_sampler_config.grasp_libraries,
            ):
                joint_names_with_grasp_file.append(joint_name)
        if len(joint_names_with_grasp_file) == 0:
            raise ValueError(f"No joints with grasp file found for {pickup_obj.name}")
        target_joint_name = np.random.choice(joint_names_with_grasp_file)
        target_joint_index = list(joint_names).index(target_joint_name)
        task_cfg.joint_index = target_joint_index
        task_cfg.joint_name = target_joint_name
        task_cfg.joint_start_position = pickup_obj.get_joint_position(target_joint_index)

        joint_range = pickup_obj.get_joint_range(target_joint_index)
        nonzero_index = np.nonzero(joint_range)
        assert len(nonzero_index) == 1, (
            f"Joint range has multiple non-zero indices for {target_joint_name}"
        )

        if self.config.task_type == "open":
            # if task type is open, keep the joint position closed
            task_cfg.joint_start_position = (
                0
                + joint_range[nonzero_index[0]]
                * self.config.task_sampler_config.target_initial_state_open_percentage
            )  # min(joint_range)
        elif self.config.task_type == "close":
            # NOTE(yejin): robot cannot grasp if open fully...
            task_cfg.joint_start_position = (
                joint_range[nonzero_index[0]]
                * self.config.task_sampler_config.target_initial_state_open_percentage
            )  # + randomness to close the joint
        pickup_obj.set_joint_position(target_joint_index, task_cfg.joint_start_position)

        log.debug(f"[TASK SAMPLING] Trying to place robot near '{pickup_obj.name}'")

        robot_view = env.current_robot.robot_view
        if isinstance(pickup_obj, MlSpacesArticulationObject):
            target_pos = pickup_obj.get_joint_leaf_body_position(
                self.config.task_config.joint_index
            )
        elif isinstance(pickup_obj, MlSpacesObject):
            target_pos = pickup_obj.position
        else:
            raise ValueError(f"Invalid pickup object type: {type(pickup_obj)}")

        # Sample robot object z offset
        min_z_offset = self.config.task_sampler_config.robot_object_z_offset_random_min
        max_z_offset = self.config.task_sampler_config.robot_object_z_offset_random_max
        robot_object_z_offset = np.random.uniform(min_z_offset, max_z_offset)
        initial_robot_z = (
            target_pos[2]
            + self.config.task_sampler_config.robot_object_z_offset
            + robot_object_z_offset
        )

        # place robot near receptacle
        robot_placed = env.place_robot_near(
            robot_view=robot_view,
            target=target_pos,
            max_tries=10,  # Use config value or reasonable default
            sampling_radius_range=self.config.task_sampler_config.base_pose_sampling_radius_range,
            robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
            preserve_z=initial_robot_z,
            face_target=True,
            check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
            visibility_resolver=self.get_visibility_resolver(env),
            excluded_positions=self.used_robot_positions[pickup_obj.name],
        )
        if not robot_placed:
            log.info(f"[TASK SAMPLING] Failed to place robot near '{pickup_obj.name}'")
            raise ValueError(f"Failed to place robot near object: {pickup_obj.name}")

        # Add successful position to cache
        self.used_robot_positions[pickup_obj.name].append(robot_view.base.pose[:3, 3])

        # Get final robot pose for return data
        task_cfg.robot_base_pose = pose_mat_to_7d(robot_view.base.pose).tolist()


class DoorOpeningTaskSampler(BaseMujocoTaskSampler):
    """
    Task sampler for RBY1 door opening tasks with dataset/house iteration support.

    This is the main variant for large-scale dataset generation across multiple ProcTHOR houses.
    For single-scene testing, use DoorOpeningFixedSceneTaskSampler instead.

    TODO(rose): Decide whether to:
      1. Add "at least one of" functionality to RandomizedExocentricCameraConfig, OR
      2. Choose specific gripper (left/right) for visibility constraints, OR
      3. Add both grippers as separate constraints (requires ALL visible)

    TODO(rose): Consider adding flag to make camera placement failures raise
    exceptions for critical camera setups (currently just logs error).
    """

    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(exp_config)
        self._task_counter = 0  # Track tasks within the same house for variety
        self._current_house_index = 0  # Track current house for resetting counters
        self._cached_thormap = None  # Cache map per house to avoid memory leaks

    def close(self) -> None:
        """Clean up task sampler resources."""
        import gc

        # Clear cached occupancy map
        if hasattr(self, "_cached_thormap") and self._cached_thormap is not None:
            del self._cached_thormap
            self._cached_thormap = None

        # Call parent cleanup
        super().close()
        gc.collect()

    def reset(self) -> None:
        """Reset the task sampler state."""
        super().reset()
        self._task_counter = 0
        self._current_house_index = 0
        self._cached_thormap = None

    def init_scene(self, env: BaseMujocoEnv) -> None:
        """Initialize scene after loading - set up cameras once per scene."""
        import gc

        # Initialize base randomizers (texture, lighting, dynamics)
        super().init_scene(env)

        # Set up all cameras from camera_config
        # Note: Cameras are set up here after scene load, not per-task
        # For dynamic cameras, they'll be repositioned per-task via get_workspace_center()

        # Clear old occupancy map before creating new one to prevent memory leak
        if self._cached_thormap is not None:
            del self._cached_thormap
            gc.collect()

        # Reset task counter for new house (ensures we cycle through doors/objects properly)
        self._task_counter = 0

        # Generate occupancy map ONCE per house to avoid memory leaks
        log.info(f"[INIT_SCENE] Generating occupancy map for house {self.current_house_index}")
        self._cached_thormap = ProcTHORMap.from_mj_model_path(
            model_path=env.current_model_path,
            agent_radius=self.config.task_sampler_config.robot_safety_radius,
            px_per_m=200,
            device_id=None,
        )
        log.info("[INIT_SCENE] Occupancy map generated successfully")

    def get_workspace_center(self, env: CPUMujocoEnv) -> np.ndarray:
        """Get workspace center for camera placement.

        For door opening tasks, uses the midpoint between door handle and robot,
        raised slightly for better camera angles.
        """
        # Get robot position
        robot_view = env.current_robot.robot_view
        robot_pos = robot_view.base.pose[:3, 3]

        # Try to get door handle position from current task config
        door_handle_pos = None
        if (
            hasattr(self.config.task_config, "door_body_name")
            and self.config.task_config.door_body_name
        ):
            try:
                door_object = Door(self.config.task_config.door_body_name, env.current_data)

                if door_object.num_handles > 0:
                    door_handle_pos = door_object.get_handle_pose()[:3]
            except Exception as e:
                log.debug(f"[CAMERA SETUP] Could not get door handle position: {e}")

        # Compute workspace center
        if door_handle_pos is not None:
            # Workspace center is midpoint between door handle and robot, raised slightly
            workspace_center = (door_handle_pos + robot_pos) / 2.0
            workspace_center[2] += 0.3  # Raise 30cm for better camera angles
        else:
            # Fallback: use robot position with some height
            workspace_center = robot_pos.copy()
            workspace_center[2] += 0.5  # Raise 50cm above robot base

        return workspace_center

    def resolve_visibility_object(self, env: CPUMujocoEnv, key: str) -> str | None:
        """Resolve special visibility object keys for door opening tasks.

        Handles:
        - __door_handle__: Current door handle body name
        - __gripper__: Robot gripper (via base class)

        Note: Used by RandomizedExocentricCameraConfig visibility_constraints.
        Example: visibility_constraints={"__door_handle__": 0.001, "__gripper__": 0.001}
        will resolve to actual body names at camera setup time.
        """
        if key == "__door_handle__":
            if (
                hasattr(self.config.task_config, "door_body_name")
                and self.config.task_config.door_body_name
            ):
                try:
                    door_body_name = self.config.task_config.door_body_name
                    # Find door handle body - look for handle associated with current door
                    door_object = Door(door_body_name, env.current_data)
                    return door_object.handle_name(env, handle_id=0)
                except Exception as e:
                    log.debug(f"[CAMERA SETUP] Could not resolve door handle: {e}")
            return None

        # Delegate to base class for other keys (e.g., __gripper__)
        return super().resolve_visibility_object(env, key)

    def randomize_scene(self, env: BaseMujocoEnv, robot_view) -> None:
        """Apply runtime scene randomization after scene is loaded.

        Note: Door joint parameter randomization is applied during scene setup
        (before MjSpec compilation) when enable_door_joint_randomization=True.
        This method can be used for additional runtime randomizations.
        """
        # randomize scene here
        super().randomize_scene(env, robot_view)

        log.info("[RBY1 SCENE RANDOMIZATION] Door joint randomization applied during scene setup")
        if (
            hasattr(self.config.task_sampler_config, "enable_door_joint_randomization")
            and self.config.task_sampler_config.enable_door_joint_randomization
        ):
            log.info("[RBY1 SCENE RANDOMIZATION] Door joint randomization: ENABLED")
            log.info(
                f"[RBY1 SCENE RANDOMIZATION]   Door stiffness range: {self.config.task_sampler_config.door_stiffness_range}"
            )
            log.info(
                f"[RBY1 SCENE RANDOMIZATION]   Door damping range: {self.config.task_sampler_config.door_damping_range}"
            )
            log.info(
                f"[RBY1 SCENE RANDOMIZATION]   Handle stiffness range: {self.config.task_sampler_config.handle_stiffness_range}"
            )
            log.info(
                f"[RBY1 SCENE RANDOMIZATION]   Handle damping range: {self.config.task_sampler_config.handle_damping_range}"
            )
        else:
            log.info("[RBY1 SCENE RANDOMIZATION] Door joint randomization: DISABLED")

        # Set robot joints
        for group_name, qpos in self.config.robot_config.init_qpos.items():
            qpos = np.array(qpos)
            if (
                self.config.robot_config.init_qpos_noise_range is not None
                and group_name in self.config.robot_config.init_qpos_noise_range
            ):
                noise_mag = np.array(self.config.robot_config.init_qpos_noise_range[group_name])
                perturb = np.random.uniform(-noise_mag, noise_mag)
            else:
                perturb = np.zeros_like(qpos)
            robot_view.get_move_group(group_name).joint_pos = qpos + perturb

        # TODO: add additional runtime randomization logic here if needed
        # e.g., object positions, lighting, etc.

    def _sample_door_and_place_robot(self, env: CPUMujocoEnv, door_body_names: list[str]):
        """Sample a door and place the robot at a valid base pose."""

        # Get list of door body names to sample from if not provided
        if len(door_body_names) == 0:
            raise HouseInvalidForTask("No doors found in the scene")

        # Sample a valid door and robot base pose
        np.random.shuffle(door_body_names)
        for chosen_door_name in door_body_names:
            door_object = Door(chosen_door_name, env.mj_datas[0])
            door_object.set_joint_position(door_object.get_hinge_joint_index(), 0.0)

            # TODO(roseh): come back here and check for memory spikes with the occupancy map
            robot_placed = env.place_robot_near(
                robot_view=env.current_robot.robot_view,
                target=door_object,
                max_tries=10,
                sampling_radius_range=self.config.task_sampler_config.base_pose_sampling_radius_range,
                robot_safety_radius=self.config.task_sampler_config.robot_safety_radius,
                face_target=True,
                check_camera_visibility=self.config.task_sampler_config.check_robot_placement_visibility,
                visibility_resolver=self.get_visibility_resolver(env),
                excluded_positions=self.used_robot_positions[door_object.name],
            )
            if robot_placed:
                log.info(f"[TASK SAMPLING] Successfully placed robot near '{door_object.name}'")
                self.used_robot_positions[door_object.name].append(
                    env.current_robot.robot_view.base.pose[:3, 3]
                )
                return door_object
            else:
                log.info(
                    f"[TASK SAMPLING] Failed to place robot near '{door_object.name}'. Trying next door..."
                )
        raise ValueError(
            "Was not able to place robot near any door in the scene. Skipping this task."
        )

    def _sample_task(self, env: CPUMujocoEnv):
        """Sample a door opening task configuration and create the task."""

        # Set current batch index to 0 (most common case for single-batch environments)
        env.current_batch_index = 0
        task_cfg = self.config.task_config
        if task_cfg.door_body_name is not None:
            door_body_names = [task_cfg.door_body_name]
        else:
            om = env.object_managers[env.current_batch_index]
            door_body_names = om.find_door_names()

        log.info("[TASK SAMPLING] Starting door sampling and robot placement...")
        door_object = self._sample_door_and_place_robot(env, door_body_names)

        # Check if any parameters are already set (from preset) and log override
        if task_cfg.door_body_name is not None:
            log.info(
                "[TASK SAMPLING] Overriding sampled parameter door_body_name with fixed value from task config"
            )
        if task_cfg.robot_base_pose is not None:
            log.info(
                "[TASK SAMPLING] Overriding sampled parameter robot_base_pose with fixed value from task config"
            )
        if task_cfg.articulated_joint_range is not None:
            log.info(
                "[TASK SAMPLING] Overriding sampled parameter articulated_joint_range with fixed value from task config"
            )
        if task_cfg.articulated_joint_reset_state is not None:
            log.info(
                "[TASK SAMPLING] Overriding sampled parameter articulated_joint_reset_state with fixed value from task config"
            )

        # Set sampled values (only if not already set from preset)
        if task_cfg.door_body_name is None:
            task_cfg.door_body_name = door_object.name
        if task_cfg.robot_base_pose is None:
            task_cfg.robot_base_pose = pose_mat_to_7d(
                env.current_robot.robot_view.base.pose
            ).tolist()
        if task_cfg.articulated_joint_range is None:
            task_cfg.articulated_joint_range = np.array(
                door_object.get_joint_range(door_object.get_hinge_joint_index())
            )
        if task_cfg.articulated_joint_reset_state is None:
            task_cfg.articulated_joint_reset_state = np.array([0.0])

        # Ensure robot is in final position before camera setup
        mujoco.mj_forward(env.current_model, env.current_data)

        # Setup cameras after task config is created and robot is placed
        # This uses the new unified camera system via setup_cameras()
        # All cameras are defined in camera_config (MJCF cameras for RBY1)
        # Dynamic cameras will be positioned based on get_workspace_center()
        self.setup_cameras(env)

        # Create and return the task using self.config (which has the modified task_config)
        return task_cfg.task_cls(env, self.config)
