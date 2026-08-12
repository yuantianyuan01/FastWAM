import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium.spaces as gyms
import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.grasp_sample import add_grasp_collision_bodies
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)


@dataclass
class MoveSegment:
    """
    A move segment is a single movement in a MoveSequence,
    i.e. moving from a point to another point.
    """

    name: str

    @property
    def duration(self) -> float:
        raise NotImplementedError


@dataclass
class TCPMoveSegment(MoveSegment):
    """
    A move segment that moves the robot TCP to a target pose.
    The movement is a straight line in task space.
    """

    start_pose: np.ndarray
    end_pose: np.ndarray
    speed: float

    @property
    def duration(self) -> float:
        lin_vel, _ = transform_to_twist(np.linalg.inv(self.start_pose) @ self.end_pose)
        return np.linalg.norm(lin_vel) / self.speed


@dataclass
class JointMoveSegment(MoveSegment):
    """
    A move segment that moves the robot joints to a target configuration.
    The movement is a straight line in joint space.
    """

    start_qpos: (
        dict[str, np.ndarray | list[float]] | None
    )  # if None, use the qpos at the start of the segment
    end_qpos: dict[str, np.ndarray | list[float]]
    duration_s: float

    @property
    def duration(self) -> float:
        return self.duration_s


class ActionPrimitive(ABC):
    """
    Abstract base class for action primitives.
    An action primitive is a single action that can be executed by the robot.
    """

    def __init__(self, robot_view: RobotView, duration: float):
        self.robot_view = robot_view
        self.duration = duration
        self.start_time = None

    @abstractmethod
    def execute(self) -> bool:
        pass

    @abstractmethod
    def get_current_action(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        self.start_time = None

    def elapsed_time(self) -> float:
        return self.robot_view.mj_data.time - self.start_time

    def check_failure(self) -> bool:
        return False

    def get_current_phase(self) -> str:
        return "unknown"


class MoveSequence(ActionPrimitive):
    """
    A sequence of (TCP or joint) move segments executed sequentially without pauses.
    The sequence is terminated when the last move segment is completed, after
    a settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        settle_time: float,
        move_segments: list[MoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
    ) -> None:
        super().__init__(robot_view, sum(seg.duration for seg in move_segments))
        self._move_segments = move_segments
        self.move_seg_idx = None
        self.move_seg_start_time = None
        self.settle_time = settle_time
        self.is_holding_object = is_holding_object
        self.gripper_empty_threshold = gripper_empty_threshold

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self.move_seg_idx = None
            self.move_seg_start_time = self.start_time

        duration_cumsum = np.cumsum([0] + [seg.duration for seg in self._move_segments])
        idx = np.searchsorted(duration_cumsum, self.elapsed_time(), side="right").item() - 1
        if idx != self.move_seg_idx:
            if idx < len(self._move_segments):
                log.info(f"Moving to {self._move_segments[idx].name}")
            self.move_seg_idx = idx
            self.move_seg_start_time = self.robot_view.mj_data.time

        return self.elapsed_time() >= self.duration + self.settle_time

    def get_current_action(self) -> dict[str, Any]:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            lin_vel, ang_vel = transform_to_twist(
                np.linalg.inv(move_seg.start_pose) @ move_seg.end_pose
            )
            curr_target_pose = move_seg.start_pose @ twist_to_transform(lin_vel * t, ang_vel * t)
        else:
            move_seg = self.move_segments[-1]
            curr_target_pose = move_seg.end_pose

        # Solve IK for current target pose
        mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg_id, curr_target_pose)

    def get_current_phase(self) -> str:
        if self.move_seg_idx is None:
            move_seg = self.move_segments[0]
        elif self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
        else:
            move_seg = self.move_segments[-1]
        return move_seg.name

    def reset(self):
        super().reset()
        self.move_seg_idx = None
        self.move_seg_start_time = None

    def check_failure(self) -> bool:
        if super().check_failure():
            return True

        if self.is_holding_object:
            gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
            gripper = self.robot_view.get_gripper(gripper_mg_id)
            if (
                gripper.inter_finger_dist
                < gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold
            ):
                log.info(
                    f"Object is not in grasp! {gripper.inter_finger_dist:.05f} < {gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold:05f}"
                )
                return True

        return False


class TCPMoveSequence(MoveSequence):
    """
    Action primitive that tracks a trajectory in task space.
    Waypoints are interpolated linearly in task space.
    The sequence is terminated after the cumulative duration of the move segments,
    plus the settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        tcp_to_jp_fn: Callable[[str, np.ndarray], dict[str, Any]],
        settle_time: float,
        move_segments: list[TCPMoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
        tcp_pos_err_threshold: float = np.inf,
        tcp_rot_err_threshold: float = np.inf,
    ) -> None:
        super().__init__(
            robot_view,
            settle_time,
            move_segments,
            is_holding_object,
            gripper_empty_threshold,
        )
        self.tcp_to_jp_fn = tcp_to_jp_fn
        self.tcp_pos_err_threshold = tcp_pos_err_threshold
        self.tcp_rot_err_threshold = tcp_rot_err_threshold

    @property
    def move_segments(self) -> list[TCPMoveSegment]:
        return self._move_segments

    def get_current_target_pose(self) -> np.ndarray:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            lin_vel, ang_vel = transform_to_twist(
                np.linalg.inv(move_seg.start_pose) @ move_seg.end_pose
            )
            curr_target_pose = move_seg.start_pose @ twist_to_transform(lin_vel * t, ang_vel * t)
        else:
            move_seg = self.move_segments[-1]
            curr_target_pose = move_seg.end_pose
        return curr_target_pose

    def get_current_action(self) -> dict[str, Any]:
        curr_target_pose = self.get_current_target_pose()

        # Solve IK for current target pose
        mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg_id, curr_target_pose)

    def check_failure(self) -> bool:
        if super().check_failure():
            return True
        elif self.move_seg_idx is None:
            # we haven't started moving yet
            return False

        curr_target_pose = self.get_current_target_pose()
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)

        trf = np.linalg.inv(gripper.leaf_frame_to_world) @ curr_target_pose
        pos_err = np.linalg.norm(trf[:3, 3])
        rot_err = R.from_matrix(trf[:3, :3]).magnitude()
        return pos_err > self.tcp_pos_err_threshold or rot_err > self.tcp_rot_err_threshold


class JointMoveSequence(MoveSequence):
    """
    Action primitive that tracks a trajectory in joint space.
    Waypoints are interpolated linearly in joint space.
    The sequence is terminated after the cumulative duration of the move segments,
    plus the settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        settle_time: float,
        move_segments: list[JointMoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            robot_view,
            settle_time,
            move_segments,
            is_holding_object,
            gripper_empty_threshold,
        )

    @property
    def move_segments(self) -> list[JointMoveSegment]:
        return self._move_segments

    def execute(self) -> bool:
        ret = super().execute()

        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            if move_seg.start_qpos is None:
                move_seg.start_qpos = self.robot_view.get_qpos_dict()

        return ret

    def get_current_action(self) -> dict[str, Any]:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            curr_target_qpos = {**move_seg.start_qpos}
            for k in move_seg.end_qpos:
                q0 = np.asarray(move_seg.start_qpos[k])
                q1 = np.asarray(move_seg.end_qpos[k])
                curr_target_qpos[k] = q0 + (q1 - q0) * t
        else:
            move_seg = self.move_segments[-1]
            curr_target_qpos = {**move_seg.end_qpos}

        for mg_id in self.robot_view.get_gripper_movegroup_ids():
            if mg_id in curr_target_qpos:
                del curr_target_qpos[mg_id]
        return curr_target_qpos


class GripperAction(ActionPrimitive):
    """
    Action primitive that opens or closes the gripper.
    """

    def __init__(self, robot_view: RobotView, target_open: bool, duration: float) -> None:
        super().__init__(robot_view, duration)
        self.target_open = target_open

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            if self.target_open:
                log.info("Opening gripper...")
            else:
                log.info("Closing gripper...")

            mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
            gripper = self.robot_view.get_gripper(mg_id)
            gripper.set_gripper_ctrl_open(self.target_open)

        return self.elapsed_time() >= self.duration

    def get_current_action(self) -> dict[str, Any]:
        return self.robot_view.get_ctrl_dict()

    def get_current_phase(self):
        if self.target_open:
            return "gripper-open"
        else:
            return "gripper-close"


class NoopAction(ActionPrimitive):
    """
    Action primitive that does nothing for a given duration.
    """

    def __init__(self, robot_view: RobotView, duration: float) -> None:
        super().__init__(robot_view, duration)
        self._action = None

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
        return self.elapsed_time() >= self.duration

    def get_current_action(self) -> dict[str, Any]:
        if self._action is None:
            self._action = self.robot_view.get_noop_ctrl_dict()
        return self._action


class GraspPoseSensor(Sensor):
    """Sensor for the planned grasp pose in 7D format."""

    def __init__(self, uuid: str = "grasp_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)
        self._logged_warning = False

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get grasp pose (using current TCP pose as proxy)."""
        if task._registered_policy is not None:
            assert isinstance(task._registered_policy, BaseObjectManipulationPlannerPolicy)
            return np.array(
                pose_mat_to_7d(task._registered_policy.target_poses["grasp"]),
                dtype=np.float32,
            )
        else:
            if not self._logged_warning:
                log.warning("Policy is not registered or does not support grasp pose sensing.")
                self._logged_warning = True
            return np.zeros(7, dtype=np.float32)


class BaseObjectManipulationPlannerPolicy(PlannerPolicy):
    """
    Base class for object manipulation (open/close/pick/pick-place) planner policies.
    This class provides common functionality for object manipulation planner policies,
    including concatenating multiple action primitives into a single trajectory.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        assert isinstance(config.policy_config, ObjectManipulationPlannerPolicyConfig)
        super().__init__(config, task)
        self.policy_config = config.policy_config
        self.robot_view = task.env.current_robot.robot_view

        self.action_primitives = []
        self.action_idx = 0
        self._retry_count = 0
        self.target_poses = {}
        self.ik_warmed_up = False
        self.sequential_ik_failures = 0

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def planners(self):
        return {}

    def create_policy_sensors(self) -> list[Sensor]:
        return super().create_policy_sensors() + [
            GraspPoseSensor(uuid="grasp_pose"),
        ]

    def get_all_phases(self):
        # Collect all possible phases here, even those not used for a particular trajectory computation
        phases = {
            "unknown": 0,
            "gripper-open": 1,
            "pregrasp": 2,
            "grasp": 3,
            "gripper-close": 4,
            "lift": 5,
            "preplace": 6,
            "place": 7,
            "retreat": 8,
            "go_home": 9,
        }
        return phases

    def reset(self, reset_retries: bool = True):
        if not self.ik_warmed_up:
            with Timer() as warmup_time:
                self.task.env.current_robot.parallel_kinematics.warmup_ik(
                    self.policy_config.grasp_feasibility_batch_size
                )
            self.ik_warmed_up = True
            log.info(f"Warmed up parallel IK solver in {warmup_time.value:.3f}s")

        self.action_primitives = self._compute_trajectory()

        self.action_idx = 0
        if reset_retries:
            self._retry_count = 0

        self.target_poses = {}
        for action_primitive in self.action_primitives:
            if isinstance(action_primitive, TCPMoveSequence):
                for move_segment in action_primitive.move_segments:
                    self.target_poses[move_segment.name] = move_segment.end_pose
            elif isinstance(action_primitive, JointMoveSequence):
                gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
                kinematics = self.task.env.current_robot.kinematics
                for move_segment in action_primitive.move_segments:
                    end_qpos = move_segment.end_qpos
                    base_pose = self.robot_view.base.pose
                    self.target_poses[move_segment.name] = kinematics.fk(end_qpos, base_pose)[
                        gripper_mg_id
                    ]

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        if self._check_for_failures():
            return self._handle_failure()

        # get the action from the current action primitive, absorbing 0-length segments as necessary
        while self.action_idx < len(self.action_primitives):  # guaranteed to terminate
            act_prim = self.action_primitives[self.action_idx]
            proceed = act_prim.execute()
            # cannot have a 0-length segment that doesn't proceed
            assert proceed or act_prim.duration > 0.0
            if proceed:
                self.action_idx += 1
            if not proceed or act_prim.duration > 0.0:
                break
        if self.action_idx < len(self.action_primitives):
            action = act_prim.get_current_action()
        else:
            action = self.action_primitives[-1].get_current_action()
            action["done"] = True
        return action

    def get_phase(self) -> str:
        if self.action_idx < len(self.action_primitives):
            act_prim = self.action_primitives[self.action_idx]
        else:
            act_prim = self.action_primitives[-1]
        return act_prim.get_current_phase()

    @abstractmethod
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        raise NotImplementedError

    def _check_for_failures(self) -> bool:
        # TODO(abhayd): check for collision with other objects
        if self.action_idx >= len(self.action_primitives):
            return True
        action_primitive = self.action_primitives[self.action_idx]
        return action_primitive.check_failure()

    def _handle_failure(self) -> dict[str, Any]:
        if self.retry_count >= self.policy_config.max_retries:
            log.info(f"❌ Max retries ({self.policy_config.max_retries}) exceeded. Task failed.")
            return {"done": True, "success": False}

        self._retry_count += 1
        log.info(
            f"🔄 Failure detected! Initiating retry {self.retry_count}/{self.policy_config.max_retries}"
        )
        self.reset(reset_retries=False)

        action = self.robot_view.get_noop_ctrl_dict()
        return action

    def _tcp_to_jp_fn(self, mg_id: str, target_pose: np.ndarray) -> dict[str, Any]:
        kinematics = self.task.env.current_robot.kinematics

        gripper_mgs = set(self.robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in self.robot_view.move_group_ids() if x not in gripper_mgs]

        jp = kinematics.ik(
            mg_id,
            target_pose,
            mgs_except_gripper,
            self.robot_view.get_qpos_dict(),
            self.robot_view.base.pose,
        )

        action = self.robot_view.get_ctrl_dict()
        if jp is not None:
            self.sequential_ik_failures = 0
            action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})
        else:
            self.sequential_ik_failures += 1
            log.info(f"⚠️ IK failed, holding current position, fails:{self.sequential_ik_failures}")
            if self.sequential_ik_failures >= self.policy_config.max_sequential_ik_failures:
                log.info("❌ Too many sequential IK failures, triggering retry.")
                return self._handle_failure()

        return action

    def check_feasible_ik(self, pose: np.ndarray) -> bool:
        if pose.ndim > 2:
            assert pose.shape[1:] == (4, 4)
            batch_size = pose.shape[0]
            assert batch_size <= self.policy_config.grasp_feasibility_batch_size

            if batch_size == 0:
                return np.array([], dtype=bool)

            # pad to the batch size to avoid triggering recompilation of the IK solver
            if batch_size < self.policy_config.grasp_feasibility_batch_size:
                n_pad = self.policy_config.grasp_feasibility_batch_size - batch_size
                pose = np.concatenate([pose, np.broadcast_to(pose[-1:], (n_pad, 4, 4))])

            robot_view = self.task.env.current_robot.robot_view
            parallel_kinematics = self.task.env.current_robot.parallel_kinematics
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            jp_dicts = parallel_kinematics.ik(
                gripper_mg_id,
                pose,
                None,
                robot_view.get_qpos_dict(),
                robot_view.base.pose,
                rel_to_base=False,
            )
            return np.array([jp_dict is not None for jp_dict in jp_dicts[:batch_size]])
        else:
            assert pose.shape == (4, 4)
            robot_view = self.task.env.current_robot.robot_view
            kinematics = self.task.env.current_robot.kinematics
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            jp_dict = kinematics.ik(
                gripper_mg_id,
                pose,
                robot_view.move_group_ids(),
                robot_view.get_qpos_dict(),
                base_pose=robot_view.base.pose,
            )
            return jp_dict is not None

    def _show_poses(self, poses, style, color=(1, 0, 0, 1)) -> None:
        if self.task.viewer is None:
            return

        assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
        viewer = self.task.viewer
        ngeom = viewer.user_scn.ngeom
        # Define relative parts of the gripper

        half_length = self.policy_config.grasp_length / 2
        half_width = self.policy_config.grasp_width / 2
        grasp_base_pos = np.array(self.policy_config.grasp_base_pos)
        gripper_parts = [
            ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], [0.0, 0.0, 0]),
            (
                "cylinder_left",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, half_length, 0],
                np.array([0.0, half_width, half_length]) + grasp_base_pos,
            ),
            (
                "cylinder_right",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, half_length, 0],
                np.array([0.0, -half_width, half_length]) + grasp_base_pos,
            ),
            (
                "connecting_bar",
                mujoco.mjtGeom.mjGEOM_BOX,
                [0.002, half_width, 0.002],
                grasp_base_pos,
            ),
        ]
        i = 0
        for T in poses:
            for _part_name, geom_type, size, offset in gripper_parts:
                offset_trf = np.eye(4)
                offset_trf[:3, 3] = offset
                A = T.copy() @ offset_trf
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[ngeom + i],
                    type=geom_type,
                    size=np.array(size),
                    pos=A[:3, 3],
                    mat=T[:3, :3].flatten(),
                    rgba=color,
                )
                i += 1
        viewer.user_scn.ngeom = ngeom + i

    @staticmethod
    def add_auxiliary_objects(config: MlSpacesExpConfig, spec: MjSpec) -> None:
        PlannerPolicy.add_auxiliary_objects(config, spec)
        policy_config = config.policy_config
        assert isinstance(policy_config, ObjectManipulationPlannerPolicyConfig)
        if policy_config.filter_colliding_grasps:
            add_grasp_collision_bodies(
                spec,
                policy_config.grasp_collision_batch_size,
                policy_config.grasp_width,
                policy_config.grasp_length,
                policy_config.grasp_height,
                np.array(policy_config.grasp_base_pos),
            )
