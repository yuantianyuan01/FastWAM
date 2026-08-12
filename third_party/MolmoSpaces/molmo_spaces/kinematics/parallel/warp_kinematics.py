"""
Provides a general-purpose, robot-agnostic, vectorized (and optionally GPU accelerated) kinematics solver.
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco import MjData, MjModel, MjSpec

from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.robots.robot_views.abstract import (
    GripperGroup,
    MJCFFrameMixin,
    SimplyActuatedMoveGroup,
)

if TYPE_CHECKING:
    from molmo_spaces.configs.robot_configs import BaseRobotConfig

logger = logging.getLogger(__name__)


@dataclass
class IKBuffers:
    """Preallocated buffers for the IK solver state"""

    pos_err: wp.array(dtype=wp.vec3f)
    rot_err: wp.array(dtype=wp.vec3f)
    jacp: wp.array3d(dtype=wp.float32)
    jacr: wp.array3d(dtype=wp.float32)
    q_dot: wp.array2d(dtype=wp.float32)
    frame_pos: wp.array(dtype=wp.vec3f)
    frame_mat: wp.array(dtype=wp.mat33f)
    frame_bodyid: wp.array(dtype=int)


@dataclass
class IKArgs:
    """Preallocated buffers for the IK solver arguments"""

    poses: wp.array(dtype=wp.mat44f)
    leaf_frame_id: wp.array(dtype=int)  # single-element array (needed for pass-by-reference)
    leaf_frame_type: wp.array(dtype=int)  # single-element array (needed for pass-by-reference)
    damping: wp.array(dtype=wp.float32)  # single-element array (needed for pass-by-reference)
    dt: wp.array(dtype=wp.float32)  # single-element array (needed for pass-by-reference)
    jacobian_mask: wp.array2d(dtype=int)


@dataclass
class SolverData:
    data: mjw.Data
    ik_buffers: IKBuffers
    ik_args: IKArgs
    ik_capture: wp.ScopedCapture | None = None


@wp.kernel
def get_err(
    # inputs
    xpos: wp.array(dtype=wp.vec3f),
    xmat: wp.array(dtype=wp.mat33f),
    poses: wp.array(dtype=wp.mat44f),
    # outputs
    pos_err: wp.array(dtype=wp.vec3f),
    rot_err: wp.array(dtype=wp.vec3f),
):
    """Calculate the body-frame position and rotation error between the current and target poses"""
    i = wp.tid()
    frame_pos = xpos[i]
    frame_rotmat = xmat[i]
    target_pose = poses[i]

    # fmt: off
    site_pose = wp.mat44(
        frame_rotmat[0, 0], frame_rotmat[0, 1], frame_rotmat[0, 2], frame_pos[0],
        frame_rotmat[1, 0], frame_rotmat[1, 1], frame_rotmat[1, 2], frame_pos[1],
        frame_rotmat[2, 0], frame_rotmat[2, 1], frame_rotmat[2, 2], frame_pos[2],
        0.0,       0.0,       0.0,       1.0,
    )
    # fmt: on
    err_trf = wp.inverse(site_pose) @ target_pose

    # fmt: off
    rotmat = wp.mat33(
        err_trf[0, 0], err_trf[0, 1], err_trf[0, 2],
        err_trf[1, 0], err_trf[1, 1], err_trf[1, 2],
        err_trf[2, 0], err_trf[2, 1], err_trf[2, 2],
    )
    # fmt: on
    q = wp.quat_from_matrix(rotmat)
    axis, angle = wp.quat_to_axis_angle(q)
    t = wp.vec3f(err_trf[0, 3], err_trf[1, 3], err_trf[2, 3])

    if wp.abs(angle) < 1e-6:
        pos_err[i] = frame_rotmat @ t
        rot_err[i] = wp.vec3f()
    else:
        w = axis * angle
        V = (
            wp.identity(3, dtype=wp.float32)
            + (1.0 - wp.cos(angle)) / angle**2.0 * wp.skew(w)
            + (angle - wp.sin(angle)) / angle**3.0 * wp.skew(w) @ wp.skew(w)
        )
        t = wp.inverse(V) @ t
        pos_err[i] = frame_rotmat @ t
        rot_err[i] = frame_rotmat @ w


@wp.kernel
def get_jac_frame_info(
    # inputs
    frame_id_: wp.array(dtype=int),  # single-element array
    frame_type_: wp.array(dtype=int),  # single-element array
    xpos: wp.array2d(dtype=wp.vec3f),
    xmat: wp.array2d(dtype=wp.mat33f),
    site_xpos: wp.array2d(dtype=wp.vec3f),
    site_xmat: wp.array2d(dtype=wp.mat33f),
    site_bodyid: wp.array(dtype=int),
    # outputs
    frame_pos: wp.array(dtype=wp.vec3f),
    frame_mat: wp.array(dtype=wp.mat33f),
    frame_bodyid: wp.array(dtype=int),
):
    """Get the position and body ID of the frame"""
    i = wp.tid()
    frame_id = frame_id_[0]
    frame_type = frame_type_[0]
    if frame_type == wp.static(mujoco.mjtObj.mjOBJ_SITE.value):
        frame_pos[i] = site_xpos[i, frame_id]
        frame_mat[i] = site_xmat[i, frame_id]
        frame_bodyid[i] = site_bodyid[frame_id]
    elif frame_type == wp.static(mujoco.mjtObj.mjOBJ_BODY.value):
        frame_pos[i] = xpos[i, frame_id]
        frame_mat[i] = xmat[i, frame_id]
        frame_bodyid[i] = frame_id


@wp.kernel
def mask_jacobian(
    J: wp.array3d(dtype=wp.float32),
    mask: wp.array2d(dtype=int),
    nv: int,
):
    """Apply a column-wise mask to a (3, nv) Jacobian matrix"""
    i = wp.tid()
    for j in range(3):
        for k in range(nv):
            J[i, j, k] = J[i, j, k] * float(mask[i, k])


mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.func
def cholesky_solve6(H: mat66f, b: vec6f) -> vec6f:
    """Solve Hx=b via Cholesky decomposition for 6x6 symmetric positive-definite matrix"""
    # Cholesky decomposition: H = L @ L^T
    L = mat66f()
    for i in range(6):
        for j in range(i + 1):
            s = wp.float32(0.0)
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = wp.sqrt(H[i, i] - s)
            else:
                L[i, j] = (H[i, j] - s) / L[j, j]

    # Forward substitution: L @ y = b
    y = vec6f()
    for i in range(6):
        s = wp.float32(0.0)
        for k in range(i):
            s += L[i, k] * y[k]
        y[i] = (b[i] - s) / L[i, i]

    # Backward substitution: L^T @ x = y
    x = vec6f()
    for i in range(5, -1, -1):
        s = wp.float32(0.0)
        for k in range(i + 1, 6):
            s += L[k, i] * x[k]
        x[i] = (y[i] - s) / L[i, i]

    return x


@wp.kernel
def lm_step(
    # inputs
    jacp: wp.array3d(dtype=wp.float32),
    jacr: wp.array3d(dtype=wp.float32),
    pos_err: wp.array(dtype=wp.vec3f),
    rot_err: wp.array(dtype=wp.vec3f),
    damping: wp.array(dtype=wp.float32),  # single-element array
    dt: wp.array(dtype=wp.float32),  # single-element array
    nv: int,
    # outputs
    qpos: wp.array2d(dtype=wp.float32),
    q_dot: wp.array2d(dtype=wp.float32),
):
    """Single step of the Levenberg-Marquardt solver."""
    i = wp.tid()

    err = vec6f(
        pos_err[i][0],
        pos_err[i][1],
        pos_err[i][2],
        rot_err[i][0],
        rot_err[i][1],
        rot_err[i][2],
    )

    # H = J @ J^T + damping * I, where J = [jacp; jacr] is (6, nv)
    H = mat66f()
    for a in range(6):
        for b in range(6):
            val = wp.float32(0.0)
            for k in range(nv):
                if a < 3:
                    Ja = jacp[i, a, k]
                else:
                    Ja = jacr[i, a - 3, k]
                if b < 3:
                    Jb = jacp[i, b, k]
                else:
                    Jb = jacr[i, b - 3, k]
                val += Ja * Jb
            if a == b:
                val += damping[0]
            H[a, b] = val

    # x = H^{-1} @ err
    x = cholesky_solve6(H, err)

    # q_dot = J^T @ x, dq = q_dot * dt
    for k in range(nv):
        val = wp.float32(0.0)
        for a in range(3):
            val += jacp[i, a, k] * x[a]
            val += jacr[i, a, k] * x[a + 3]
        q_dot[i, k] = val
        qpos[i, k] += val * dt[0]


class SimpleWarpKinematics(ParallelKinematics):
    """
    A warp-based general-purpose parallel inverse kinematics solver for robots.
    This solver only supports optimizing `SimplyActuatedMoveGroups` to reach a target pose for a given `MJCFFrameMixin` move group.
    Most robots satisfy this assumption, but more complicated robots may need custom kinematics implementations.
    """

    def __init__(self, robot_config: "BaseRobotConfig", device: str = "cpu"):
        """
        Args:
            robot_config: The robot configuration.
            device: The warp device to use for the solver.
        """
        super().__init__(robot_config)
        self._device = device

        spec = MjSpec()
        robot_config.robot_cls.add_robot_to_scene(
            robot_config,
            spec,
            prefix=robot_config.robot_namespace,
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=True,
        )
        self._mj_model: MjModel = spec.compile()
        with wp.ScopedDevice(self._device):
            self._mjw_model = mjw.put_model(self._mj_model)

        mj_data = MjData(self._mj_model)
        self._robot_view = robot_config.robot_view_factory(mj_data, robot_config.robot_namespace)

        self._actuated_move_groups: OrderedDict[str, SimplyActuatedMoveGroup] = OrderedDict()
        self._frame_move_groups: dict[str, MJCFFrameMixin] = {}
        for mg_id in self._robot_view.move_group_ids():
            mg = self._robot_view.get_move_group(mg_id)
            assert mg.n_joints == 0 or isinstance(mg, (SimplyActuatedMoveGroup, GripperGroup))
            if isinstance(mg, SimplyActuatedMoveGroup):
                self._actuated_move_groups[mg_id] = mg
            if isinstance(mg, MJCFFrameMixin):
                self._frame_move_groups[mg_id] = mg

        if self._mj_model.nq != self._mj_model.nv:
            raise ValueError(
                "Number of position variables (nq) must equal number of velocity variables (nv) for warp-based IK solver"
            )

    @cache
    def _get_data(self, batch_size: int) -> SolverData:
        with wp.ScopedDevice(self._device):
            data = mjw.make_data(self._mj_model, nworld=batch_size)
            return SolverData(
                data=data,
                ik_buffers=IKBuffers(
                    pos_err=wp.zeros(batch_size, dtype=wp.vec3f),
                    rot_err=wp.zeros(batch_size, dtype=wp.vec3f),
                    jacp=wp.zeros((batch_size, 3, self._mj_model.nv), dtype=wp.float32),
                    jacr=wp.zeros((batch_size, 3, self._mj_model.nv), dtype=wp.float32),
                    q_dot=wp.zeros((batch_size, self._mj_model.nv), dtype=wp.float32),
                    frame_pos=wp.zeros(batch_size, dtype=wp.vec3f),
                    frame_mat=wp.zeros(batch_size, dtype=wp.mat33f),
                    frame_bodyid=wp.zeros(batch_size, dtype=int),
                ),
                ik_args=IKArgs(
                    poses=wp.zeros(batch_size, dtype=wp.mat44f),
                    leaf_frame_id=wp.zeros(1, dtype=int),
                    leaf_frame_type=wp.zeros(1, dtype=int),
                    damping=wp.zeros(1, dtype=wp.float32),
                    dt=wp.zeros(1, dtype=wp.float32),
                    jacobian_mask=wp.ones((batch_size, self._mj_model.nv), dtype=int),
                ),
            )

    def _dicts_to_qpos_arr(self, qpos_dicts: list[dict[str, np.ndarray]]) -> np.ndarray:
        ret = np.empty((len(qpos_dicts), self._mj_model.nq), dtype=np.float32)
        for i, qpos_dict in enumerate(qpos_dicts):
            for mg_id, mg in self._actuated_move_groups.items():
                ret[i, mg.joint_posadr] = qpos_dict[mg_id]
        return ret

    def _qpos_arr_to_dicts(self, qpos_arr: np.ndarray) -> list[dict[str, np.ndarray]]:
        ret = [{} for _ in range(qpos_arr.shape[0])]
        for i, qpos_dict in enumerate(ret):
            for mg_id, mg in self._actuated_move_groups.items():
                qpos_dict[mg_id] = qpos_arr[i, mg.joint_posadr]
        return ret

    def warmup_ik(self, batch_size: int):
        self._get_data(batch_size)

        mj_data = MjData(self._mj_model)
        robot_view = self._robot_config.robot_view_factory(
            mj_data, self._robot_config.robot_namespace
        )
        for mg_id, qpos in self._robot_config.init_qpos.items():
            robot_view.get_move_group(mg_id).joint_pos = qpos
        mujoco.mj_forward(self._mj_model, mj_data)

        mg_id = next(iter(self._frame_move_groups.keys()))
        pose = np.broadcast_to(
            robot_view.get_move_group(mg_id).leaf_frame_to_robot[None], (batch_size, 4, 4)
        )

        self.ik(
            mg_id,
            pose,
            None,
            robot_view.get_qpos_dict(),
            np.eye(4),
            rel_to_base=True,
            max_iter=1,
        )

    def fk(
        self,
        qpos_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Compute forward kinematics for all simple move groups.
        Non MJCFFrameMixin move groups (e.g. bases) are not included in the output.

        Args:
            qpos_dicts: The joint positions.
            base_poses: The base pose(s) of the robots. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the returned pose(s) should be relative to the base frame.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
        """
        is_batch, batch_size, qpos_dicts, base_poses = self._batchify(qpos_dicts, base_poses)
        solver_data = self._get_data(batch_size)
        data = solver_data.data

        qpos_arr = self._dicts_to_qpos_arr(qpos_dicts)
        with wp.ScopedDevice(self._device):
            wp.copy(data.qpos, wp.from_numpy(qpos_arr, dtype=wp.float32))
            mjw.fwd_position(self._mjw_model, data)

        dol = {}
        for mg_id, mg in self._frame_move_groups.items():
            trf = np.repeat(np.expand_dims(np.eye(4), axis=0), batch_size, axis=0)
            xpos, xmat = (
                (data.xpos, data.xmat)
                if mg.leaf_frame_type == "body"
                else (data.site_xpos, data.site_xmat)
            )
            trf[:, :3, 3] = xpos[:, mg.leaf_frame_id].numpy()
            trf[:, :3, :3] = xmat[:, mg.leaf_frame_id].numpy()
            if isinstance(self._robot_view.base, SimplyActuatedMoveGroup):
                # if the base is actuated, trf is in world frame so we need to convert to base frame if necessary
                if rel_to_base:
                    trf = np.linalg.solve(base_poses, trf)
            elif not rel_to_base:
                # if the base is unactuated, trf is in the base frame so we need to convert to world frame if necessary
                trf = base_poses @ trf
            dol[mg_id] = trf

        ret = []
        for i in range(batch_size):
            d = {}
            for mg_id, trf in dol.items():
                d[mg_id] = trf[i]
            ret.append(d)
        return ret if is_batch else ret[0]

    def _ik_solve_step(self, solver_data: SolverData, batch_size: int):
        data = solver_data.data
        ik_buffers = solver_data.ik_buffers
        ik_args = solver_data.ik_args
        mjw.fwd_position(self._mjw_model, data)

        frame_pos = ik_buffers.frame_pos
        frame_mat = ik_buffers.frame_mat
        frame_bodyid = ik_buffers.frame_bodyid
        wp.launch(
            get_jac_frame_info,
            dim=batch_size,
            inputs=[
                ik_args.leaf_frame_id,
                ik_args.leaf_frame_type,
                data.xpos,
                data.xmat,
                data.site_xpos,
                data.site_xmat,
                self._mjw_model.site_bodyid,
            ],
            outputs=[frame_pos, frame_mat, frame_bodyid],
            device=self._device,
        )

        # calculate error
        pos_err = ik_buffers.pos_err
        rot_err = ik_buffers.rot_err
        wp.launch(
            get_err,
            dim=batch_size,
            inputs=[frame_pos, frame_mat, ik_args.poses],
            outputs=[ik_buffers.pos_err, ik_buffers.rot_err],
            device=self._device,
        )

        # calculate Jacobian
        jacp, jacr = ik_buffers.jacp, ik_buffers.jacr
        mjw.jac(self._mjw_model, data, jacp, jacr, frame_pos, frame_bodyid)

        # apply Jacobian mask to lock move groups
        wp.launch(
            mask_jacobian,
            dim=batch_size,
            inputs=[jacp, ik_args.jacobian_mask, self._mjw_model.nv],
            device=self._device,
        )
        wp.launch(
            mask_jacobian,
            dim=batch_size,
            inputs=[jacr, ik_args.jacobian_mask, self._mjw_model.nv],
            device=self._device,
        )

        # solve for joint velocities and update joint positions
        wp.launch(
            lm_step,
            dim=batch_size,
            inputs=[jacp, jacr, pos_err, rot_err, ik_args.damping, ik_args.dt, self._mjw_model.nv],
            outputs=[data.qpos, ik_buffers.q_dot],
            device=self._device,
        )

    def _get_err_norm(self, solver_data: SolverData, batch_size: int) -> np.ndarray:
        data = solver_data.data
        ik_buffers = solver_data.ik_buffers
        ik_args = solver_data.ik_args

        frame_pos = wp.zeros(batch_size, dtype=wp.vec3f)
        frame_mat = wp.zeros(batch_size, dtype=wp.mat33f)
        frame_bodyid = ik_buffers.frame_bodyid
        wp.launch(
            get_jac_frame_info,
            dim=batch_size,
            inputs=[
                ik_args.leaf_frame_id,
                ik_args.leaf_frame_type,
                data.xpos,
                data.xmat,
                data.site_xpos,
                data.site_xmat,
                self._mjw_model.site_bodyid,
            ],
            outputs=[frame_pos, frame_mat, frame_bodyid],
            device=self._device,
        )

        # calculate error
        pos_err = wp.zeros(batch_size, dtype=wp.vec3f)
        rot_err = wp.zeros(batch_size, dtype=wp.vec3f)
        wp.launch(
            get_err,
            dim=batch_size,
            inputs=[frame_pos, frame_mat, ik_args.poses],
            outputs=[pos_err, rot_err],
            device=self._device,
        )

        err = np.sqrt(
            np.linalg.norm(pos_err.numpy(), axis=-1) ** 2
            + np.linalg.norm(rot_err.numpy(), axis=-1) ** 2
        )
        return err

    def _create_jacobian_mask(
        self, batch_size: int, unlocked_move_group_ids: list[str]
    ) -> np.ndarray:
        mask = np.zeros(self._mj_model.nv, dtype=np.int32)
        for mg_id in unlocked_move_group_ids:
            mg = self._actuated_move_groups[mg_id]
            mask[mg.joint_posadr] = 1
        return np.repeat(np.expand_dims(mask, axis=0), batch_size, axis=0)

    def ik(
        self,
        move_group_id: str,
        poses: np.ndarray,
        unlocked_move_group_ids: list[str] | None,
        q0_dicts: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        base_poses: np.ndarray,
        rel_to_base: bool = False,
        converge_eps: float = 1e-3,
        success_eps: float = 5e-4,
        max_iter: int = 50,
        damping: float = 1e-12,
        dt: float = 1.0,
    ):
        """
        Solve inverse kinematics to reach a target pose.

        Args:
            move_group_id: The ID of the move group to solve for.
            poses: The target poses. Shape: (batch_size, 4, 4) or (4, 4)
            unlocked_move_group_ids: The IDs of the move groups that are not locked. If None, all move groups are unlocked.
            q0_dicts: The initial joint positions.
            base_poses: The base poses. Shape: (batch_size, 4, 4) or (4, 4)
            rel_to_base: Whether the target poses are relative to the base frame.
            converge_eps: The convergence threshold in joint space.
            success_eps: The success threshold in twist space.
            max_iter: The maximum number of iterations.
            damping: The damping factor for the Levenberg-Marquardt solver.
            dt: The time step for velocity integration.

        Returns:
            A list of qpos dictionaries for each robot in the batch, or a single qpos dictionary if unbatched.
                If the solver fails to converge for a given robot, the corresponding qpos dictionary is None.
        """
        if move_group_id not in self._frame_move_groups:
            raise ValueError(f"Move group {move_group_id} is not a MJCFFrameMixin")

        if unlocked_move_group_ids is None:
            unlocked_move_group_ids = list(self._actuated_move_groups.keys())
        else:
            for mg_id in unlocked_move_group_ids:
                if mg_id not in self._actuated_move_groups:
                    raise ValueError(f"Move group {mg_id} is not a simply actuated move group!")

        is_batch, batch_size, q0_dicts, base_poses, poses = self._batchify(
            q0_dicts, base_poses, poses
        )
        solver_data = self._get_data(batch_size)
        data = solver_data.data
        ik_args = solver_data.ik_args

        with wp.ScopedDevice(self._device):
            ee_mg = self._frame_move_groups[move_group_id]
            assert isinstance(ee_mg, MJCFFrameMixin)
            leaf_frame_id = ee_mg.leaf_frame_id
            leaf_frame_type = (
                mujoco.mjtObj.mjOBJ_BODY
                if ee_mg.leaf_frame_type == "body"
                else mujoco.mjtObj.mjOBJ_SITE
            )

            if isinstance(self._robot_view.base, SimplyActuatedMoveGroup):
                # if the base is actuated, the solving happens in world frame so ensure targets are in world frame
                if rel_to_base:
                    poses = base_poses @ poses
            elif not rel_to_base:
                # if the base is unactuated, the solving happens in base frame so ensure targets are in base frame
                poses = np.linalg.solve(base_poses, poses)

            wp.copy(ik_args.poses, wp.from_numpy(poses.astype(np.float32), dtype=wp.mat44f))
            ik_args.leaf_frame_id.fill_(leaf_frame_id)
            ik_args.leaf_frame_type.fill_(leaf_frame_type.value)
            ik_args.damping.fill_(damping)
            ik_args.dt.fill_(dt)
            wp.copy(
                ik_args.jacobian_mask,
                wp.from_numpy(
                    self._create_jacobian_mask(batch_size, unlocked_move_group_ids), dtype=wp.int32
                ),
            )

            q0_arr = self._dicts_to_qpos_arr(q0_dicts)
            wp.copy(data.qpos, wp.from_numpy(q0_arr, dtype=wp.float32))

            for i in range(max_iter):
                if self._device.startswith("cuda"):
                    if solver_data.ik_capture is None:
                        with wp.ScopedCapture(device=self._device) as capture:
                            self._ik_solve_step(solver_data, batch_size)
                        solver_data.ik_capture = capture
                    wp.capture_launch(solver_data.ik_capture.graph)
                else:
                    self._ik_solve_step(solver_data, batch_size)
                wp.synchronize()

                q_dot = solver_data.ik_buffers.q_dot.numpy()
                if np.all(np.linalg.norm(q_dot, axis=-1) < converge_eps):
                    logger.debug(
                        f"[SimpleWarpKinematics] Batch of size {batch_size} converged in {i} iterations"
                    )
                    break
            else:
                logger.debug(
                    f"[SimpleWarpKinematics] Batch of size {batch_size} failed to converge in {max_iter} iterations"
                )

            mjw.kinematics(self._mjw_model, data)
            err_norm = self._get_err_norm(solver_data, batch_size)
            success = np.array(err_norm < success_eps)

        ret_q_arr = data.qpos.numpy()
        ret_q_dicts = self._qpos_arr_to_dicts(ret_q_arr)
        for q0_dict, ret_q_dict in zip(q0_dicts, ret_q_dicts):
            for k, v in q0_dict.items():
                if k not in ret_q_dict:
                    ret_q_dict[k] = v

        ret = [d if s else None for d, s in zip(ret_q_dicts, success)]
        return ret if is_batch else ret[0]
