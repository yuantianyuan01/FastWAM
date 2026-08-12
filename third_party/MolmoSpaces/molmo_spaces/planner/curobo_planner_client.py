"""
gRPC client for the CuroboPlanner server.

Usage:
    client = CuroboClient("localhost:10000", arm="left")
    trajectory, success = client.motion_plan(joint_position, [goal_pose])

The address is host:port with no http:// prefix. For backward compatibility,
http://host:port is also accepted and the scheme is stripped automatically.
"""

import json

import grpc

_SERVICE = "curobo_planner.CuroboPlanner"
_MAX_MSG = 512 * 1024 * 1024

_RPC_NAMES = [
    "Health",
    "JointNames",
    "EeLink",
    "JointLimits",
    "LockJoints",
    "Fk",
    "Ik",
    "MotionPlan",
    "MotionPlanBatch",
    "UpdateWorld",
    "UpdateObstaclePoses",
    "EnableObstacles",
    "AttachObject",
    "DetachObject",
    "AttachedObjects",
    "Reset",
]


def _encode(obj: dict) -> bytes:
    return json.dumps(obj).encode()


def _decode(data: bytes) -> dict:
    return json.loads(data)


class CuroboClient:
    def __init__(
        self,
        address: str = "localhost:10000",
        arm: str = "left",
        timeout: float = 30.0,
        # backward-compat alias: CuroboClient(base_url="http://localhost:10000")
        base_url: str | None = None,
    ):
        if base_url is not None:
            address = base_url
        # Strip http(s):// scheme so callers can pass URLs unchanged
        if "://" in address:
            address = address.split("://", 1)[1]

        self.arm = arm
        self._timeout = timeout
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_MSG),
                ("grpc.max_receive_message_length", _MAX_MSG),
            ],
        )
        self._rpcs = {
            name: self._channel.unary_unary(
                f"/{_SERVICE}/{name}",
                request_serializer=_encode,
                response_deserializer=_decode,
            )
            for name in _RPC_NAMES
        }

    def close(self) -> None:
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---------------------------------------------------------------------------
    # Internal call helper
    # ---------------------------------------------------------------------------

    def _call(self, method: str, payload: dict) -> dict:
        return self._rpcs[method]({"arm": self.arm, **payload}, timeout=self._timeout)

    # ---------------------------------------------------------------------------
    # Health / info
    # ---------------------------------------------------------------------------

    def health(self) -> bool:
        try:
            return self._rpcs["Health"]({}, timeout=self._timeout)["status"] == "ok"
        except Exception:
            return False

    def joint_names(self) -> list[str]:
        return self._call("JointNames", {})["joint_names"]

    def ee_link(self) -> str:
        return self._call("EeLink", {})["ee_link"]

    def joint_limits(self) -> dict:
        return self._call("JointLimits", {})

    def get_lock_joints(self) -> dict:
        """Returns the base lock_joints dict from the server's robot config."""
        return self._call("LockJoints", {})["lock_joints"]

    # ---------------------------------------------------------------------------
    # FK / IK
    # ---------------------------------------------------------------------------

    def fk(self, joint_config: list[float]) -> list[float]:
        """Returns 7D pose [x, y, z, qw, qx, qy, qz]."""
        return self._call("Fk", {"joint_config": joint_config})["pose"]

    def ik(
        self,
        goal_pose: list[float],
        seed_config: list[float] | None = None,
        return_seeds: int = 1,
        disable_collision: bool = False,
        lock_joints: dict | None = None,
    ) -> tuple[list[float] | None, bool]:
        """Returns (joint_config, success). joint_config is None on failure.

        If lock_joints is provided, it is applied atomically on the server before
        solving IK, preventing races when multiple workers share the same server.
        lock_joints: {joint_name: value, ...}
        """
        payload: dict = {
            "goal_pose": goal_pose,
            "seed_config": seed_config,
            "return_seeds": return_seeds,
            "disable_collision": disable_collision,
        }
        if lock_joints is not None:
            payload["lock_joints"] = lock_joints
        result = self._call("Ik", payload)
        return result["joint_config"], result["success"]

    # ---------------------------------------------------------------------------
    # Motion planning
    # ---------------------------------------------------------------------------

    def motion_plan(
        self,
        joint_position: list[float],
        goal_poses: list[list[float]],
        obstacles: list[dict] | None = None,
        lock_joints: dict | None = None,
        verbose: bool = False,
    ) -> tuple[list[list[float]], bool]:
        """
        Returns (trajectory, success).
        trajectory is a list of joint-position waypoints.

        If obstacles is provided, the world is updated atomically before planning,
        preventing races when multiple workers share the same server.
        obstacles: [{"name": str, "pose": [7d], "dims": [3d]}, ...]

        If lock_joints is provided, it is applied atomically before planning.
        lock_joints: {joint_name: value, ...}
        """
        payload: dict = {
            "joint_position": joint_position,
            "goal_poses": goal_poses,
            "verbose": verbose,
        }
        if obstacles is not None:
            payload["obstacles"] = obstacles
        if lock_joints is not None:
            payload["lock_joints"] = lock_joints
        result = self._call("MotionPlan", payload)
        return result["trajectory"], result["success"]

    def motion_plan_batch(
        self,
        joint_positions: list[list[float]],
        goal_poses: list[list[float]],
        obstacles: list[dict] | None = None,
        lock_joints: dict | None = None,
        verbose: bool = False,
    ) -> tuple[list[list[list[float]]], list[bool]]:
        """
        Batch planning. Returns (trajectories, successes).
        trajectories[i] is [] if query i failed.

        If obstacles is provided, the world is updated atomically before planning,
        preventing races when multiple workers share the same server.
        obstacles: [{"name": str, "pose": [7d], "dims": [3d]}, ...]

        If lock_joints is provided, it is applied atomically before planning.
        lock_joints: {joint_name: value, ...}
        """
        payload: dict = {
            "joint_positions": joint_positions,
            "goal_poses": goal_poses,
            "verbose": verbose,
        }
        if obstacles is not None:
            payload["obstacles"] = obstacles
        if lock_joints is not None:
            payload["lock_joints"] = lock_joints
        result = self._call("MotionPlanBatch", payload)
        return result["trajectories"], result["successes"]

    # ---------------------------------------------------------------------------
    # World management
    # ---------------------------------------------------------------------------

    def update_world(self, obstacles: list[dict]) -> None:
        """
        obstacles: [{"name": str, "pose": [7d], "dims": [3d]}, ...]
        Replaces entire world state.
        """
        self._call("UpdateWorld", {"obstacles": obstacles})

    def update_obstacle_poses(
        self,
        obstacle_names_to_poses: dict[str, list[float]],
        pre_invert_poses: bool = True,
    ) -> None:
        self._call(
            "UpdateObstaclePoses",
            {
                "obstacles": obstacle_names_to_poses,
                "pre_invert_poses": pre_invert_poses,
            },
        )

    def enable_obstacles(self, obstacle_names: list[str], enable: bool) -> None:
        self._call("EnableObstacles", {"obstacle_names": obstacle_names, "enable": enable})

    # ---------------------------------------------------------------------------
    # Object attachment
    # ---------------------------------------------------------------------------

    def attach_object(
        self,
        object_names: list[str],
        joint_position: list[float],
        attach_link_names: list[str],
    ) -> None:
        self._call(
            "AttachObject",
            {
                "object_names": object_names,
                "joint_position": joint_position,
                "attach_link_names": attach_link_names,
            },
        )

    def detach_object(self, attach_link_names: list[str]) -> None:
        self._call("DetachObject", {"attach_link_names": attach_link_names})

    def attached_objects(self) -> dict:
        return self._call("AttachedObjects", {})

    # ---------------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------------

    def reset(self) -> None:
        self._call("Reset", {})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Example CuroboClient query")
    parser.add_argument("--address", default="localhost:10000")
    parser.add_argument("--arm", default="left", choices=["left", "right"])
    args = parser.parse_args()

    client = CuroboClient(args.address, arm=args.arm)

    print(f"Connecting to {args.address}, arm={args.arm}")
    assert client.health(), "Server not reachable"

    joint_names = client.joint_names()
    limits = client.joint_limits()
    print(f"Joints ({len(joint_names)}): {joint_names}")
    print(f"EE link: {client.ee_link()}")

    lower = limits["lower"]
    upper = limits["upper"]
    start = [(lo + hi) / 2 for lo, hi in zip(lower, upper)]

    print(f"\nStart config (midpoint of limits): {[round(v, 3) for v in start]}")

    goal_pose = client.fk(start)
    print(f"FK at start: {[round(v, 4) for v in goal_pose]}")

    print("\nRunning motion_plan (start -> FK of start, trivial plan)...")
    trajectory, success = client.motion_plan(start, [goal_pose])
    print(f"Success: {success}, trajectory length: {len(trajectory)} waypoints")
