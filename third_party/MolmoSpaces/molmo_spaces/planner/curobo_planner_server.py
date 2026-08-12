"""
gRPC server wrapping CuroboPlanner for both arms.

Launch:
    python curobo_planner_server.py --datagen-config RBY1PickAndPlaceDataGenConfig --port 10000

    Or with environment variables:
    DATAGEN_CONFIG=RBY1PickAndPlaceDataGenConfig PORT=10000 python curobo_planner_server.py

Clients connect via gRPC at host:port (no http:// prefix).

Both left and right planners are initialized at startup. Each request must include
"arm": "left"|"right" in the payload.

One server per job. Workers on the same machine hit localhost:<port>.

No .proto compilation required — service is defined via grpc.GenericRpcHandler
with JSON-over-gRPC serialization.

All GPU operations are serialized through a single dedicated GPU worker thread.
This guarantees GPU memory is flat regardless of the number of clients.
"""

import argparse
import concurrent.futures as cf
import glob
import importlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from concurrent import futures

import grpc
import torch

from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.planner.curobo_planner import CuroboPlanner

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Per-arm planner instances — initialized in main()
planners: dict[str, CuroboPlanner] = {}

# ---------------------------------------------------------------------------
# wandb / metrics state
# ---------------------------------------------------------------------------

_use_wandb: bool = False
_t_start: float = 0.0


def _wandb_log(metrics: dict) -> None:
    if _use_wandb:
        import wandb

        wandb.log(metrics)


# ---------------------------------------------------------------------------
# GPU memory probe (mirrors test_grpc_batch.py)
# ---------------------------------------------------------------------------


def _gpu_memory_mb(gpu_index: int = 0) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def _poll_gpu_memory_loop(
    stop_event: threading.Event, interval: float = 1.0, gpu_index: int = 0
) -> None:
    """Background thread: poll GPU memory and stream to wandb."""
    while not stop_event.is_set():
        mb = _gpu_memory_mb(gpu_index)
        if mb is not None:
            _wandb_log({"gpu/memory_mb": mb, "gpu/t": time.perf_counter() - _t_start})
        time.sleep(interval)


_SERVICE = "curobo_planner.CuroboPlanner"
# 512 MB — large batch trajectories can be several MB
_MAX_MSG = 512 * 1024 * 1024

# ---------------------------------------------------------------------------
# GPU worker thread — the ONLY thread that ever touches the GPU
# ---------------------------------------------------------------------------

_gpu_queue: queue.Queue = queue.Queue()


def _gpu_worker() -> None:
    """Single thread that owns all GPU operations. Never call GPU code outside this."""
    while True:
        fn, fut = _gpu_queue.get()
        try:
            fut.set_result(fn())
        except Exception as e:
            fut.set_exception(e)


def _run_on_gpu(fn):
    """Submit a callable to the GPU worker thread and block until it completes."""
    fut = cf.Future()
    _gpu_queue.put((fn, fut))
    return fut.result()


# ---------------------------------------------------------------------------
# Wire encoding (JSON over gRPC, no proto compilation required)
# ---------------------------------------------------------------------------


def _encode(obj: dict) -> bytes:
    return json.dumps(obj).encode()


def _decode(data: bytes) -> dict:
    return json.loads(data)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_planner(body: dict, context: grpc.ServicerContext) -> CuroboPlanner:
    arm = body.get("arm")
    if arm not in ("left", "right"):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "'arm' must be 'left' or 'right'")
    return planners[arm]


def _apply_world(p: CuroboPlanner, obstacles: list[dict], context: grpc.ServicerContext) -> None:
    """Build a WorldConfig from obstacles and update the planner. Must be called inside GPU worker."""
    from curobo.geom.types import Cuboid, WorldConfig

    cuboids = []
    for obj in obstacles:
        if not all(k in obj for k in ("name", "pose", "dims")):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"obstacle missing required fields (name, pose, dims): {obj}",
            )
        cuboids.append(Cuboid(name=obj["name"], pose=obj["pose"], dims=obj["dims"]))
    world_cfg = WorldConfig(cuboid=cuboids)
    p.motion_gen.update_world(world_cfg)
    p.world_config = world_cfg


def _apply_lock_joints(p: CuroboPlanner, lock_joints: dict) -> None:
    """Apply locked joints to the planner. Must be called inside GPU worker."""
    p.motion_gen.update_locked_joints(lock_joints, p.curobo_robot_config_dict)


# ---------------------------------------------------------------------------
# Servicer — one method per RPC
# ---------------------------------------------------------------------------


class _CuroboPlannerServicer:
    # --- Health / info (no GPU access) ---

    def Health(self, req: dict, context) -> dict:
        return {"status": "ok", "arms": list(planners.keys())}

    def JointNames(self, req: dict, context) -> dict:
        p = _get_planner(req, context)
        return {"joint_names": p.joint_names}

    def EeLink(self, req: dict, context) -> dict:
        p = _get_planner(req, context)
        return {"ee_link": p.ee_link_name}

    def JointLimits(self, req: dict, context) -> dict:
        p = _get_planner(req, context)
        limits = p.joint_limits  # shape (2, dof): [lower, upper]
        return {"lower": limits[0].tolist(), "upper": limits[1].tolist()}

    def LockJoints(self, req: dict, context) -> dict:
        p = _get_planner(req, context)
        return {"lock_joints": p.curobo_robot_config_dict["kinematics"].get("lock_joints", {})}

    def AttachedObjects(self, req: dict, context) -> dict:
        p = _get_planner(req, context)
        return {
            "object_to_link": p.attached_object2link_dict,
            "link_to_object": p.attached_link2object_dict,
        }

    # --- FK / IK ---

    def Fk(self, req: dict, context) -> dict:
        if "joint_config" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'joint_config'")
        p = _get_planner(req, context)

        def _run():
            pose = p.fk_solve(req["joint_config"])
            return {"pose": pose.tolist()}

        return _run_on_gpu(_run)

    def Ik(self, req: dict, context) -> dict:
        if "goal_pose" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'goal_pose'")
        p = _get_planner(req, context)

        def _run():
            if "lock_joints" in req:
                _apply_lock_joints(p, req["lock_joints"])
            joint_config, _ = p.ik_solve(
                goal_pose=req["goal_pose"],
                seed_config=req.get("seed_config"),
                return_seeds=req.get("return_seeds", 1),
                disable_collision=req.get("disable_collision", False),
            )
            torch.cuda.empty_cache()
            return {"success": joint_config is not None, "joint_config": joint_config}

        return _run_on_gpu(_run)

    # --- Motion planning ---

    def MotionPlan(self, req: dict, context) -> dict:
        if "joint_position" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'joint_position'")
        if "goal_poses" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'goal_poses'")
        p = _get_planner(req, context)

        def _run():
            t0 = time.perf_counter()
            if "obstacles" in req:
                _apply_world(p, req["obstacles"], context)
            if "lock_joints" in req:
                _apply_lock_joints(p, req["lock_joints"])
            trajectory, result = p.motion_plan(
                joint_position=req["joint_position"],
                goal_pose_lists=req["goal_poses"],
                verbose=req.get("verbose", False),
            )
            success = result.success.item()
            solve_time = float(result.solve_time) if result.solve_time is not None else None
            status = str(result.status)
            del result
            torch.cuda.empty_cache()
            _wandb_log(
                {
                    "plan/latency_s": time.perf_counter() - t0,
                    "plan/success": int(success),
                    "plan/arm": req.get("arm"),
                }
            )
            return {
                "success": success,
                "trajectory": trajectory,
                "status": status,
                "solve_time": solve_time,
            }

        return _run_on_gpu(_run)

    def MotionPlanBatch(self, req: dict, context) -> dict:
        if "joint_positions" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'joint_positions'")
        if "goal_poses" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'goal_poses'")
        joint_positions = req["joint_positions"]
        goal_poses = req["goal_poses"]
        if len(joint_positions) != len(goal_poses):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "'joint_positions' and 'goal_poses' must have the same length",
            )
        p = _get_planner(req, context)

        def _run():
            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            if "obstacles" in req:
                _apply_world(p, req["obstacles"], context)
            if "lock_joints" in req:
                _apply_lock_joints(p, req["lock_joints"])
            result = p.plan_batch(
                joint_positions=joint_positions,
                goal_pose_lists=goal_poses,
                verbose=req.get("verbose", False),
            )
            # Immediately pull everything to CPU then release the GPU result.
            successes = result.success.cpu().numpy().tolist()
            solve_time = float(result.solve_time) if result.solve_time is not None else None
            position_cpu = None
            if result.optimized_plan is not None and result.optimized_plan.position is not None:
                pos = result.optimized_plan.position
                if pos.ndim == 2:
                    pos = pos.unsqueeze(0)
                position_cpu = pos.cpu()
            del result
            torch.cuda.empty_cache()
            # Build trajectories entirely from CPU tensor — no GPU access after this point.
            trajectories = []
            for i, ok in enumerate(successes):
                if ok and position_cpu is not None:
                    try:
                        trajectories.append(position_cpu[i].tolist())
                    except Exception as e:
                        log.warning(f"Failed to extract trajectory at index {i}: {e}")
                        trajectories.append([])
                else:
                    trajectories.append([])
            n = len(successes)
            _wandb_log(
                {
                    "batch/latency_s": time.perf_counter() - t0,
                    "batch/success_rate": sum(successes) / n if n else 0,
                    "batch/num_successes": sum(successes),
                    "batch/batch_size": n,
                    "batch/total_waypoints": sum(len(t) for t in trajectories),
                    "batch/arm": req.get("arm"),
                }
            )
            return {"successes": successes, "trajectories": trajectories, "solve_time": solve_time}

        return _run_on_gpu(_run)

    # --- World management ---

    def UpdateWorld(self, req: dict, context) -> dict:
        if "obstacles" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'obstacles'")
        from curobo.geom.types import Cuboid, WorldConfig

        p = _get_planner(req, context)

        def _run():
            cuboids = []
            for obj in req["obstacles"]:
                if not all(k in obj for k in ("name", "pose", "dims")):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"obstacle missing required fields (name, pose, dims): {obj}",
                    )
                cuboids.append(Cuboid(name=obj["name"], pose=obj["pose"], dims=obj["dims"]))
            new_world_cfg = WorldConfig(cuboid=cuboids)
            p.motion_gen.update_world(new_world_cfg)
            p.world_config = new_world_cfg
            return {"status": "ok", "num_obstacles": len(cuboids)}

        return _run_on_gpu(_run)

    def UpdateObstaclePoses(self, req: dict, context) -> dict:
        if "obstacles" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'obstacles'")
        p = _get_planner(req, context)

        def _run():
            p.update_world_obstacle_poses(
                obstacle_names_to_poses=req["obstacles"],
                pre_invert_poses=req.get("pre_invert_poses", True),
            )
            return {"status": "ok"}

        return _run_on_gpu(_run)

    def EnableObstacles(self, req: dict, context) -> dict:
        if "obstacle_names" not in req or "enable" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'obstacle_names' or 'enable'")
        p = _get_planner(req, context)

        def _run():
            p.enable_obstacles(req["obstacle_names"], req["enable"])
            return {"status": "ok"}

        return _run_on_gpu(_run)

    # --- Object attachment ---

    def AttachObject(self, req: dict, context) -> dict:
        required = ("object_names", "joint_position", "attach_link_names")
        if not all(k in req for k in required):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing one of: {required}")
        p = _get_planner(req, context)

        def _run():
            p.attach_obj(
                object_names=req["object_names"],
                joint_position=req["joint_position"],
                attach_link_names=req["attach_link_names"],
            )
            return {"status": "ok"}

        return _run_on_gpu(_run)

    def DetachObject(self, req: dict, context) -> dict:
        if "attach_link_names" not in req:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing 'attach_link_names'")
        p = _get_planner(req, context)

        def _run():
            p.detach_obj(req["attach_link_names"])
            return {"status": "ok"}

        return _run_on_gpu(_run)

    # --- Reset ---

    def Reset(self, req: dict, context) -> dict:
        p = _get_planner(req, context)

        def _run():
            p.reset()
            return {"status": "ok"}

        return _run_on_gpu(_run)


# ---------------------------------------------------------------------------
# gRPC handler registration (no .proto / protoc required)
# ---------------------------------------------------------------------------

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


class _CuroboPlannerHandler(grpc.GenericRpcHandler):
    def __init__(self, servicer: _CuroboPlannerServicer):
        self._handlers = {
            name: grpc.unary_unary_rpc_method_handler(
                getattr(servicer, name),
                request_deserializer=_decode,
                response_serializer=_encode,
            )
            for name in _RPC_NAMES
        }

    def service_name(self) -> str:
        return _SERVICE

    def service(self, handler_call_details):
        method = handler_call_details.method.rsplit("/", 1)[-1]
        return self._handlers.get(method)


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def _auto_import_configs() -> None:
    """Auto-import all config files so they register themselves in the registry."""
    import molmo_spaces.data_generation.config as config_pkg

    config_dir = os.path.dirname(os.path.abspath(config_pkg.__file__))
    for config_path in glob.glob(os.path.join(config_dir, "*.py")):
        if os.path.basename(config_path) == "__init__.py":
            continue
        module_name = (
            f"molmo_spaces.data_generation.config."
            f"{os.path.splitext(os.path.basename(config_path))[0]}"
        )
        try:
            importlib.import_module(module_name)
        except Exception as e:
            log.warning(f"Could not load config module {module_name}: {e}")


def build_planners(datagen_config_arg: str) -> dict[str, CuroboPlanner]:
    if ":" in datagen_config_arg:
        module_name, config_name = datagen_config_arg.split(":", 1)
        importlib.import_module(module_name)
    else:
        config_name = datagen_config_arg
        _auto_import_configs()

    datagen_config = get_config_class(config_name)()

    policy_config = datagen_config.policy_config
    if policy_config is None:
        raise ValueError(
            f"Datagen config {config_name!r} has no policy_config. "
            "Make sure the config uses a curobo-based policy."
        )

    result = {}
    for arm in ("left", "right"):
        attr = f"{arm}_curobo_planner_config"
        curobo_config = getattr(policy_config, attr, None)
        if curobo_config is None:
            raise ValueError(
                f"policy_config for {config_name!r} has no {attr!r}. "
                f"Available curobo attributes: {[a for a in dir(policy_config) if 'curobo' in a]}"
            )
        log.info(f"Building {arm} CuroboPlanner — config: {curobo_config}")
        result[arm] = CuroboPlanner(curobo_config)

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CuroboPlanner gRPC server (both arms)")

    parser.add_argument(
        "--datagen-config",
        default=os.environ.get("DATAGEN_CONFIG"),
        required=not os.environ.get("DATAGEN_CONFIG"),
        help=(
            "Registered datagen config name (e.g. RBY1PickAndPlaceDataGenConfig). "
            "Supports 'module:ClassName' syntax for direct imports."
        ),
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 10000)))
    parser.add_argument("--gpu-index", type=int, default=int(os.environ.get("GPU_INDEX", 0)))
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "curobo-grpc-stress")
    )
    parser.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    log.info("Initializing CuRobo planners for both arms (this may take 30-60s for CUDA warmup)...")
    planners = build_planners(args.datagen_config)
    log.info("CuroboPlanner ready for both arms.")

    # Start the dedicated GPU worker thread before accepting any requests.
    threading.Thread(target=_gpu_worker, daemon=True).start()
    log.info("GPU worker thread started.")

    # ------------------------------------------------------------------
    # wandb init + GPU memory polling
    # ------------------------------------------------------------------
    _t_start = time.perf_counter()
    if not args.no_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "datagen_config": args.datagen_config,
                "host": args.host,
                "port": args.port,
                "gpu_index": args.gpu_index,
            },
        )
        _use_wandb = True
        log.info(f"wandb run: {wandb.run.url}")

    _stop_gpu_polling = threading.Event()
    threading.Thread(
        target=_poll_gpu_memory_loop,
        args=(_stop_gpu_polling, 1.0, args.gpu_index),
        daemon=True,
    ).start()
    log.info(
        f"GPU memory polling started (gpu_index={args.gpu_index}, wandb={'on' if _use_wandb else 'off'})."
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_MSG),
            ("grpc.max_receive_message_length", _MAX_MSG),
        ],
    )
    server.add_generic_rpc_handlers([_CuroboPlannerHandler(_CuroboPlannerServicer())])
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    log.info(f"gRPC server listening on {args.host}:{args.port}")
    server.wait_for_termination()
