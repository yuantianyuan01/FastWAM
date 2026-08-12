"""
Parallel batch-planning stress test for the CuroboPlanner gRPC server.

Spins up N worker threads that each fire M motion_plan_batch requests.
Metrics (GPU memory, per-batch latency/success) are logged to wandb by the
server. This script prints a client-side throughput summary to stdout.

Usage:
    python test_grpc_batch.py --address localhost:10000 \
        --workers 4 --batches-per-worker 10 --batch-size 4
"""

import argparse
import concurrent.futures
import random
import time

import numpy as np

from molmo_spaces.planner.curobo_planner_client import CuroboClient

# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


def _random_config(lower: list[float], upper: list[float], rng: random.Random) -> list[float]:
    return [rng.uniform(lo, hi) for lo, hi in zip(lower, upper)]


def _precompute_batches(
    clients: dict[str, CuroboClient],
    limits: dict[str, dict],
    num_batches: int,
    batch_size: int,
    seed: int,
) -> list[tuple[str, list, list]]:
    """
    Build (arm, joint_positions, goal_poses) triples for each batch.

    Goal poses are FK of randomly sampled configs so every goal is reachable.
    """
    rng = random.Random(seed)
    batches = []
    for _ in range(num_batches):
        arm = rng.choice(["left", "right"])
        lower = limits[arm]["lower"]
        upper = limits[arm]["upper"]
        starts = [_random_config(lower, upper, rng) for _ in range(batch_size)]
        goal_configs = [_random_config(lower, upper, rng) for _ in range(batch_size)]
        goal_poses = [clients[arm].fk(gc) for gc in goal_configs]
        batches.append((arm, starts, goal_poses))
    return batches


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(
    worker_id: int,
    address: str,
    batches: list[tuple[str, list, list]],
    timeout: float,
) -> list[dict]:
    clients = {
        arm: CuroboClient(address=address, arm=arm, timeout=timeout) for arm in ("left", "right")
    }
    results = []
    for arm, starts, goals in batches:
        t0 = time.perf_counter()
        trajectories, successes = clients[arm].motion_plan_batch(
            joint_positions=starts,
            goal_poses=goals,
        )
        results.append(
            {
                "elapsed_s": time.perf_counter() - t0,
                "success_rate": sum(successes) / len(successes),
                "total_waypoints": sum(len(t) for t in trajectories),
            }
        )
    for c in clients.values():
        c.close()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CuroboPlanner gRPC batch stress test")
    parser.add_argument(
        "--address", default="jupiter-cs-aus-106.reviz.ai2.in:10000", help="gRPC server address"
    )
    parser.add_argument("--workers", type=int, default=10, help="Parallel client threads")
    parser.add_argument("--batches-per-worker", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Queries per batch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-RPC timeout (s)")
    args = parser.parse_args()

    total_plans = args.workers * args.batches_per_worker * args.batch_size

    # ------------------------------------------------------------------
    # Connect and fetch planner metadata
    # ------------------------------------------------------------------
    setup_clients = {
        arm: CuroboClient(address=args.address, arm=arm, timeout=60.0) for arm in ("left", "right")
    }
    assert setup_clients["left"].health(), f"Server not reachable at {args.address}"

    limits = {arm: c.joint_limits() for arm, c in setup_clients.items()}
    dof = len(setup_clients["left"].joint_names())
    print(f"Connected — both arms, DOF={dof}")
    for c in setup_clients.values():
        c.close()

    # ------------------------------------------------------------------
    # Pre-compute batches (FK calls, not timed)
    # ------------------------------------------------------------------
    print(
        f"Pre-computing {args.workers} × {args.batches_per_worker} batches "
        f"of size {args.batch_size} with random arm selection..."
    )
    precompute_clients = {
        arm: CuroboClient(address=args.address, arm=arm, timeout=60.0) for arm in ("left", "right")
    }
    all_batches = [
        _precompute_batches(
            precompute_clients,
            limits,
            args.batches_per_worker,
            args.batch_size,
            seed=args.seed + worker_id,
        )
        for worker_id in range(args.workers)
    ]
    for c in precompute_clients.values():
        c.close()
    print("Pre-computation done.")

    # ------------------------------------------------------------------
    # Run parallel workers
    # ------------------------------------------------------------------
    print(
        f"\nRunning {args.workers} workers × {args.batches_per_worker} batches "
        f"× {args.batch_size} queries = {total_plans} total plans...\n"
    )
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(_worker, i, args.address, all_batches[i], args.timeout)
            for i in range(args.workers)
        ]
        worker_results = [f.result() for f in futs]

    total_elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Stdout summary (client-perceived latency / throughput)
    # ------------------------------------------------------------------
    all_results = [r for wr in worker_results for r in wr]
    latencies = np.array([r["elapsed_s"] for r in all_results])
    success_rates = np.array([r["success_rate"] for r in all_results])
    waypoints = np.array([r["total_waypoints"] for r in all_results])

    print("\n=== Summary ===")
    print(f"  total_elapsed_s:    {total_elapsed:.3f}")
    print(f"  plans_per_second:   {total_plans / total_elapsed:.3f}")
    print(f"  batches_per_second: {len(all_results) / total_elapsed:.3f}")
    print(f"  latency_mean_s:     {latencies.mean():.3f}")
    print(f"  latency_p50_s:      {np.percentile(latencies, 50):.3f}")
    print(f"  latency_p95_s:      {np.percentile(latencies, 95):.3f}")
    print(f"  latency_max_s:      {latencies.max():.3f}")
    print(f"  success_rate_mean:  {success_rates.mean():.3f}")
    print(f"  waypoints_mean:     {waypoints.mean():.3f}")


if __name__ == "__main__":
    main()
