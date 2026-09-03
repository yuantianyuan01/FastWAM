"""Run original LIBERO tasks with one persistent model worker per GPU."""

import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TextIO

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.summarize_results import summarize_results
from experiments.libero.worker_pool import pending_task_count, read_worker_status, requeue_task


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            with redirect_stdout(StringIO()):
                task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"{suite_name}: {n_tasks} tasks")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"Task list created: {output_file} ({total_tasks} tasks)")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
        "EVALUATION.num_trials",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    return [
        override
        for override in HydraConfig.get().overrides.task
        if not _is_blocked_override(override)
    ]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Pass task=..., for example "
            "task=libero_uncond_2cam224_1e-4."
        )
    return str(task_choice)


def _task_result_exists(output_dir: Path, suite_name: str, task_id: int) -> bool:
    return any((output_dir / suite_name).glob(f"gpu*_task{task_id}_results.json"))


def _worker_current_task(status_dir: Path, gpu_id: str) -> tuple[str, int] | None:
    status = read_worker_status(status_dir, gpu_id)
    if status is None or status["status"] != "running":
        return None
    suite_name, task_id = status["message"].split(",", 1)
    return suite_name, int(task_id)


def _completed_task_count(output_dir: Path, tasks: list[tuple[str, int]]) -> int:
    return sum(_task_result_exists(output_dir, suite, task_id) for suite, task_id in tasks)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    output_dir: Path,
    extra_overrides: list[str],
    config_name: str,
) -> None:
    if num_gpus <= 0:
        raise ValueError("MULTIRUN.num_gpus must be positive.")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_task_file = task_file.resolve()
    output_task_file = (output_dir / source_task_file.name).resolve()
    if source_task_file != output_task_file:
        shutil.copy2(source_task_file, output_task_file)
    task_file = output_task_file

    task_lines = [
        line.strip()
        for line in task_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = []
    for task_line in task_lines:
        suite_name, task_id = task_line.split(",", 1)
        tasks.append((suite_name, int(task_id)))
    if not tasks:
        raise ValueError(f"Task file is empty: {task_file}")

    worker_root = output_dir / ".worker_pool"
    pending_file = worker_root / "pending_tasks.txt"
    lock_file = worker_root / "pending_tasks.lock"
    stop_file = worker_root / "stop"
    status_dir = worker_root / "status"
    log_dir = worker_root / "logs"
    failed_file = worker_root / "failed_tasks.txt"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending_file.write_text("\n".join(task_lines) + "\n", encoding="utf-8")
    failed_file.write_text("", encoding="utf-8")
    stop_file.unlink(missing_ok=True)
    lock_dir = lock_file.with_name(f"{lock_file.name}.lockdir")
    if lock_dir.exists():
        lock_dir.rmdir()
    for status_file in status_dir.glob("worker_*.json"):
        status_file.unlink()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = [
            gpu.strip()
            for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if gpu.strip()
        ]
        if len(visible_gpus) < num_gpus:
            raise ValueError(
                f"MULTIRUN.num_gpus={num_gpus}, but CUDA_VISIBLE_DEVICES only has "
                f"{len(visible_gpus)} entries."
            )
        gpu_ids = visible_gpus[:num_gpus]
    else:
        gpu_ids = [str(gpu_id) for gpu_id in range(num_gpus)]

    root_dir = str(project_root)
    extra_args_display = shlex.join(extra_overrides) if extra_overrides else ""
    base_env = os.environ.copy()

    print("\nStarting persistent LIBERO worker pool...")
    print(f"task: {task_choice}")
    print(f"checkpoint: {ckpt}")
    print(f"GPUs: {','.join(gpu_ids)}")
    print(f"trials per task: {num_trials}")
    print(f"task list: {task_file}")
    print(f"output: {output_dir}")
    if extra_args_display:
        print(f"forwarded overrides: {extra_args_display}")
    print("recovery: quarantine a crashed GPU and requeue its unfinished task")

    workers: dict[str, tuple[subprocess.Popen, Path, TextIO]] = {}
    for gpu_id in gpu_ids:
        log_file = log_dir / f"worker_{gpu_id}.log"
        worker_env = base_env.copy()
        worker_env.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "LIBERO_WORKER_MODE": "1",
                "LIBERO_WORKER_ID": gpu_id,
                "LIBERO_WORKER_PENDING_FILE": str(pending_file),
                "LIBERO_WORKER_LOCK_FILE": str(lock_file),
                "LIBERO_WORKER_STOP_FILE": str(stop_file),
                "LIBERO_WORKER_STATUS_DIR": str(status_dir),
                "LIBERO_WORKER_FAILED_FILE": str(failed_file),
            }
        )
        cmd = [
            sys.executable,
            "experiments/libero/eval_libero_single.py",
            "--config-name",
            config_name,
            f"task={task_choice}",
            f"ckpt={ckpt}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.num_trials={num_trials}",
            f"EVALUATION.output_dir={output_dir}",
            *extra_overrides,
        ]
        log_handle = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=root_dir,
            env=worker_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        workers[gpu_id] = (process, log_file, log_handle)
        print(f"started worker GPU{gpu_id}: {log_file}")

    status_interval = int(os.environ.get("STATUS_INTERVAL", "60"))
    monitor_interval = int(os.environ.get("MONITORING_INTERVAL", "5"))
    last_status_time = 0.0
    terminal_error = None

    try:
        while workers:
            for gpu_id, (process, log_file, log_handle) in list(workers.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_handle.close()
                del workers[gpu_id]

                if returncode == 0 and stop_file.exists():
                    continue

                current_task = _worker_current_task(status_dir, gpu_id)
                if current_task is not None and not _task_result_exists(
                    output_dir, current_task[0], current_task[1]
                ):
                    if requeue_task(pending_file, lock_file, current_task):
                        print(f"requeued GPU{gpu_id} task: {current_task[0]},{current_task[1]}")

                if failed_file.stat().st_size > 0:
                    terminal_error = f"worker GPU{gpu_id} reported a task error; log: {log_file}"
                    break
                print(
                    f"worker GPU{gpu_id} exited with code {returncode}; "
                    f"GPU{gpu_id} is quarantined for this run. Log: {log_file}"
                )

            if terminal_error is not None:
                break

            pending = pending_task_count(pending_file, lock_file)
            if pending == 0 and workers:
                statuses = [read_worker_status(status_dir, gpu_id) for gpu_id in workers]
                if all(status is not None and status["status"] == "idle" for status in statuses):
                    stop_file.write_text("done\n", encoding="utf-8")

            now = time.time()
            if now - last_status_time >= status_interval:
                completed = _completed_task_count(output_dir, tasks)
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"completed={completed}/{len(tasks)} pending={pending} "
                    f"active_workers={len(workers)}"
                )
                last_status_time = now

            if workers:
                time.sleep(monitor_interval)
    finally:
        if terminal_error is not None or not stop_file.exists():
            for process, _, _ in workers.values():
                process.terminate()
        for process, _, log_handle in workers.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if not log_handle.closed:
                log_handle.close()

    if terminal_error is not None:
        print(failed_file.read_text(encoding="utf-8"))
        raise RuntimeError(terminal_error)

    pending = pending_task_count(pending_file, lock_file)
    completed = _completed_task_count(output_dir, tasks)
    if pending > 0 or completed < len(tasks):
        raise RuntimeError(
            f"LIBERO evaluation incomplete: completed={completed}, pending={pending}, total={len(tasks)}"
        )

    print("All workers exited; generating summary...")
    summarize_results(str(output_dir), ckpt=ckpt, config=task_choice)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN
    output_dir = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
        if not task_file.exists():
            task_file = create_task_file(task_file, list(manager.task_suite_names))
        else:
            print(f"Using existing task list: {task_file}")
    else:
        task_file = create_task_file(output_dir / "tasks.txt", list(manager.task_suite_names))

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))
    if bool(manager.get("create_only", False)):
        print("create_only=True; task list generated without starting workers.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
        config_name=HydraConfig.get().job.config_name,
    )


if __name__ == "__main__":
    main()
