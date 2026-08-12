import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLL_INTERVAL_SEC = 2
TERMINATE_TIMEOUT_SEC = 10


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_file",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_tasks(cfg: DictConfig, xpl_root: Path) -> list[str]:
    if cfg.EVALUATION.task_name is not None and str(cfg.EVALUATION.task_name).strip():
        return [str(cfg.EVALUATION.task_name)]

    if cfg.EVALUATION.task_file is not None and str(cfg.EVALUATION.task_file).strip():
        task_file = _resolve_path(str(cfg.EVALUATION.task_file), base=PROJECT_ROOT)
        return [
            line.strip().split(",", 1)[0]
            for line in task_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    candidate_files = [
        xpl_root.parent / "RoboDojo_official" / "tasks.txt",
        xpl_root / "tasks.txt",
    ]
    for candidate in candidate_files:
        if candidate.exists():
            return [
                line.strip().split(",", 1)[0]
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

    task_config_dir = xpl_root.parent / "task" / "RoboDojo" / "config"
    if task_config_dir.exists():
        tasks = sorted(
            path.stem
            for path in task_config_dir.glob("*.yml")
            if path.stem != "_task"
        )
        if tasks:
            return tasks

    raise FileNotFoundError(
        "No RoboDojo task list found. Pass EVALUATION.task_name=<task> or "
        "EVALUATION.task_file=<tasks.txt>."
    )


def _parse_result_file(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    success = payload.get("success")
    score = payload.get("score")
    if success is not None:
        success = float(success)
    if score is not None:
        score = float(score)
    return success, score


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    port: int
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robodojo.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.xpolicylab_root is None:
        raise ValueError("EVALUATION.xpolicylab_root must point to an XPolicyLab checkout.")

    xpl_root = _resolve_path(str(cfg.EVALUATION.xpolicylab_root), base=PROJECT_ROOT)
    if not xpl_root.exists():
        raise FileNotFoundError(f"XPolicyLab root not found: {xpl_root}")

    robodojo_root = _resolve_path(str(cfg.EVALUATION.robodojo_root), base=PROJECT_ROOT)
    robodojo_assets_root = _resolve_path(str(cfg.EVALUATION.robodojo_assets_root), base=PROJECT_ROOT)
    robodojo_data_root = _resolve_path(str(cfg.EVALUATION.robodojo_data_root), base=PROJECT_ROOT)
    ckpt_root = _resolve_path(str(cfg.EVALUATION.ckpt_root), base=PROJECT_ROOT)

    policy_dir = xpl_root / "policy" / "FastWAM"
    eval_script = policy_dir / "eval.sh"
    if not eval_script.exists():
        raise FileNotFoundError(
            f"FastWAM RoboDojo policy is not linked at {policy_dir}. "
            "Run: ln -sfn <FastWAM>/experiments/robodojo/fastwam_policy "
            f"{policy_dir}"
        )

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("MULTIRUN.num_gpus and MULTIRUN.max_tasks_per_gpu must be positive.")

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    tasks = _load_tasks(cfg, xpl_root)
    (output_dir / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    manager_log = output_dir / "manager.log"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    failed_tasks_file = output_dir / "failed_tasks.txt"
    run_id_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    extra_env = os.environ.copy()
    extra_env.setdefault("FASTWAM_ROOT", str(PROJECT_ROOT))
    extra_env.setdefault("ROBODOJO_ROOT", str(robodojo_root))
    extra_env.setdefault("ROBODOJO_ASSETS_ROOT", str(robodojo_assets_root))
    extra_env.setdefault("ROBODOJO_DATA_ROOT", str(robodojo_data_root))
    extra_env.setdefault("FASTWAM_CKPT_ROOT", str(ckpt_root))
    extra_env.setdefault("EVAL_ENV_TYPE", str(cfg.EVALUATION.get("eval_env_type", "sim")))
    if cfg.EVALUATION.dataset_stats_path is not None:
        extra_env["FASTWAM_DATASET_STATS_PATH"] = str(
            _resolve_path(str(cfg.EVALUATION.dataset_stats_path), base=PROJECT_ROOT)
        )
    for override in _collect_worker_overrides():
        if override.startswith("EVALUATION.dataset_stats_path="):
            extra_env["FASTWAM_DATASET_STATS_PATH"] = override.split("=", 1)[1]

    pending = deque(tasks)
    running: list[RunningState] = []
    completed: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def gpu_running_count(gpu_id: int) -> int:
        return sum(1 for state in running if state.gpu_id == gpu_id and state.process.poll() is None)

    def launch(task_name: str, gpu_id: int) -> RunningState:
        port = int(cfg.EVALUATION.policy_port_base) + gpu_id
        task_run_id = f"{run_id_prefix}_{task_name}_gpu{gpu_id}"
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_dir / f"gpu{gpu_id}.log"
        cmd = [
            "bash",
            str(eval_script),
            str(cfg.EVALUATION.bench_name),
            task_name,
            str(cfg.ckpt),
            str(cfg.EVALUATION.env_cfg_type),
            str(cfg.EVALUATION.action_type),
            str(cfg.EVALUATION.seed),
            str(gpu_id),
            str(gpu_id),
            str(cfg.EVALUATION.policy_conda_env),
            str(cfg.EVALUATION.eval_env_conda_env),
        ]
        env = extra_env.copy()
        env["FASTWAM_POLICY_PORT"] = str(port)
        env["ROBODOJO_RUN_ID"] = task_run_id
        log(f"launch task={task_name} gpu={gpu_id} port={port} run_id={task_run_id}")
        f = log_file.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=str(policy_dir), env=env, text=True, stdout=f, stderr=subprocess.STDOUT)
        return RunningState(task_name=task_name, gpu_id=gpu_id, port=port, process=process)

    def try_launch(gpu_id: int) -> None:
        while pending and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            running.append(launch(pending.popleft(), gpu_id))

    def terminate_all() -> None:
        for state in running:
            if state.process.poll() is None:
                state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in running:
            if state.process.poll() is None:
                try:
                    state.process.wait(timeout=max(0.0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    state.process.kill()

    for gpu_id in range(num_gpus):
        try_launch(gpu_id)

    while running:
        progressed = False
        for state in list(running):
            rc = state.process.poll()
            if rc is None:
                continue
            progressed = True
            running.remove(state)
            if rc != 0:
                failed.append({"task_name": state.task_name, "gpu_id": state.gpu_id, "return_code": rc})
                log(f"failed task={state.task_name} gpu={state.gpu_id} rc={rc}")
            else:
                result_path = (
                    xpl_root.parent
                    / "eval_result"
                    / str(cfg.EVALUATION.bench_name)
                    / state.task_name
                    / "FastWAM"
                    / str(cfg.EVALUATION.env_cfg_type)
                    / f"{cfg.EVALUATION.seed}_ckpt_name={cfg.ckpt},action_type={cfg.EVALUATION.action_type}"
                    / f"{run_id_prefix}_{state.task_name}_gpu{state.gpu_id}"
                    / "_result.json"
                )
                success, score = _parse_result_file(result_path)
                completed[state.task_name] = {
                    "success": success,
                    "score": score,
                    "result_path": str(result_path),
                }
                log(f"done task={state.task_name} gpu={state.gpu_id} success={success} score={score}")
            try_launch(state.gpu_id)
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "success", "score", "result_path"])
        for task in tasks:
            rec = completed.get(task, {})
            writer.writerow([task, rec.get("success"), rec.get("score"), rec.get("result_path")])

    summary_json.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": task,
                        **completed.get(task, {}),
                    }
                    for task in tasks
                ],
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with failed_tasks_file.open("w", encoding="utf-8") as f:
        for rec in failed:
            f.write(f"{rec['task_name']},gpu={rec['gpu_id']},return_code={rec['return_code']}\n")

    log(f"summary saved: {summary_csv} and {summary_json}")
    if failed:
        terminate_all()
        raise RuntimeError(f"{len(failed)} RoboDojo tasks failed; see {failed_tasks_file}")


if __name__ == "__main__":
    main()
