import argparse
import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "VLABench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")

from experiments.vlabench.fastwam_policy import FastWAMVLABenchPolicy
from VLABench.evaluation.evaluator import Evaluator
from VLABench.tasks import *  # noqa: F401,F403 - VLABench registers tasks through imports
from VLABench.robots import *  # noqa: F401,F403 - VLABench registers robots through imports


def _resolve(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _load_tasks(vlabench_root: Path, task_name: str | None, eval_track: str | None) -> tuple[list[str], dict | None]:
    if task_name:
        return [task_name], None
    if eval_track:
        track_path = vlabench_root / "VLABench" / "configs" / "evaluation" / "tracks" / f"{eval_track}.json"
        with track_path.open("r", encoding="utf-8") as f:
            episode_config = json.load(f)
        return list(episode_config.keys()), episode_config
    return [
        "select_painting",
        "select_book",
        "select_drink",
        "select_chemistry_tube",
        "select_poker",
        "select_mahjong",
        "select_toy",
        "select_fruit",
        "add_condiment",
        "insert_flower",
    ], None


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_vlabench.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must point to a FastWAM checkpoint file.")
    if cfg.EVALUATION.dataset_stats_path is None:
        raise ValueError("EVALUATION.dataset_stats_path must point to dataset_stats.json.")

    vlabench_root = _resolve(cfg.EVALUATION.vlabench_root)
    assets_root = _resolve(cfg.EVALUATION.vlabench_assets_root)
    output_dir = _resolve(cfg.EVALUATION.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    os.environ.setdefault("VLABENCH_ROOT", str(vlabench_root / "VLABench"))
    os.environ.setdefault("VLABENCH_ASSETS_ROOT", str(assets_root))

    eval_track = cfg.EVALUATION.get("eval_track")
    tasks, episode_config = _load_tasks(
        vlabench_root=vlabench_root,
        task_name=cfg.EVALUATION.task_name,
        eval_track=eval_track,
    )

    policy = FastWAMVLABenchPolicy(
        checkpoint_path=str(_resolve(cfg.ckpt)),
        dataset_stats_path=str(_resolve(cfg.EVALUATION.dataset_stats_path)),
        action_horizon=int(cfg.EVALUATION.action_horizon),
        replan_steps=int(cfg.EVALUATION.replan_steps),
        num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
        device=str(cfg.EVALUATION.device),
    )
    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=int(cfg.EVALUATION.n_episode),
        episode_config=episode_config,
        max_substeps=1,
        save_dir=str(output_dir),
        visulization=bool(cfg.EVALUATION.save_video),
        metrics=list(cfg.EVALUATION.metrics),
    )
    result = evaluator.evaluate(policy)
    result_dir = output_dir / "FastWAM" / str(eval_track or "tasks")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "evaluation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
