"""Evaluation utilities for MolmoSpaces benchmarks.

Programmatic usage:
    from molmo_spaces.evaluation import run_evaluation

    results = run_evaluation(
        eval_config_cls=MyEvalConfig,
        benchmark_dir="/path/to/benchmark",
        checkpoint_path="/path/to/checkpoint",
    )
    print(f"Success rate: {results.success_rate:.1%}")

See run_evaluation() for full documentation.
"""

from molmo_spaces.evaluation.benchmark_schema import (
    BaseTaskSpec,
    BenchmarkMetadata,
    CameraSpec,
    EpisodeSpec,
    ExocentricCameraSpec,
    LanguageSpec,
    NavToObjTaskSpec,
    OpenCloseTaskSpec,
    PickAndPlaceTaskSpec,
    PickTaskSpec,
    RobotMountedCameraSpec,
    RobotSpec,
    SceneModificationsSpec,
    SourceSpec,
    TaskSpec,
    load_all_episodes,
    load_benchmark,
)
from molmo_spaces.evaluation.eval_main import EvaluationResults, run_evaluation
from molmo_spaces.evaluation.json_eval_runner import JsonEvalRunner

__all__ = [
    # Primary programmatic API
    "run_evaluation",
    "EvaluationResults",
    # Runner
    "JsonEvalRunner",
    # Benchmark schema types
    "BaseTaskSpec",
    "BenchmarkMetadata",
    "CameraSpec",
    "EpisodeSpec",
    "ExocentricCameraSpec",
    "LanguageSpec",
    "NavToObjTaskSpec",
    "OpenCloseTaskSpec",
    "PickAndPlaceTaskSpec",
    "PickTaskSpec",
    "RobotMountedCameraSpec",
    "RobotSpec",
    "SceneModificationsSpec",
    "SourceSpec",
    "TaskSpec",
    # Utility functions
    "load_all_episodes",
    "load_benchmark",
]
