"""Benchmark utilities for kinematics outlier detection in trajectory H5 files.

NOTE ON TEMPORAL INDEXING: All sensor arrays (qpos, cmd, jpr) share the same
index space, but within a step() the controller target is set before physics
runs, so qpos[t] only partially converges toward cmd[t] (~30% per step).
We therefore use qpos[t+1] instead of qpos[t] to measure tracking quality::

    tracking_error[t]          = qpos[t+1] - cmd[t]
    relative_tracking_error[t] = (qpos[t+1] - cmd[t]) / jpr[t]

NOTE ON COLLISION DETECTION: The better way to detect collisions would be via
residual torques (measured minus expected from rigid-body dynamics), but no
torque sensors are currently recorded. Tracking error is a reasonable proxy
--collisions cause position deviations-- though less sensitive than torques.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Collection
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import sigmaclip
from tqdm import tqdm

log = logging.getLogger(__name__)


def resolve_asset_id(
    object_name: str,
    task_config,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    house_index: int | None = None,
) -> str | None:
    """Resolve the asset ID (UID) for a task object by name.

    Tries two strategies in order:

    1. **added_objects** (frozen config / JSON benchmark): If the object was
       dynamically added to the scene (e.g. a place receptacle), its XML path
       is stored in ``task_config.added_objects``. The UID is the stem of the
       XML filename (``<uid>.xml``).

    2. **Scene metadata** (via SceneMeta): For objects that are part of the
       base scene (e.g. pickup objects), the asset_id is looked up from the
       scene's ``*_metadata.json`` using ``scene_dataset``, ``data_split``,
       and ``house_index``.

    Args:
        object_name: MuJoCo body name of the object (e.g. ``"Mug_12"``
            or ``"place_receptacle/Bowl_25"``).
        task_config: A task config object (e.g. ``PickAndPlaceTaskConfig``)
            that has an ``added_objects`` dict mapping object names to
            relative XML paths.
        scene_dataset: Scene dataset name (e.g. ``"procthor-objaverse"``).
            Required for the SceneMeta fallback.
        data_split: Data split (e.g. ``"val"``). Required for the SceneMeta
            fallback.
        house_index: House index within the dataset/split. Required for the
            SceneMeta fallback.

    Returns:
        The asset UID string, or ``None`` if it could not be resolved.
    """
    # Strategy 1: look up in added_objects (dynamically added assets like receptacles)
    if isinstance(task_config, dict):
        added_objects = task_config.get("added_objects") or {}
    else:
        added_objects = getattr(task_config, "added_objects", None) or {}
    if object_name in added_objects:
        xml_rel_path = added_objects[object_name]
        uid = Path(xml_rel_path).stem
        return uid

    # Strategy 2: look up in scene metadata via SceneMeta
    if scene_dataset is not None and data_split is not None and house_index is not None:
        from molmo_spaces.molmo_spaces_constants import get_scenes
        from molmo_spaces.utils.lazy_loading_utils import install_scene_from_path
        from molmo_spaces.utils.scene_metadata_utils import SceneMeta

        scene_source = get_scenes(scene_dataset, data_split)[data_split][house_index]
        if isinstance(scene_source, dict):
            scene_source = scene_source["ceiling"]
        assert scene_source.endswith(".xml")
        install_scene_from_path(scene_source)

        scene_meta = SceneMeta.get_scene_metadata(scene_source)
        if scene_meta is not None:
            asset_id = scene_meta.get("objects", {}).get(object_name, {}).get("asset_id")
            if asset_id is not None:
                return asset_id

    log.warning(f"Could not resolve asset_id for '{object_name}'")
    return None


# THOR object category simplifications for diversity analysis
THOR_CAT_SIMPLIFY = {
    "saltshaker": "s/p shaker",
    "peppershaker": "s/p shaker",
    "tomato": "fruit",
    "apple": "fruit",
    "butterknife": "knife",
    "boiler": "kettle",
    "winebottle": "bottle",
    "atomizer": "spray bottle",
    "remotecontrol": "remote control",
    "soapdispenser": "soap dispenser",
    "tissuepaper": "tissue paper",
}


def compute_bounds_std(
    stats: dict,
    std_mult: float,
) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    bounds = {}
    for key, s in stats.items():
        if s["count"] < 10:
            continue

        mean = s["mean"]
        variance = s["M2"] / s["count"]
        std = np.sqrt(variance)

        lower = mean - std_mult * std
        upper = mean + std_mult * std

        bounds[key] = (lower, upper, mean, std)

    return bounds


def _sigma_clip(
    values: np.ndarray,
    clip_sigma: float = 4.0,
) -> tuple[float, float, int]:
    """Iterative sigma clipping via :func:`scipy.stats.sigmaclip`.

    Returns ``(mean, std, n_kept)`` after convergence.
    """
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return 0.0, 0.0, 0
    clipped, _, _ = sigmaclip(vals, low=clip_sigma, high=clip_sigma)
    mu = float(np.mean(clipped))
    sigma = float(np.std(clipped))
    return mu, sigma, len(clipped)


def _safe_decode_json_array(dataset) -> list[dict] | None:
    """Decode an H5 dataset of JSON-encoded byte rows into a list of dicts.

    Returns None when the dataset cannot be decoded (e.g. numeric arrays).
    """
    try:
        data = dataset[:]
        results = []
        for i in range(data.shape[0]):
            json_str = data[i, :].tobytes().decode("utf-8").rstrip("\x00")
            try:
                results.append(json.loads(json_str) if json_str else {})
            except json.JSONDecodeError:
                results.append({})
        return results
    except Exception:
        return None


def _collect_stats_worker(args) -> dict[tuple[str, int], list[float]]:
    """Collect raw relative tracking error values from one H5 file.

    Returns a dict mapping ``(ag, dim)`` to a list of raw ratio values
    to be sigma-clipped centrally after merging.
    """
    (
        h5_file_str,
        action_groups,
        skip_first,
        relative_tracking_error_eps,
    ) = args
    reltrack_raw: dict[tuple[str, int], list[float]] = defaultdict(list)

    h5_file = Path(h5_file_str)

    try:
        with h5py.File(h5_file, "r") as f:
            for traj_key in [k for k in f if k.startswith("traj_")]:
                traj_data = f[traj_key]
                if "actions" not in traj_data:
                    continue
                actions_grp = traj_data["actions"]

                # Resolve qpos location: obs/agent/qpos or fallback to top-level qpos
                qpos_ds = None
                if (
                    "obs" in traj_data
                    and "agent" in traj_data["obs"]
                    and "qpos" in traj_data["obs"]["agent"]
                ):
                    qpos_ds = traj_data["obs"]["agent"]["qpos"]
                elif "qpos" in traj_data:
                    qpos_ds = traj_data["qpos"]

                jpr_all = (
                    _safe_decode_json_array(actions_grp["joint_pos_rel"])
                    if "joint_pos_rel" in actions_grp
                    else None
                )
                cmd_all = (
                    _safe_decode_json_array(actions_grp["joint_pos"])
                    if "joint_pos" in actions_grp
                    else None
                )
                actual_all = _safe_decode_json_array(qpos_ds) if qpos_ds is not None else None

                # --- relative tracking error: (qpos[t+1] - cmd[t]) / jpr[t] ---
                # See NOTE ON TEMPORAL INDEXING in module docstring.
                if jpr_all is not None and cmd_all is not None and actual_all is not None:
                    n = min(len(jpr_all), len(cmd_all), len(actual_all) - 1)
                    for t in range(n):
                        if t < skip_first:
                            continue
                        jpr, cmd, act = jpr_all[t], cmd_all[t], actual_all[t + 1]
                        for ag in action_groups:
                            if (
                                ag in jpr
                                and jpr[ag]
                                and ag in cmd
                                and cmd[ag]
                                and ag in act
                                and act[ag]
                            ):
                                jpr_vals = np.array(jpr[ag])
                                cmd_vals = np.array(cmd[ag])
                                act_vals = np.array(act[ag])
                                if jpr_vals.shape == cmd_vals.shape == act_vals.shape:
                                    track_err = act_vals - cmd_vals
                                    for dim_idx in range(len(jpr_vals)):
                                        denom = abs(jpr_vals[dim_idx])
                                        if denom < relative_tracking_error_eps:
                                            continue
                                        ratio = track_err[dim_idx] / jpr_vals[dim_idx]
                                        reltrack_raw[(f"reltrack_{ag}", dim_idx)].append(
                                            float(ratio)
                                        )
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    return dict(reltrack_raw)


def _check_bounds_and_append(
    outliers: list[dict],
    bounds: dict,
    house_name: str,
    h5_file: str,
    traj_idx: int,
    t: int,
    ag: str,
    dim_idx: int,
    val: float,
    lower_only: bool = False,
):
    """Check a single value against bounds and append to outliers if exceeded.

    If *lower_only* is True, only the lower bound is enforced (values above
    the upper bound are silently accepted).
    """
    key = (ag, dim_idx)
    if key not in bounds:
        return
    lower, upper, mean, std = bounds[key]
    if val < lower or (not lower_only and val > upper):
        std_away = abs(val - mean) / std if std > 0 else 0
        outliers.append(
            {
                "house": house_name,
                "h5_file": h5_file,
                "traj_idx": traj_idx,
                "timestep": t,
                "action_group": ag,
                "dim_idx": dim_idx,
                "action_dim": f"{ag}[{dim_idx}]",
                "value": val,
                "mean": mean,
                "std": std,
                "std_away": std_away,
            }
        )


def _find_outliers_worker(args) -> list[dict]:
    (
        h5_file_str,
        action_groups,
        skip_first,
        bounds,
        relative_tracking_error_eps,
        reltrack_negative_only,
    ) = args
    outliers = []
    h5_file = Path(h5_file_str)
    house_dir = h5_file.parent
    house_name = house_dir.name

    try:
        with h5py.File(h5_file, "r") as f:
            for traj_key in [k for k in f if k.startswith("traj_")]:
                traj_idx = int(traj_key.split("_")[1])
                traj_data = f[traj_key]
                if "actions" not in traj_data:
                    continue
                actions_grp = traj_data["actions"]

                # Resolve qpos location: obs/agent/qpos or fallback to top-level qpos
                qpos_ds = None
                if (
                    "obs" in traj_data
                    and "agent" in traj_data["obs"]
                    and "qpos" in traj_data["obs"]["agent"]
                ):
                    qpos_ds = traj_data["obs"]["agent"]["qpos"]
                elif "qpos" in traj_data:
                    qpos_ds = traj_data["qpos"]

                jpr_all = (
                    _safe_decode_json_array(actions_grp["joint_pos_rel"])
                    if "joint_pos_rel" in actions_grp
                    else None
                )
                cmd_all = (
                    _safe_decode_json_array(actions_grp["joint_pos"])
                    if "joint_pos" in actions_grp
                    else None
                )
                actual_all = _safe_decode_json_array(qpos_ds) if qpos_ds is not None else None

                # --- relative tracking error: (qpos[t+1] - cmd[t]) / jpr[t] ---
                # See NOTE ON TEMPORAL INDEXING in module docstring.
                if jpr_all is not None and cmd_all is not None and actual_all is not None:
                    n = min(len(jpr_all), len(cmd_all), len(actual_all) - 1)
                    for t in range(n):
                        if t < skip_first:
                            continue
                        jpr, cmd, act = jpr_all[t], cmd_all[t], actual_all[t + 1]
                        for ag in action_groups:
                            if (
                                ag in jpr
                                and jpr[ag]
                                and ag in cmd
                                and cmd[ag]
                                and ag in act
                                and act[ag]
                            ):
                                jpr_vals = np.array(jpr[ag])
                                cmd_vals = np.array(cmd[ag])
                                act_vals = np.array(act[ag])
                                if jpr_vals.shape == cmd_vals.shape == act_vals.shape:
                                    track_err = act_vals - cmd_vals
                                    for dim_idx in range(len(jpr_vals)):
                                        denom = abs(jpr_vals[dim_idx])
                                        if denom < relative_tracking_error_eps:
                                            continue
                                        ratio = track_err[dim_idx] / jpr_vals[dim_idx]
                                        _check_bounds_and_append(
                                            outliers,
                                            bounds,
                                            house_name,
                                            str(h5_file),
                                            traj_idx,
                                            t,
                                            f"reltrack_{ag}",
                                            dim_idx,
                                            float(ratio),
                                            lower_only=reltrack_negative_only,
                                        )
    except KeyboardInterrupt:
        raise
    except Exception:
        pass

    return outliers


def episodes_with_kinematics_outliers(
    data_path: Path | str,
    max_files: int | None = None,
    num_workers: int = 32,
    std_mult: float = 8.0,  # to determine outlier
    action_groups: Collection[str] = ("arm",),
    skip_first_n_steps: int = 1,
    min_joint_pos_rel_magnitude: float = 1.5e-2,
    std_mult_clip_sigma: float = 4.0,
    std_mult_negative_only: bool = True,
    print_stats: bool = False,
) -> tuple[list[dict], dict]:
    """Find episodes with kinematics outliers across H5 trajectory files.

    Uses relative tracking error ``(qpos[t+1] - cmd[t]) / jpr[t]`` as the
    outlier signal (see NOTE ON TEMPORAL INDEXING in module docstring).

    Args:
        data_path: Root directory containing house_*/trajectories*.h5 files.
        max_files: Cap on the number of H5 files to process (for testing).
        num_workers: Parallel workers for stats collection and outlier detection.
        std_mult: Number of standard deviations beyond which a value is an outlier.
        action_groups: Which move-group keys to examine (e.g. ``("arm",)``).
        skip_first_n_steps: Ignore the first N timesteps of each trajectory.
        min_joint_pos_rel_magnitude: Minimum absolute ``joint_pos_rel`` value
            (in radians) below which the sample is discarded.  Small
            commanded deltas with possibly high undershoot
            are assumed to occur during the grasping approach.
            Choose this threshold so that the grasping mode
            fades out, leaving only the free-space motion regime.
        std_mult_clip_sigma: Number of sigmas for iterative sigma-clipping
            to robustly estimate the spread.
        std_mult_negative_only: When True, only flag values below the lower
            bound (undershooting).  Positive overshooting is accepted.
        print_stats: Print per-dimension statistics.

    Returns:
        Tuple of ``(outlier_episodes, bounds)``.
    """
    data_path = Path(data_path)

    h5_files = list(data_path.glob("house_*/trajectories*.h5"))
    if max_files:
        h5_files = h5_files[:max_files]

    print(f"Found {len(h5_files)} H5 files")
    print(f"Using {num_workers} workers")
    print(f"Outlier threshold: mean {'±' if not std_mult_negative_only else '-'} {std_mult} * std")

    print("\n[Pass 1] Computing stats...")

    worker_args = [
        (
            str(f),
            action_groups,
            skip_first_n_steps,
            min_joint_pos_rel_magnitude,
        )
        for f in h5_files
    ]

    all_reltrack_raw: dict[tuple[str, int], list[float]] = defaultdict(list)
    with Pool(num_workers) as pool:
        for reltrack_raw in tqdm(
            pool.imap_unordered(_collect_stats_worker, worker_args, chunksize=20),
            total=len(worker_args),
            desc="Stats",
        ):
            for key, vals in reltrack_raw.items():
                all_reltrack_raw[key].extend(vals)
    merged_stats: dict = {}

    if all_reltrack_raw:
        print(f"\n  Sigma-clipping relative tracking error (clip={std_mult_clip_sigma}σ) ...")
        for key, raw_vals in sorted(all_reltrack_raw.items()):
            clipped_mean, clipped_std, n_kept = _sigma_clip(
                np.array(raw_vals), clip_sigma=std_mult_clip_sigma
            )
            merged_stats[key] = {
                "count": n_kept,
                "mean": clipped_mean,
                "M2": clipped_std**2 * n_kept,
            }
            ag, dim = key
            print(
                f"    {ag}[{dim}]: {len(raw_vals)} total → {n_kept} after clip, "
                f"mean = {clipped_mean:.6f}, std = {clipped_std:.6f}"
            )

    bounds = compute_bounds_std(merged_stats, std_mult)

    if print_stats:
        header = (
            f"{'Dimension':<25} {'Mean':>12} {'Std':>12} {'Lower':>12} {'Upper':>12} {'Count':>12}"
        )
        section_keys = sorted((k, v) for k, v in bounds.items())
        if section_keys:
            print(f"\n  [relative_tracking_error]  ({len(section_keys)} dims)")
            print(f"  {header}")
            print(f"  {'-' * 87}")
            for key, (lower, upper, mean, std) in section_keys:
                ag, dim = key
                count = merged_stats[key]["count"]
                print(
                    f"  {f'{ag}[{dim}]':<25}"
                    f"{mean:>12.6f} {std:>12.6f} {lower:>12.6f} {upper:>12.6f} {count:>12}"
                )

    print("\n[Pass 2] Finding outliers...")

    worker_args = [
        (
            str(f),
            action_groups,
            skip_first_n_steps,
            bounds,
            min_joint_pos_rel_magnitude,
            std_mult_negative_only,
        )
        for f in h5_files
    ]
    all_outliers = []
    with Pool(num_workers) as pool:
        for outliers in tqdm(
            pool.imap_unordered(_find_outliers_worker, worker_args, chunksize=20),
            total=len(worker_args),
            desc="Outliers",
        ):
            all_outliers.extend(outliers)

    print(f"\nFound {len(all_outliers)} outlier values (>{std_mult} std from mean)")

    for d in all_outliers:
        d["body_part"] = d["action_group"].removeprefix("reltrack_")

    # Group by episode
    outlier_episodes: dict = defaultdict(
        lambda: {
            "timesteps": [],
            "body_parts": [],
            "dims": [],
            "values": [],
            "std_aways": [],
        }
    )

    for d in all_outliers:
        episode_key = (d["house"], d["h5_file"], d["traj_idx"])
        outlier_episodes[episode_key]["timesteps"].append(d["timestep"])
        outlier_episodes[episode_key]["body_parts"].append(d["body_part"])
        outlier_episodes[episode_key]["dims"].append(d["action_dim"])
        outlier_episodes[episode_key]["values"].append(d["value"])
        outlier_episodes[episode_key]["std_aways"].append(d["std_away"])

    for key, episode_dict in outlier_episodes.items():
        episode_dict["house"] = key[0]
        episode_dict["h5_file"] = key[1]
        episode_dict["traj_idx"] = key[2]

    sorted_episodes = sorted(
        outlier_episodes.values(), key=lambda x: len(x["timesteps"]), reverse=True
    )
    return sorted_episodes, bounds


def _collect_values_worker(args) -> dict[tuple[str, int], list[float]]:
    """Collect raw relative tracking error values from one H5 file for histograms.

    Collects ALL ratios (skipping only exact-zero denominators) plus
    absolute denominators so the caller can re-filter at various eps
    thresholds.
    """
    (
        h5_file_str,
        action_groups,
        skip_first,
    ) = args
    values: dict[tuple[str, int], list[float]] = defaultdict(list)

    h5_file = Path(h5_file_str)

    try:
        with h5py.File(h5_file, "r") as f:
            for traj_key in [k for k in f if k.startswith("traj_")]:
                traj_data = f[traj_key]
                if "actions" not in traj_data:
                    continue
                actions_grp = traj_data["actions"]

                qpos_ds = None
                if (
                    "obs" in traj_data
                    and "agent" in traj_data["obs"]
                    and "qpos" in traj_data["obs"]["agent"]
                ):
                    qpos_ds = traj_data["obs"]["agent"]["qpos"]
                elif "qpos" in traj_data:
                    qpos_ds = traj_data["qpos"]

                jpr_all = (
                    _safe_decode_json_array(actions_grp["joint_pos_rel"])
                    if "joint_pos_rel" in actions_grp
                    else None
                )
                cmd_all = (
                    _safe_decode_json_array(actions_grp["joint_pos"])
                    if "joint_pos" in actions_grp
                    else None
                )
                actual_all = _safe_decode_json_array(qpos_ds) if qpos_ds is not None else None

                if jpr_all is not None and cmd_all is not None and actual_all is not None:
                    n = min(len(jpr_all), len(cmd_all), len(actual_all) - 1)
                    for t in range(n):
                        if t < skip_first:
                            continue
                        jpr, cmd, act = jpr_all[t], cmd_all[t], actual_all[t + 1]
                        for ag in action_groups:
                            if (
                                ag in jpr
                                and jpr[ag]
                                and ag in cmd
                                and cmd[ag]
                                and ag in act
                                and act[ag]
                            ):
                                jpr_vals = np.array(jpr[ag])
                                cmd_vals = np.array(cmd[ag])
                                act_vals = np.array(act[ag])
                                if jpr_vals.shape == cmd_vals.shape == act_vals.shape:
                                    track_err = act_vals - cmd_vals
                                    for dim_idx in range(len(jpr_vals)):
                                        denom = abs(jpr_vals[dim_idx])
                                        if denom == 0:
                                            continue
                                        ratio = track_err[dim_idx] / jpr_vals[dim_idx]
                                        values[(f"reltrack_{ag}", dim_idx)].append(float(ratio))
                                        values[(f"reltrack_denom_{ag}", dim_idx)].append(
                                            float(denom)
                                        )
                                        values[(f"reltrack_jpr_{ag}", dim_idx)].append(
                                            float(jpr_vals[dim_idx])
                                        )
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    return dict(values)


def save_signal_histograms(
    data_path: Path | str,
    output_dir: Path | str,
    bounds: dict[tuple[str, int], tuple[float, float, float, float]] | None = None,
    action_groups: Collection[str] = ("arm",),
    skip_first_n_steps: int = 1,
    max_files: int | None = None,
    num_workers: int = 32,
    min_joint_pos_rel_magnitude: float = 1.5e-2,  # TODO ensure it matches the one in episodes_with_kinematics_outliers
) -> None:
    """Collect raw relative tracking error values and save histograms as PNGs.

    Generates one figure per eps threshold level, each with one subplot per
    joint dimension.  If *bounds* is provided the outlier thresholds are
    drawn as vertical lines.

    Also generates two joint-histogram (contour) plots of ``joint_pos_rel``
    vs relative tracking error: one with a tight x-range and one with a
    wide range showing the chosen *min_joint_pos_rel_magnitude* threshold.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(data_path.glob("house_*/trajectories*.h5"))
    if max_files:
        h5_files = h5_files[:max_files]

    print(f"\n[Histogram pass] Collecting raw values from {len(h5_files)} H5 files ...")
    worker_args = [
        (
            str(f),
            action_groups,
            skip_first_n_steps,
        )
        for f in h5_files
    ]

    merged_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    with Pool(num_workers) as pool:
        for wv in tqdm(
            pool.imap_unordered(_collect_values_worker, worker_args, chunksize=20),
            total=len(worker_args),
            desc="Values",
        ):
            for key, vals in wv.items():
                merged_values[key].extend(vals)

    # Group keys by dimension (exclude helper keys)
    dim_map: dict[int, tuple[str, int]] = {}
    for ag, dim in sorted(merged_values.keys()):
        if ag.startswith(("reltrack_denom_", "reltrack_jpr_")):
            continue
        dim_map[dim] = (ag, dim)

    n_dims = len(dim_map)
    if n_dims == 0:
        print("No relative tracking error data found for histograms.")
        return

    reltrack_eps_levels = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1.5e-2, 3e-2, 6e-2, 8e-2, 1e-1]

    for hit, eps in enumerate(reltrack_eps_levels):
        ncols = min(n_dims, 4)
        nrows = (n_dims + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(
            f"Relative tracking error  (min |cmd_delta| ≥ {eps:.1e})",
            fontsize=14,
            fontweight="bold",
        )

        for idx, (_dim, (ag, dim_idx)) in enumerate(sorted(dim_map.items())):
            ax = axes[idx // ncols][idx % ncols]
            ratios = np.array(merged_values.get((ag, dim_idx), []))
            denoms = np.array(
                merged_values.get((f"reltrack_denom_{ag.removeprefix('reltrack_')}", dim_idx), [])
            )

            if len(ratios) == 0 or len(denoms) == 0:
                ax.set_title(f"{ag}[{dim_idx}] (no data)")
                continue

            mask_eps = denoms >= eps
            filtered = ratios[mask_eps]
            n_total = len(filtered)

            if n_total == 0:
                ax.set_title(f"{ag}[{dim_idx}] (0 samples at eps={eps:.1e})")
                continue

            key = (ag, dim_idx)
            if bounds and key in bounds:
                _, _, b_mean, b_std = bounds[key]
                plot_lo = b_mean - 10.0 * b_std
                plot_hi = b_mean + 10.0 * b_std
            else:
                plot_lo, plot_hi = -1.5, 1.5

            mask_range = (filtered >= plot_lo) & (filtered <= plot_hi)
            n_clipped = int(np.sum(~mask_range))
            plot_vals = filtered[mask_range]

            n_bins = max(10, int(np.sqrt(len(plot_vals)))) if len(plot_vals) > 0 else 10
            ax.hist(
                plot_vals,
                bins=n_bins,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            ax.set_xlim(plot_lo, plot_hi)

            if bounds and key in bounds:
                lower, upper, mean, std = bounds[key]
                ax.axvline(mean, color="green", linestyle="-", linewidth=1.2, label="mean")
                ax.axvline(lower, color="red", linestyle="--", linewidth=1.2, label="lower")
                ax.axvline(upper, color="red", linestyle="--", linewidth=1.2, label="upper")
                ax.legend(fontsize=7)
                title = f"{ag}[{dim_idx}]  (N={n_total:,}) {std=:.3f}"
            else:
                title = f"{ag}[{dim_idx}]  (N={n_total:,})"
            if n_clipped:
                title += f"  [{n_clipped} outside plot range]"
            ax.set_title(title, fontsize=9)

        for idx in range(n_dims, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        eps_label = f"{eps:.1e}".replace("+", "").replace("-", "m")
        pdf_path = output_dir / f"{hit}_histogram_relative_tracking_error_eps{eps_label}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved histogram: {pdf_path}")

    # --- Joint histograms (contour) of jpr vs relative tracking error ---
    jpr_floor = 5e-6

    # Precompute per-dimension data (shared by both plots)
    contour_data: dict[int, tuple[str, int, np.ndarray, np.ndarray]] = {}
    for dim, (ag, dim_idx) in sorted(dim_map.items()):
        ratios = np.array(merged_values.get((ag, dim_idx), []))
        jpr_signed = np.array(
            merged_values.get((f"reltrack_jpr_{ag.removeprefix('reltrack_')}", dim_idx), [])
        )
        if len(ratios) == 0 or len(jpr_signed) == 0:
            continue
        mask = np.abs(jpr_signed) >= jpr_floor
        x, y = jpr_signed[mask], ratios[mask]
        if len(x) > 0:
            contour_data[dim] = (ag, dim_idx, x, y)

    contour_configs = [
        {
            "suffix": "tight",
            "title": "Joint histogram: jpr vs reltrack (tight)",
            "x_range": lambda x: (-0.01, 0.01),
            "show_threshold": False,
        },
        {
            "suffix": "wide",
            "title": "Joint histogram: jpr vs reltrack (wide)",
            "x_range": lambda x: (np.percentile(x, 0.25), np.percentile(x, 99.75)),
            "show_threshold": True,
        },
    ]

    for cfg in contour_configs:
        ncols = min(n_dims, 4)
        nrows = (n_dims + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(cfg["title"], fontsize=14, fontweight="bold")

        for idx, (dim, (ag, dim_idx)) in enumerate(sorted(dim_map.items())):
            ax = axes[idx // ncols][idx % ncols]
            if dim not in contour_data:
                ax.set_title(f"{ag}[{dim_idx}] (no data)")
                continue

            _, _, x, y = contour_data[dim]
            x_lo, x_hi = cfg["x_range"](x)

            n_bins_2d = max(10, int(np.sqrt(np.sqrt(len(x)))))
            x_edges = np.linspace(x_lo, x_hi, n_bins_2d + 1)
            y_edges = np.linspace(-1.5, 1.5, n_bins_2d + 1)

            hist2d, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            max_val = hist2d.max()
            if max_val > 0:
                levels = np.linspace(max_val * 0.05, max_val, 12)
                cs = ax.contour(xc, yc, hist2d.T, levels=levels, cmap="viridis", linewidths=0.8)
                ax.clabel(cs, inline=True, fontsize=6, fmt="%1.0f")

            if cfg["show_threshold"]:
                thr = min_joint_pos_rel_magnitude
                ax.axvline(-thr, color="red", linestyle="--", linewidth=1.2, label=f"±{thr:.1e}")
                ax.axvline(thr, color="red", linestyle="--", linewidth=1.2)
                key = (ag, dim_idx)
                if bounds and key in bounds:
                    lower, upper, _, _ = bounds[key]
                    ax.axhline(
                        lower,
                        color="purple",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"lower ({lower:.2f})",
                    )
                    ax.axhline(
                        upper,
                        color="purple",
                        linestyle="--",
                        linewidth=1.2,
                        label=f"upper ({upper:.2f})",
                    )
                ax.legend(fontsize=7, loc="upper right")

            ax.set_xlabel("joint_pos_rel")
            ax.set_ylabel("relative tracking error")
            ax.set_title(f"{ag}[{dim_idx}]  (N={len(x):,})", fontsize=9)

        for idx in range(n_dims, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf_path = output_dir / f"joint_histogram_jpr_vs_reltrack_{cfg['suffix']}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved joint histogram: {pdf_path}")


def save_outlier_gifs(
    outlier_episodes: list[dict],
    output_dir: Path | str,
    merge_gap: int = 20,
    context_frames: int = 5,
    camera_preference: tuple[str, ...] = ("exo_camera_1", "wrist_camera"),
    fps_gif: float = 5.0,
    sample_rate: float = 0.1,
    max_samples: int = 50,
) -> int:
    """Save GIFs of merged outlier segments for visual inspection.

    Outlier timesteps within *merge_gap* frames of each other are merged
    into a single segment (iteratively, until no more merges are possible).
    Each segment is then padded by *context_frames* on both sides and saved
    as one GIF.

    Filename convention::

        {house}_{batch}_traj{idx}_f{start}-{end}_{max_std:.1f}std_{signal}.gif

    Args:
        outlier_episodes: Output of :func:`episodes_with_kinematics_outliers`.
        output_dir: Directory to write GIF files into (created if needed).
        merge_gap: Maximum frame distance between two outlier timesteps for
            them to be merged into the same segment.
        context_frames: Number of extra frames to include before the first
            and after the last outlier in each segment.
        camera_preference: Ordered list of camera names to try when looking
            for a video reference inside the H5 file.
        fps_gif: Playback speed of the output GIF (frames per second).
        sample_rate: relative amount of examples to render
        max_samples: absolute max number of samples to save
            (only applied if sample_rate < 1.0).

    Returns:
        Number of GIF files written.
    """
    import imageio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    if sample_rate < 1.0 and outlier_episodes:
        rlist = outlier_episodes[:]
        import random

        random.shuffle(rlist)
        outlier_episodes = rlist[: min(max_samples, max(1, int(len(rlist) * sample_rate)))]

    for episode in tqdm(outlier_episodes, desc="Saving outlier GIFs"):
        h5_path = Path(episode["h5_file"])
        traj_idx = episode["traj_idx"]
        house = episode["house"]
        timesteps = episode["timesteps"]
        std_aways = episode["std_aways"]

        # Derive batch label from H5 filename
        # e.g. "trajectories_batch_0_of_1.h5" -> "batch_0_of_1"
        batch_label = h5_path.stem.replace("trajectories", "").lstrip("_") or "0"

        # --- locate the video file ---
        # Try 1: read video filename from H5 sensor_data metadata
        video_path = None
        try:
            with h5py.File(h5_path, "r") as f:
                traj_grp = f.get(f"traj_{traj_idx}")
                if traj_grp is not None:
                    sd_grp = None
                    obs_grp = traj_grp.get("obs")
                    if obs_grp is not None:
                        sd_grp = obs_grp.get("sensor_data")

                    if sd_grp is not None:
                        camera_name = None
                        for pref in camera_preference:
                            if pref in sd_grp:
                                camera_name = pref
                                break
                        if camera_name is None:
                            candidates = list(sd_grp.keys())
                            if candidates:
                                camera_name = candidates[0]

                        if camera_name is not None:
                            byte_arr = sd_grp[camera_name][:]
                            video_filename = byte_arr.tobytes().decode("utf-8").rstrip("\x00")
                            video_path = h5_path.parent / video_filename
        except Exception:
            pass

        # Try 2: construct expected path from H5 filename + traj index
        # Pattern: episode_{traj_idx:08d}_{camera}{suffix}.mp4
        if video_path is None or not video_path.exists():
            suffix = h5_path.stem.replace("trajectories", "")  # e.g. "_batch_1_of_2"
            house_dir = h5_path.parent
            for cam in camera_preference:
                candidate = house_dir / f"episode_{traj_idx:08d}_{cam}{suffix}.mp4"
                if candidate.exists():
                    video_path = candidate
                    break

        # Try 3: glob for any MP4 matching this episode index
        if video_path is None or not video_path.exists():
            pattern = f"episode_{traj_idx:08d}_*{suffix}.mp4"
            matches = sorted(house_dir.glob(pattern))
            if matches:
                video_path = matches[0]

        if video_path is None or not video_path.exists():
            print(
                f"  Warning: no video found for traj_{traj_idx} of {h5_path}"
                f" (tried H5 metadata, constructed paths, and glob)"
            )
            continue

        # --- read video frames once per episode ---
        try:
            reader = imageio.get_reader(str(video_path), "ffmpeg")
            frames = [frame for frame in reader]
            reader.close()
        except Exception as e:
            print(f"  Warning: failed to read video {video_path}: {e}")
            continue

        n_frames = len(frames)
        if n_frames == 0:
            print(f"  Warning: video has 0 frames: {video_path}")
            continue

        # Per timestep, keep the worst (highest std_away)
        per_step: dict[int, float] = {}
        for t, sa in zip(timesteps, std_aways):
            if t not in per_step or sa > per_step[t]:
                per_step[t] = sa

        # Merge nearby outlier timesteps into segments (gap <= merge_gap)
        sorted_steps = sorted(per_step.keys())
        segments: list[list[int]] = []
        for t in sorted_steps:
            if segments and t - segments[-1][-1] <= merge_gap:
                segments[-1].append(t)
            else:
                segments.append([t])

        duration_ms = int(1000.0 / fps_gif)

        for seg_steps in segments:
            max_std = max(per_step[t] for t in seg_steps)

            # Frame range with context padding
            seg_start = max(0, seg_steps[0] - context_frames)
            seg_end = min(n_frames - 1, seg_steps[-1] + context_frames)
            clip = frames[seg_start : seg_end + 1]
            if not clip:
                continue

            gif_name = (
                f"{house}_{batch_label}_traj{traj_idx}_f{seg_start}-{seg_end}_{max_std:.1f}std.gif"
            )
            gif_path = output_dir / gif_name
            try:
                imageio.mimsave(str(gif_path), clip, duration=duration_ms, loop=0)
                n_saved += 1
            except Exception as e:
                print(f"  Warning: failed to save {gif_path}: {e}")

    print(f"Saved {n_saved} outlier GIFs to {output_dir}")
    return n_saved
