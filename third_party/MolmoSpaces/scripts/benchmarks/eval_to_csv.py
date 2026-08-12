import os, json, tempfile
import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.stats import beta as beta_dist
import logging
log = logging.getLogger(__name__)

THOR_CAT_SIMPLIFY = {
    "saltshaker": "S/P Shaker", "peppershaker": "S/P Shaker",
    "tomato": "Fruit", "apple": "Fruit",
    "butterknife": "Knife", "boiler": "Kettle",
    "winebottle": "Bottle", "atomizer": "Spray Bottle",
    "remotecontrol": "Remote Control", "soapdispenser": "Soap Dispenser",
    "tissuepaper": "Tissue Paper",
}

def get_success_any(success_array: np.ndarray) -> bool:
    """True if any of the elements of success_array are True."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(np.any(success_array))


def get_success_last_frame(success_array: np.ndarray) -> bool:
    """True iff the last element of success_array is True (current metric)."""
    if success_array is None or len(success_array) == 0:
        return False
    return bool(success_array[-1])


def _extract_object_name(obs_scene_bytes):
    try:
        obs = json.loads(obs_scene_bytes.decode("utf-8"))
        raw = obs.get("object_name", "unknown")
        cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
        return cleaned.split()[0] if cleaned else "unknown"
    except Exception:
        return "unknown"


def _simplify(name: str) -> str:
    simp = THOR_CAT_SIMPLIFY.get(name.lower(), name)
    return " ".join(w.capitalize() for w in simp.split())


def _bayesian_ci(successes, total, alpha=0.05):
    if total == 0:
        return 0.0, 0.0
    a, b = 1 + successes, 1 + (total - successes)
    return beta_dist.ppf(alpha / 2, a, b) * 100, beta_dist.ppf(1 - alpha / 2, a, b) * 100


def _copy_group(src, dst):
    for k, item in src.items():
        if isinstance(item, h5py.Dataset):
            dst.create_dataset(k, data=item[()])
        elif isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(k))


def _decode_json_sequence(raw_uint8):
    rows = []
    for row in raw_uint8:
        d = json.loads(bytes(row).rstrip(b"\x00").decode("utf-8"))
        flat = []
        for v in d.values():
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        rows.append(flat)
    return np.array(rows, dtype=np.float64)


def _episode_joint_jerk(ep, dt, max_steps=None):
    raw_q = None
    try:
        raw_q = ep["obs"]["agent"]["qpos"][:]
    except KeyError:
        try:
            raw_q = ep["actions"]["joint_pos"][:]
        except KeyError:
            pass
    if raw_q is None:
        return np.nan
    if max_steps is not None:
        raw_q = raw_q[:max_steps]
    q = _decode_json_sequence(raw_q)
    if q.shape[0] < 4:
        return np.nan
    d3 = q[3:] - 3 * q[2:-1] + 3 * q[1:-2] - q[:-3]
    d3 /= dt ** 3
    return float(np.mean(np.linalg.norm(d3, axis=1)))


def _combine_trajectories(folder_path):
    folder = Path(folder_path)
    h5_files = sorted(folder.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 found under {folder_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()
    out = h5py.File(tmp.name, "w")
    ep = 0
    for src_path in h5_files:
        try:
            src = h5py.File(src_path, "r")
            for tk in [k for k in src.keys() if k.startswith("traj_")]:
                dst = out.create_group(f"episode_{ep:04d}_{tk}")
                _copy_group(src[tk], dst)
                ep += 1
            src.close()
        except Exception as e:
            print(f"Warning: skipping {src_path}: {e}")
    out.close()
    print(f"Combined {ep} episodes from {len(h5_files)} files → {tmp.name}")
    return tmp.name


def _build_row(policy_name, category, s, t, jerk_list, report_both, oracle_s=0):
    rate = 100.0 * s / t if t else 0.0
    ci_lo, ci_hi = _bayesian_ci(s, t)
    mean_jj = float(np.mean(jerk_list)) if jerk_list else np.nan
    std_jj = float(np.std(jerk_list)) if jerk_list else np.nan
    row = dict(
        policy=policy_name, category=category, successes=s, total=t,
        success_rate_pct=round(rate, 2),
        ci_95_low_pct=round(ci_lo, 2), ci_95_high_pct=round(ci_hi, 2),
    )
    if report_both:
        o_rate = 100.0 * oracle_s / t if t else 0.0
        o_ci_lo, o_ci_hi = _bayesian_ci(oracle_s, t)
        row.update(
            oracle_successes=oracle_s,
            oracle_rate_pct=round(o_rate, 2),
            oracle_ci_95_low_pct=round(o_ci_lo, 2),
            oracle_ci_95_high_pct=round(o_ci_hi, 2),
        )
    row.update(
        jerk_joint_mean=round(mean_jj, 6) if not np.isnan(mean_jj) else np.nan,
        jerk_joint_std=round(std_jj, 6) if not np.isnan(std_jj) else np.nan,
    )
    return row, rate


def eval_to_csv(
    run_path: str,
    policy_name: str,
    success_condition: str = "at-end",
    output_csv: str = "eval_results.csv",
    dt: float = 0.1,
    max_steps: int | None = None
):
    report_both = success_condition == "both"

    combined_h5 = _combine_trajectories(run_path)
    per_obj = defaultdict(lambda: {"success": 0, "oracle_success": 0, "total": 0, "jerk_joint": []})
    total_s, total_os, total_n = 0, 0, 0
    all_jerk_joint = []

    with h5py.File(combined_h5, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("episode_"):
                continue
            ep = f[key]

            if "success" in ep:
                s_arr = ep["success"][:max_steps] if max_steps is not None else ep["success"][:]
                if report_both:
                    success = get_success_last_frame(s_arr)
                    oracle_success = get_success_any(s_arr)
                elif success_condition == "at-end":
                    success = get_success_last_frame(s_arr)
                    oracle_success = None
                elif success_condition == "oracle":
                    success = get_success_any(s_arr)
                    oracle_success = None
                else:
                    raise ValueError(f"Unknown success condition: {success_condition}")
            else:
                log.info(f"Warning: no success array for {key}, skipping")
                continue

            jj = _episode_joint_jerk(ep, dt, max_steps=max_steps)
            obj = _simplify(_extract_object_name(ep["obs_scene"][()])) if "obs_scene" in ep else "Unknown"

            per_obj[obj]["total"] += 1
            per_obj[obj]["success"] += int(success)
            if oracle_success is not None:
                per_obj[obj]["oracle_success"] += int(oracle_success)
            if not np.isnan(jj):
                per_obj[obj]["jerk_joint"].append(jj)
                all_jerk_joint.append(jj)

            total_n += 1
            total_s += int(success)
            if oracle_success is not None:
                total_os += int(oracle_success)

    rows = []
    for obj in sorted(per_obj):
        d = per_obj[obj]
        row, _ = _build_row(policy_name, obj, d["success"], d["total"],
                            d["jerk_joint"], report_both, d["oracle_success"])
        rows.append(row)

    overall_row, rate = _build_row(policy_name, "OVERALL", total_s, total_n,
                                   all_jerk_joint, report_both, total_os)
    rows.append(overall_row)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    with open(output_csv, "w") as fout:
        fout.write(f"# policy_name: {policy_name}\n")
        fout.write(f"# run_path: {run_path}\n")
        fout.write(f"# dt: {dt}\n")
        fout.write(f"# max_steps: {max_steps}\n")
        df.to_csv(fout, index=False)

    summary = f"SR: {round(rate, 2)}%"
    if report_both:
        o_rate = 100.0 * total_os / total_n if total_n else 0.0
        summary = f"at-end: {round(rate, 2)}% | oracle: {round(o_rate, 2)}%"
    print(f"\nSaved → {os.path.abspath(output_csv)} {summary} of {total_n} episodes")

    os.unlink(combined_h5)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy and save results to CSV")
    parser.add_argument("run_path", help="Path to evaluation output directory")
    parser.add_argument("policy_name", help="Name of the policy")
    parser.add_argument("--success-condition", type=str, default="at-end", help="at-end | oracle | both")
    parser.add_argument("--output-csv", default="eval_results.csv", help="Output CSV file (default: eval_results.csv)")
    parser.add_argument("--dt", type=float, default=67/1000, help="Time step [s] (default: 0.1)")
    parser.add_argument("--steps-per-episode", type=int, default=None, help="Max steps per episode (default: None)")

    args = parser.parse_args()

    eval_to_csv(
        run_path=args.run_path,
        policy_name=args.policy_name,
        #reward_threshold=args.reward_threshold,
        success_condition=args.success_condition,
        output_csv=args.output_csv,
        dt=args.dt,
        max_steps=args.steps_per_episode,
    )
