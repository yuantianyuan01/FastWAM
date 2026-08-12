#!/usr/bin/env bash
# Internal smoke/benchmark sweep for runnable RoboDojo tasks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

dataset="RoboDojo"
execution_mode="eval"
ckpt=""
env_cfg="arx_x5"
expert_num="100"
action_type="ee"
seed="0"
policy_gpu="0"
env_gpu="0"
gpu_ids=""
policy_gpu_ids=""
env_gpu_ids=""
policy_env=""
eval_env="RoboDojo"
eval_num="1"
policy_name=""
policy_host=""
policy_port=""
bind_host="0.0.0.0"
connect_timeout="5"
policy_dir=""
run_id="$(date +%Y-%m-%d_%H-%M-%S)_smoke"
summary_path=""
markdown_path=""
log_dir=""
only_tasks=""
tasks_file=""
dimensions=""
resume="false"
fail_fast="false"
dry_run="false"
limit=""

usage() {
  cat <<'EOF'
Usage: bash scripts/internal/smoke_all_tasks.sh [options]

Runs RoboDojo tasks through scripts/robodojo.sh eval, client, or server. Default is
sequential; pass `--gpu-ids` or `--policy-gpu-ids/--env-gpu-ids` for balanced
multi-GPU parallel execution.

Options:
  --mode NAME         Internal runner mode: eval, client, or server (default: eval).
  --only a,b,c        Comma-separated task subset.
  --tasks-file PATH   Newline-separated task subset. Comments and blank lines are ignored.
  --dimension NAMES   Run capability dimensions (comma-separated or repeated):
                      generalization, memory, precision, long-horizon, open, all.
  --gpu-ids IDS       Balanced multi-GPU sweep. Comma-separated ids, reused for policy+env.
  --policy-gpu-ids IDS  Optional per-worker policy GPU ids. Defaults to --gpu-ids.
  --env-gpu-ids IDS     Optional per-worker env GPU ids. Defaults to --gpu-ids.
  --resume            Skip tasks already marked PASS in the summary file.
  --fail-fast         Stop after the first failed task.
  --dry-run           Print eval commands and mark tasks DRY_RUN without launching eval.
  --all               Explicitly run all runnable tasks (default when --only is omitted).
  --limit NUM         Run only the first NUM tasks after filtering.
  --summary PATH      JSON summary path (default: smoke_results/<run_id>.json)
  --markdown PATH     Markdown summary path (default: smoke_results/<run_id>.md)
  --run-id ID         Stable run id used in result paths and summaries.
  --eval-num NUM      Episode count for each task (default: 1). Use `native` to use per-task counts from _task.yml.
  --dataset NAME      eval.sh dataset arg (default: RoboDojo)
  --ckpt NAME         Policy checkpoint name (required)
  --env-cfg NAME      env_cfg stem (default: arx_x5)
  --expert-num NUM    Expert data count argument (default: 100)
  --action-type NAME  Policy action type (default: ee)
  --seed NUM          Eval seed / layout seed (default: 0)
  --policy-name NAME  Required in client mode when --policy-dir is omitted.
  --policy-host HOSTS Policy host or comma-separated hosts in client mode.
  --policy-port PORTS Policy port or comma-separated ports in client mode.
  --bind-host HOSTS   Server bind host or comma-separated hosts in server mode.
  --connect-timeout SEC  Client pre-flight reachability timeout (default: 5)
  --policy-gpu ID     GPU id for policy server (default: 0)
  --env-gpu ID        GPU id for Isaac Sim client (default: 0)
  --policy-env NAME   Policy conda env or uv env path (required)
  --eval-env NAME     Simulator conda env (default: RoboDojo)
  --policy-dir PATH   Policy directory containing eval.sh (required)
  -h, --help          Show this help

Examples:
  bash scripts/internal/smoke_all_tasks.sh --policy-dir XPolicyLab/policy/demo_policy --ckpt ckpt --policy-env env --only stack_bowls
  bash scripts/internal/smoke_all_tasks.sh --policy-dir XPolicyLab/policy/demo_policy --ckpt ckpt --policy-env env --dimension memory
  bash scripts/internal/smoke_all_tasks.sh --policy-dir XPolicyLab/policy/demo_policy --ckpt ckpt --policy-env env --eval-num native --gpu-ids 0,1,2,3
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[smoke_all_tasks] Missing value for $1" >&2
    exit 2
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) need_value "$@"; execution_mode="$2"; shift 2 ;;
    --only) need_value "$@"; only_tasks="$2"; shift 2 ;;
    --tasks-file) need_value "$@"; tasks_file="$2"; shift 2 ;;
    --dimension)
      need_value "$@"
      dimensions="${dimensions:+${dimensions},}$2"
      shift 2
      ;;
    --resume) resume="true"; shift ;;
    --all) shift ;;
    --fail-fast) fail_fast="true"; shift ;;
    --dry-run) dry_run="true"; shift ;;
    --limit) need_value "$@"; limit="$2"; shift 2 ;;
    --summary) need_value "$@"; summary_path="$2"; shift 2 ;;
    --markdown) need_value "$@"; markdown_path="$2"; shift 2 ;;
    --run-id) need_value "$@"; run_id="$2"; shift 2 ;;
    --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
    --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt="$2"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --expert-num) need_value "$@"; expert_num="$2"; shift 2 ;;
    --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
    --seed) need_value "$@"; seed="$2"; shift 2 ;;
    --policy-name) need_value "$@"; policy_name="$2"; shift 2 ;;
    --policy-host) need_value "$@"; policy_host="$2"; shift 2 ;;
    --policy-port) need_value "$@"; policy_port="$2"; shift 2 ;;
    --bind-host) need_value "$@"; bind_host="$2"; shift 2 ;;
    --connect-timeout) need_value "$@"; connect_timeout="$2"; shift 2 ;;
    --policy-gpu) need_value "$@"; policy_gpu="$2"; shift 2 ;;
    --env-gpu) need_value "$@"; env_gpu="$2"; shift 2 ;;
    --gpu-ids) need_value "$@"; gpu_ids="$2"; shift 2 ;;
    --policy-gpu-ids) need_value "$@"; policy_gpu_ids="$2"; shift 2 ;;
    --env-gpu-ids) need_value "$@"; env_gpu_ids="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --eval-env) need_value "$@"; eval_env="$2"; shift 2 ;;
    --policy-dir) need_value "$@"; policy_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[smoke_all_tasks] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

dimension_args=(--resolve-dimensions)
if [[ -n "${dimensions}" ]]; then
  dimension_args+=(--dimension "${dimensions}")
fi
dimensions="$(python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" "${dimension_args[@]}")"

if [[ "${execution_mode}" != "eval" && "${execution_mode}" != "client" && "${execution_mode}" != "server" ]]; then
  echo "[smoke_all_tasks] --mode must be eval, client, or server" >&2
  exit 2
fi

if [[ -n "${policy_dir}" ]]; then
  policy_dir="$(abs_path "${policy_dir}")"
fi

if [[ "${execution_mode}" == "eval" ]]; then
  if [[ -z "${policy_dir}" || -z "${ckpt}" || -z "${policy_env}" ]]; then
    echo "[smoke_all_tasks] eval mode requires --policy-dir, --ckpt, and --policy-env" >&2
    usage >&2
    exit 2
  fi
  if [[ ! -f "${policy_dir}/eval.sh" ]]; then
    echo "[smoke_all_tasks] policy eval.sh not found: ${policy_dir}/eval.sh" >&2
    exit 1
  fi
  policy_name="$(basename "$(cd "${policy_dir}" && pwd)")"
elif [[ "${execution_mode}" == "client" ]]; then
  if [[ -n "${policy_dir}" ]]; then
    policy_name="$(basename "$(cd "${policy_dir}" && pwd)")"
  fi
  if [[ -z "${policy_name}" || -z "${policy_host}" || -z "${policy_port}" ]]; then
    echo "[smoke_all_tasks] client mode requires --policy-name or --policy-dir, plus --policy-host and --policy-port" >&2
    usage >&2
    exit 2
  fi
  if [[ ! -f "${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.py" ]]; then
    echo "[smoke_all_tasks] policy deploy adapter not found: ${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.py" >&2
    exit 1
  fi
else
  if [[ -z "${policy_dir}" || -z "${ckpt}" || -z "${policy_env}" || -z "${policy_port}" ]]; then
    echo "[smoke_all_tasks] server mode requires --policy-dir, --ckpt, --policy-env, and --policy-port" >&2
    usage >&2
    exit 2
  fi
  if [[ ! -f "${policy_dir}/setup_eval_policy_server.sh" ]]; then
    echo "[smoke_all_tasks] policy server setup script not found: ${policy_dir}/setup_eval_policy_server.sh" >&2
    exit 1
  fi
  policy_name="$(basename "$(cd "${policy_dir}" && pwd)")"
fi

summary_path="${summary_path:-${ROOT_DIR}/smoke_results/${run_id}.json}"
markdown_path="${markdown_path:-${ROOT_DIR}/smoke_results/${run_id}.md}"
log_dir="${ROOT_DIR}/smoke_results/${run_id}/logs"
mkdir -p "$(dirname "${summary_path}")" "$(dirname "${markdown_path}")" "${log_dir}"

RESULTS_TSV="$(mktemp)"
trap 'rm -f "${RESULTS_TSV}"' EXIT

if [[ "${resume}" == "true" && -f "${summary_path}" ]]; then
  python3 - "${summary_path}" "${RESULTS_TSV}" <<'PY'
import csv
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
fieldnames = ["status", "task", "exit_code", "eval_time", "elapsed_sec", "result_path", "log_path", "message"]
with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    for row in payload.get("results", []):
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
fi

load_tasks() {
  python3 - "${ROOT_DIR}" "${only_tasks}" "${tasks_file}" "${limit}" "${dimensions}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
only = sys.argv[2]
tasks_file = sys.argv[3]
limit = sys.argv[4]
dimensions = sys.argv[5]
sys.path.insert(0, str(root))

import subprocess
inventory_cmd = [
    sys.executable,
    str(root / "scripts" / "internal" / "task_inventory.py"),
    "--only-runnable",
]
if dimensions:
    inventory_cmd.extend(["--dimension", dimensions])
task_names = subprocess.check_output(inventory_cmd, text=True).splitlines()

selected = None
if only:
    selected = [item.strip() for item in only.split(",") if item.strip()]
if tasks_file:
    file_tasks = []
    for line in Path(tasks_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            file_tasks.append(line)
    selected = (selected or []) + file_tasks
if selected is not None:
    wanted = set(selected)
    unknown = sorted(wanted - set(task_names))
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")
    task_names = [name for name in task_names if name in wanted]
if limit:
    task_names = task_names[: int(limit)]
print("\n".join(task_names))
PY
}

passed_in_summary() {
  local task="$1"
  [[ -s "${RESULTS_TSV}" ]] || return 1
  python3 - "${RESULTS_TSV}" "${task}" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
task = sys.argv[2]
latest_status = None
if path.exists():
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("task") == task:
                latest_status = row.get("status")
raise SystemExit(0 if latest_status == "PASS" else 1)
PY
}

write_summaries() {
  python3 - "${RESULTS_TSV}" "${summary_path}" "${markdown_path}" "${run_id}" "${eval_num}" "${dimensions}" <<'PY'
import csv
import json
from pathlib import Path
import sys

tsv_path, json_path, md_path, run_id, eval_num, dimensions = sys.argv[1:]
raw_rows = []
if Path(tsv_path).exists():
    with open(tsv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw_rows.append(row)

latest_by_task = {}
task_order = []
for row in raw_rows:
    task = row.get("task", "")
    if task in latest_by_task:
        task_order.remove(task)
    latest_by_task[task] = row
    task_order.append(task)
rows = [latest_by_task[task] for task in task_order]

counts = {status: sum(row["status"] == status for row in rows) for status in ["PASS", "FAIL", "SKIP", "DRY_RUN"]}
try:
    eval_num_value = int(eval_num)
except ValueError:
    eval_num_value = eval_num
payload = {
    "run_id": run_id,
    "eval_num": eval_num_value,
    "dimensions": [item for item in dimensions.split(",") if item],
    "counts": counts,
    "results": rows,
}
Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

lines = [
    f"# RoboDojo Smoke Summary `{run_id}`",
    "",
    f"- eval_num: `{eval_num}`",
    f"- dimensions: `{dimensions or 'all'}`",
    f"- pass: `{counts['PASS']}`",
    f"- fail: `{counts['FAIL']}`",
    f"- skip: `{counts['SKIP']}`",
    f"- dry_run: `{counts['DRY_RUN']}`",
    "",
    "| Status | Task | Exit | Eval Time | Seconds | Result | Log | Message |",
    "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
]
for row in rows:
    lines.append(
        f"| {row['status']} | `{row['task']}` | {row['exit_code']} | {row['eval_time']} | "
        f"{row['elapsed_sec']} | `{row['result_path']}` | `{row['log_path']}` | {row['message']} |"
    )
Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

count_failures() {
  python3 - "${RESULTS_TSV}" "${summary_path}" <<'PY'
import csv
import json
from pathlib import Path
import sys

tsv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
if summary_path.exists():
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(int(payload.get("counts", {}).get("FAIL", 0)))
    raise SystemExit(0)

latest_by_task = {}
if tsv_path.exists():
    with tsv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            latest_by_task[row.get("task", "")] = row.get("status", "")
print(sum(status == "FAIL" for status in latest_by_task.values()))
PY
}

resolve_gpu_workers() {
  local resolved_policy_csv="${policy_gpu_ids}"
  local resolved_env_csv="${env_gpu_ids}"

  if [[ -n "${gpu_ids}" ]]; then
    [[ -n "${resolved_policy_csv}" ]] || resolved_policy_csv="${gpu_ids}"
    [[ -n "${resolved_env_csv}" ]] || resolved_env_csv="${gpu_ids}"
  fi
  if [[ -n "${resolved_policy_csv}" && -z "${resolved_env_csv}" ]]; then
    resolved_env_csv="${resolved_policy_csv}"
  fi
  if [[ -n "${resolved_env_csv}" && -z "${resolved_policy_csv}" ]]; then
    resolved_policy_csv="${resolved_env_csv}"
  fi

  POLICY_GPU_WORKERS=()
  ENV_GPU_WORKERS=()
  if [[ -n "${resolved_policy_csv}" ]]; then
    IFS=',' read -r -a POLICY_GPU_WORKERS <<< "${resolved_policy_csv}"
  fi
  if [[ -n "${resolved_env_csv}" ]]; then
    IFS=',' read -r -a ENV_GPU_WORKERS <<< "${resolved_env_csv}"
  fi

  if [[ "${#POLICY_GPU_WORKERS[@]}" -ne "${#ENV_GPU_WORKERS[@]}" ]]; then
    echo "[smoke_all_tasks] policy/env gpu worker count mismatch" >&2
    exit 2
  fi
}

resolve_client_endpoints() {
  local group_count="$1"
  local -a hosts=()
  local -a ports=()
  local idx

  CLIENT_HOST_WORKERS=()
  CLIENT_PORT_WORKERS=()

  IFS=',' read -r -a hosts <<< "${policy_host}"
  IFS=',' read -r -a ports <<< "${policy_port}"

  if [[ "${group_count}" -lt 1 ]]; then
    echo "[smoke_all_tasks] invalid client group count: ${group_count}" >&2
    exit 2
  fi
  if [[ "${#hosts[@]}" -lt 1 || "${#ports[@]}" -lt 1 ]]; then
    echo "[smoke_all_tasks] client mode requires non-empty --policy-host and --policy-port" >&2
    exit 2
  fi

  if [[ "${#hosts[@]}" -eq 1 ]]; then
    if [[ "${#ports[@]}" -ne "${group_count}" ]]; then
      echo "[smoke_all_tasks] single host mode requires --policy-port count to equal group count (${group_count})" >&2
      exit 2
    fi
    for (( idx=0; idx<group_count; idx++ )); do
      CLIENT_HOST_WORKERS+=("${hosts[0]}")
      CLIENT_PORT_WORKERS+=("${ports[idx]}")
    done
    return 0
  fi

  if [[ "${#hosts[@]}" -ne "${group_count}" ]]; then
    echo "[smoke_all_tasks] multi-host mode requires --policy-host count to equal group count (${group_count})" >&2
    exit 2
  fi
  if [[ "${#ports[@]}" -ne "${group_count}" ]]; then
    echo "[smoke_all_tasks] multi-host mode requires --policy-port count to equal group count (${group_count})" >&2
    exit 2
  fi

  for (( idx=0; idx<group_count; idx++ )); do
    CLIENT_HOST_WORKERS+=("${hosts[idx]}")
    CLIENT_PORT_WORKERS+=("${ports[idx]}")
  done
}

resolve_server_endpoints() {
  local group_count="$1"
  local -a hosts=()
  local -a ports=()
  local idx

  SERVER_HOST_WORKERS=()
  SERVER_PORT_WORKERS=()

  IFS=',' read -r -a hosts <<< "${bind_host}"
  IFS=',' read -r -a ports <<< "${policy_port}"

  if [[ "${group_count}" -lt 1 ]]; then
    echo "[smoke_all_tasks] invalid server group count: ${group_count}" >&2
    exit 2
  fi
  if [[ "${#ports[@]}" -ne "${group_count}" ]]; then
    echo "[smoke_all_tasks] server mode requires --policy-port count to equal group count (${group_count})" >&2
    exit 2
  fi
  if [[ "${#hosts[@]}" -eq 1 ]]; then
    for (( idx=0; idx<group_count; idx++ )); do
      SERVER_HOST_WORKERS+=("${hosts[0]}")
      SERVER_PORT_WORKERS+=("${ports[idx]}")
    done
    return 0
  fi
  if [[ "${#hosts[@]}" -ne "${group_count}" ]]; then
    echo "[smoke_all_tasks] server mode requires --bind-host count to equal group count (${group_count}) when multiple hosts are provided" >&2
    exit 2
  fi
  for (( idx=0; idx<group_count; idx++ )); do
    SERVER_HOST_WORKERS+=("${hosts[idx]}")
    SERVER_PORT_WORKERS+=("${ports[idx]}")
  done
}

parallel_mode_enabled() {
  [[ "${#POLICY_GPU_WORKERS[@]}" -gt 1 ]]
}

ensure_task_count_for_workers() {
  local worker_count="$1"
  if [[ "${#ACTIVE_TASKS[@]}" -lt "${worker_count}" ]]; then
    echo "[smoke_all_tasks] task count (${#ACTIVE_TASKS[@]}) must be >= worker count (${worker_count}) in multi-GPU mode" >&2
    exit 2
  fi
}

build_parallel_assignment() {
  local task_list_file="$1"
  local assignment_path="$2"
  python3 - "${env_cfg}" "${task_list_file}" "${assignment_path}" \
    "$(IFS=,; echo "${POLICY_GPU_WORKERS[*]}")" \
    "$(IFS=,; echo "${ENV_GPU_WORKERS[*]}")" <<'PY'
import json
import sys
from pathlib import Path

RUNTIME_WEIGHTS = {
    "imitate_sorting_sequence/arx_x5": 10596,
    "press_by_number/arx_x5": 2955,
    "sweep_blocks_random/arx_x5": 2392,
    "align_blocks/arx_x5": 2013,
    "plug_in_charger/arx_x5": 1777,
    "fold_clothes/arx_x5": 1435,
    "pour_by_language/arx_x5": 9448,
    "fill_egg_holder/arx_x5": 3063,
    "pack_objects_into_box/arx_x5": 2725,
    "pour_balls_into_vase/arx_x5": 2509,
    "solve_equation/arx_x5": 1867,
    "insert_key/arx_x5": 1396,
    "play_tic_tac_toe/arx_x5": 9323,
    "arrange_largest_number_random/arx_x5": 3101,
    "match_and_pick_from_conveyor/arx_x5": 2915,
    "sort_nesting_dolls_by_size/arx_x5": 2377,
    "stack_blocks_by_language/arx_x5": 1733,
    "store_laptop_and_headphones/arx_x5": 1715,
    "fasten_screws/arx_x5": 7876,
    "make_toast_random/arx_x5": 3775,
    "swap_blocks/arx_x5": 2941,
    "stack_bowls/arx_x5": 1806,
    "pour_liquid_into_cup_random/arx_x5": 1798,
    "hang_mugs/arx_x5": 1734,
    "general_pickup/arx_x5": 1201,
    "play_stacking_toy/arx_x5": 5416,
    "make_kong/arx_x5": 3818,
    "pick_from_conveyor_by_image/arx_x5": 3150,
    "pack_objects_into_box_random/arx_x5": 2917,
    "play_Xylophone/arx_x5": 2174,
    "hang_mugs_random/arx_x5": 2047,
    "push_T_random/arx_x5": 1593,
    "classify_objects_by_language/arx_x5": 4955,
    "cover_blocks/arx_x5": 3837,
    "make_toast/arx_x5": 3781,
    "pour_liquid_into_cup/arx_x5": 2543,
    "arrange_largest_number/arx_x5": 2346,
    "store_laptop_and_headphones_random/arx_x5": 1981,
    "swap_T/arx_x5": 1746,
    "classify_objects/arx_x5": 4881,
    "organize_table/arx_x5": 4104,
    "store_tools_in_toolbox/arx_x5": 3783,
    "sweep_blocks/arx_x5": 2170,
    "fold_clothes_random/arx_x5": 1858,
    "stack_blocks_random/arx_x5": 1768,
    "deposit_coin/arx_x5": 1373,
    "stack_blocks/arx_x5": 1252,
    "build_tower/arx_x5": 4750,
    "fill_pen_holder/arx_x5": 4698,
    "sort_nesting_dolls_by_size_random/arx_x5": 3021,
    "put_bottles_into_dustbin/arx_x5": 2887,
    "insert_tubes/arx_x5": 2296,
    "stack_bowls_random/arx_x5": 2146,
    "push_T/arx_x5": 1311,
}


def score(groups: list[list[dict]]) -> tuple[int, int]:
    loads = [sum(item["seconds"] for item in group) for group in groups]
    return max(loads), max(loads) - min(loads)


def partition(tasks: list[dict], group_count: int) -> list[list[dict]]:
    groups = [[] for _ in range(group_count)]
    loads = [0] * group_count
    for task in tasks:
        idx = min(range(group_count), key=lambda i: (loads[i], i))
        groups[idx].append(task)
        loads[idx] += task["seconds"]

    while True:
        current = score(groups)
        loads = [sum(item["seconds"] for item in group) for group in groups]
        src_idx = max(range(len(groups)), key=lambda i: (loads[i], i))
        best = None

        for task_idx, task in enumerate(groups[src_idx]):
            for dst_idx in range(len(groups)):
                if dst_idx == src_idx:
                    continue
                candidate = [list(group) for group in groups]
                candidate[src_idx].pop(task_idx)
                candidate[dst_idx].append(task)
                candidate_score = score(candidate)
                if candidate_score < current and (best is None or candidate_score < best[0]):
                    best = (candidate_score, candidate)

        for task_idx, task in enumerate(groups[src_idx]):
            for dst_idx in range(len(groups)):
                if dst_idx == src_idx:
                    continue
                for other_idx, other in enumerate(groups[dst_idx]):
                    candidate = [list(group) for group in groups]
                    candidate[src_idx][task_idx] = other
                    candidate[dst_idx][other_idx] = task
                    candidate_score = score(candidate)
                    if candidate_score < current and (best is None or candidate_score < best[0]):
                        best = (candidate_score, candidate)

        if best is None:
            break
        groups = best[1]

    for group in groups:
        group.sort(key=lambda item: (-item["seconds"], item["task"]))
    return groups


env_cfg = sys.argv[1]
task_file = Path(sys.argv[2])
assignment_path = Path(sys.argv[3])
policy_gpus = [item for item in sys.argv[4].split(",") if item]
env_gpus = [item for item in sys.argv[5].split(",") if item]
if len(policy_gpus) != len(env_gpus):
    raise SystemExit("policy/env gpu list length mismatch")
if not policy_gpus:
    raise SystemExit("no gpu workers resolved")

task_names = [
    line.strip() for line in task_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
tasks = []
missing = []
for task_name in task_names:
    key = f"{task_name}/{env_cfg}"
    seconds = RUNTIME_WEIGHTS.get(key)
    if seconds is None:
        missing.append(key)
        continue
    tasks.append({"task": task_name, "key": key, "seconds": seconds})
if missing:
    raise SystemExit("missing runtime entries: " + ", ".join(missing))

tasks.sort(key=lambda item: (-item["seconds"], item["task"]))
groups = partition(tasks, len(policy_gpus))
payload = {
    "weights_source": "embedded",
    "env_cfg": env_cfg,
    "groups": [
        {
            "worker": idx,
            "policy_gpu": policy_gpu,
            "env_gpu": env_gpu,
            "total_seconds": sum(item["seconds"] for item in group),
            "tasks": group,
        }
        for idx, (policy_gpu, env_gpu, group) in enumerate(zip(policy_gpus, env_gpus, groups))
    ],
}
assignment_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

record_result_row() {
  local target_tsv="$1"
  local status="$2"
  local task="$3"
  local exit_code="$4"
  local eval_time="$5"
  local elapsed_sec="$6"
  local result_path="$7"
  local log_path="$8"
  local message="$9"
  if [[ ! -s "${target_tsv}" ]]; then
    printf 'status\ttask\texit_code\teval_time\telapsed_sec\tresult_path\tlog_path\tmessage\n' > "${target_tsv}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${task}" "${exit_code}" "${eval_time}" "${elapsed_sec}" "${result_path}" "${log_path}" "${message}" \
    >> "${target_tsv}"
}

record_result() {
  local status="$1"
  local task="$2"
  local exit_code="$3"
  local eval_time="$4"
  local elapsed_sec="$5"
  local result_path="$6"
  local log_path="$7"
  local message="$8"
  if [[ ! -s "${RESULTS_TSV}" ]]; then
    printf 'status\ttask\texit_code\teval_time\telapsed_sec\tresult_path\tlog_path\tmessage\n' > "${RESULTS_TSV}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${task}" "${exit_code}" "${eval_time}" "${elapsed_sec}" "${result_path}" "${log_path}" "${message}" \
    >> "${RESULTS_TSV}"
  write_summaries
}

resolve_gpu_workers
if [[ "${#POLICY_GPU_WORKERS[@]}" -eq 1 ]]; then
  policy_gpu="${POLICY_GPU_WORKERS[0]}"
  env_gpu="${ENV_GPU_WORKERS[0]}"
fi

task_output="$(load_tasks)"
TASKS=()
if [[ -n "${task_output}" ]]; then
  mapfile -t TASKS <<< "${task_output}"
fi
ACTIVE_TASKS=()
for task in "${TASKS[@]}"; do
  if [[ "${resume}" == "true" ]] && passed_in_summary "${task}"; then
    echo "[smoke_all_tasks] SKIP ${task} (already PASS in summary)"
    continue
  fi
  ACTIVE_TASKS+=("${task}")
done

echo "[smoke_all_tasks] tasks=${#TASKS[@]} active=${#ACTIVE_TASKS[@]} dimensions=${dimensions:-all} eval_num=${eval_num} run_id=${run_id}"
echo "[smoke_all_tasks] summary=${summary_path}"
echo "[smoke_all_tasks] markdown=${markdown_path}"

extract_eval_time() {
  local result_path="$1"
  python3 - "${result_path}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload.get("eval_time", 0)))
PY
}

run_eval_for_task() {
  local task="$1"
  local task_policy_gpu="$2"
  local task_env_gpu="$3"
  local target_tsv="$4"
  local task_host="${5:-}"
  local task_port="${6:-}"
  local task_run_id="${run_id}_${task}"
  local result_path="${ROOT_DIR}/eval_result/RoboDojo/${task}/${policy_name}/${env_cfg}/${seed}_ckpt_name=${ckpt},action_type=${action_type}/${task_run_id}/_result.json"
  local log_path="${log_dir}/${task}.log"
  local rc elapsed eval_time message start_sec end_sec

  if [[ "${execution_mode}" == "eval" ]]; then
    echo "[smoke_all_tasks] RUN ${task} (policy_gpu=${task_policy_gpu}, env_gpu=${task_env_gpu})"
  else
    echo "[smoke_all_tasks] RUN ${task} (host=${task_host}, port=${task_port}, env_gpu=${task_env_gpu})"
  fi
  start_sec="$(date +%s)"
  set +e
  if [[ "${execution_mode}" == "eval" ]]; then
    eval_cmd=(
      bash "${ROOT_DIR}/scripts/robodojo.sh" eval
      --dataset "${dataset}"
      --task "${task}"
      --ckpt "${ckpt}"
      --env-cfg "${env_cfg}"
      --expert-num "${expert_num}"
      --action-type "${action_type}"
      --seed "${seed}"
      --policy-gpu "${task_policy_gpu}"
      --env-gpu "${task_env_gpu}"
      --policy-env "${policy_env}"
      --eval-env "${eval_env}"
      --policy-dir "${policy_dir}"
    )
  else
    eval_cmd=(
      bash "${ROOT_DIR}/scripts/robodojo.sh" client
      --dataset "${dataset}"
      --task "${task}"
      --env-cfg "${env_cfg}"
      --policy-name "${policy_name}"
      --policy-host "${task_host}"
      --policy-port "${task_port}"
      --seed "${seed}"
      --env-gpu "${task_env_gpu}"
      --ckpt "${ckpt}"
      --action-type "${action_type}"
      --connect-timeout "${connect_timeout}"
    )
  fi
  if [[ "${eval_num}" != "native" ]]; then
    eval_cmd+=(--eval-num "${eval_num}")
  fi
  if [[ "${dry_run}" == "true" ]]; then
    eval_cmd+=(--dry-run)
  fi
  ROBODOJO_RUN_ID="${task_run_id}" \
  ROBODOJO_FATAL_RESTART_COUNT=0 \
  "${eval_cmd[@]}" \
    > "${log_path}" 2>&1
  rc=$?
  set -e
  end_sec="$(date +%s)"
  elapsed=$((end_sec - start_sec))

  if [[ "${dry_run}" == "true" ]]; then
    record_result_row "${target_tsv}" "DRY_RUN" "${task}" "${rc}" "-" "${elapsed}" "${result_path}" "${log_path}" "command rendered only"
    return 0
  fi

  eval_time="-"
  message=""
  if [[ -f "${result_path}" ]]; then
    eval_time="$(extract_eval_time "${result_path}")"
  else
    message="missing _result.json"
  fi

  if [[ "${rc}" -eq 0 && "${eval_time}" =~ ^[0-9]+$ && "${eval_time}" -ge 1 ]]; then
    record_result_row "${target_tsv}" "PASS" "${task}" "${rc}" "${eval_time}" "${elapsed}" "${result_path}" "${log_path}" "ok"
    return 0
  fi

  if [[ -z "${message}" ]]; then
    message="exit=${rc}, eval_time=${eval_time}"
  fi
  record_result_row "${target_tsv}" "FAIL" "${task}" "${rc}" "${eval_time}" "${elapsed}" "${result_path}" "${log_path}" "${message}"
  return 1
}

run_server_for_group() {
  local bootstrap_task="$1"
  local task_policy_gpu="$2"
  local task_host="$3"
  local task_port="$4"
  local -a server_cmd

  if [[ -z "${bootstrap_task}" ]]; then
    echo "[smoke_all_tasks] server group has no bootstrap task" >&2
    return 1
  fi

  echo "[smoke_all_tasks] START SERVER bootstrap_task=${bootstrap_task} policy_gpu=${task_policy_gpu} bind=${task_host}:${task_port}"
  server_cmd=(
    bash "${ROOT_DIR}/scripts/robodojo.sh" server
    --dataset "${dataset}"
    --task "${bootstrap_task}"
    --ckpt "${ckpt}"
    --env-cfg "${env_cfg}"
    --action-type "${action_type}"
    --seed "${seed}"
    --policy-gpu "${task_policy_gpu}"
    --policy-env "${policy_env}"
    --policy-dir "${policy_dir}"
    --policy-port "${task_port}"
    --bind-host "${task_host}"
  )
  if [[ "${dry_run}" == "true" ]]; then
    server_cmd+=(--dry-run)
  fi
  exec "${server_cmd[@]}"
}

run_parallel_group() {
  local worker_id="$1"
  local task_policy_gpu="$2"
  local task_env_gpu="$3"
  local task_file="$4"
  local target_tsv="$5"
  local fail_flag="$6"
  local task_host="${7:-}"
  local task_port="${8:-}"
  local task failed="0"

  while IFS= read -r task || [[ -n "${task}" ]]; do
    [[ -n "${task}" ]] || continue
    if [[ "${fail_fast}" == "true" && -f "${fail_flag}" ]]; then
      break
    fi
    if ! run_eval_for_task "${task}" "${task_policy_gpu}" "${task_env_gpu}" "${target_tsv}" "${task_host}" "${task_port}"; then
      failed="1"
      if [[ "${fail_fast}" == "true" ]]; then
        : > "${fail_flag}"
        break
      fi
    fi
  done < "${task_file}"

  if [[ "${failed}" == "1" ]]; then
    return 1
  fi
}

run_sequential_tasks() {
  local task
  local task_host=""
  local task_port=""
  if [[ "${execution_mode}" == "client" ]]; then
    resolve_client_endpoints 1
    task_host="${CLIENT_HOST_WORKERS[0]}"
    task_port="${CLIENT_PORT_WORKERS[0]}"
  elif [[ "${execution_mode}" == "server" ]]; then
    resolve_server_endpoints 1
    run_server_for_group "${ACTIVE_TASKS[0]}" "${policy_gpu}" "${SERVER_HOST_WORKERS[0]}" "${SERVER_PORT_WORKERS[0]}"
    return $?
  fi
  for task in "${ACTIVE_TASKS[@]}"; do
    if ! run_eval_for_task "${task}" "${policy_gpu}" "${env_gpu}" "${RESULTS_TSV}" "${task_host}" "${task_port}"; then
      write_summaries
      if [[ "${fail_fast}" == "true" ]]; then
        echo "[smoke_all_tasks] fail-fast stopping at ${task}" >&2
        return 1
      fi
    fi
    write_summaries
  done
}

run_parallel_tasks() {
  local active_tasks_file assignment_path assignment_meta_dir fail_flag
  local worker_index policy_gpu_id env_gpu_id task_file worker_tsv total_seconds
  local task_host task_port bootstrap_task
  local -a worker_pids=()
  local -a worker_tsvs=()

  active_tasks_file="$(mktemp)"
  assignment_meta_dir="$(mktemp -d)"
  assignment_path="${assignment_meta_dir}/assignment.json"
  fail_flag="${assignment_meta_dir}/fail_fast.flag"
  trap 'rm -f "${active_tasks_file}"; rm -rf "${assignment_meta_dir}"' RETURN

  printf '%s\n' "${ACTIVE_TASKS[@]}" > "${active_tasks_file}"
  build_parallel_assignment "${active_tasks_file}" "${assignment_path}" > /dev/null
  if [[ "${execution_mode}" == "client" ]]; then
    resolve_client_endpoints "${#ENV_GPU_WORKERS[@]}"
  elif [[ "${execution_mode}" == "server" ]]; then
    resolve_server_endpoints "${#POLICY_GPU_WORKERS[@]}"
  fi

  python3 - "${assignment_path}" "${assignment_meta_dir}" > /dev/null <<'PY'
import json
from pathlib import Path
import sys

assignment = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta_dir = Path(sys.argv[2])
rows = []
for group in assignment["groups"]:
    task_file = meta_dir / f"worker_{group['worker']}.tasks"
    task_file.write_text(
        "\n".join(item["task"] for item in group["tasks"]) + ("\n" if group["tasks"] else ""),
        encoding="utf-8",
    )
    rows.append(
        "\t".join(
            [
                str(group["worker"]),
                str(group["policy_gpu"]),
                str(group["env_gpu"]),
                str(task_file),
                str(group["total_seconds"]),
                ",".join(item["task"] for item in group["tasks"]),
            ]
        )
    )
print("\n".join(rows))
PY

  while IFS=$'\t' read -r worker_index policy_gpu_id env_gpu_id task_file total_seconds task_names; do
    [[ -n "${worker_index}" ]] || continue
    worker_tsv="${assignment_meta_dir}/worker_${worker_index}.tsv"
    worker_tsvs+=("${worker_tsv}")
    if [[ "${execution_mode}" == "client" ]]; then
      task_host="${CLIENT_HOST_WORKERS[worker_index]}"
      task_port="${CLIENT_PORT_WORKERS[worker_index]}"
      echo "[smoke_all_tasks] group=${worker_index} host=${task_host} port=${task_port} env_gpu=${env_gpu_id} load=${total_seconds}s tasks=${task_names:-<empty>}"
      run_parallel_group "${worker_index}" "${policy_gpu_id}" "${env_gpu_id}" "${task_file}" "${worker_tsv}" "${fail_flag}" "${task_host}" "${task_port}" &
    elif [[ "${execution_mode}" == "server" ]]; then
      task_host="${SERVER_HOST_WORKERS[worker_index]}"
      task_port="${SERVER_PORT_WORKERS[worker_index]}"
      bootstrap_task="$(head -n 1 "${task_file}" | tr -d '\r')"
      echo "[smoke_all_tasks] group=${worker_index} bootstrap_task=${bootstrap_task:-<empty>} policy_gpu=${policy_gpu_id} bind=${task_host}:${task_port} load=${total_seconds}s tasks=${task_names:-<empty>}"
      run_server_for_group "${bootstrap_task}" "${policy_gpu_id}" "${task_host}" "${task_port}" &
    else
      task_host=""
      task_port=""
      echo "[smoke_all_tasks] group=${worker_index} policy_gpu=${policy_gpu_id} env_gpu=${env_gpu_id} load=${total_seconds}s tasks=${task_names:-<empty>}"
      run_parallel_group "${worker_index}" "${policy_gpu_id}" "${env_gpu_id}" "${task_file}" "${worker_tsv}" "${fail_flag}" "${task_host}" "${task_port}" &
    fi
    worker_pids+=("$!")
  done < <(
    python3 - "${assignment_path}" "${assignment_meta_dir}" <<'PY'
import json
from pathlib import Path
import sys

assignment = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta_dir = Path(sys.argv[2])
for group in assignment["groups"]:
    task_file = meta_dir / f"worker_{group['worker']}.tasks"
    print(
        "\t".join(
            [
                str(group["worker"]),
                str(group["policy_gpu"]),
                str(group["env_gpu"]),
                str(task_file),
                str(group["total_seconds"]),
                ",".join(item["task"] for item in group["tasks"]),
            ]
        )
    )
PY
  )

  local worker_rc overall_rc="0"
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      overall_rc="1"
    fi
  done

  if [[ "${execution_mode}" == "server" ]]; then
    if [[ "${overall_rc}" != "0" ]]; then
      return 1
    fi
    return 0
  fi

  python3 - "${RESULTS_TSV}" "${active_tasks_file}" "${worker_tsvs[@]}" <<'PY'
import csv
from pathlib import Path
import sys

out_path = Path(sys.argv[1])
task_order = [line.strip() for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if line.strip()]
rows = []
if out_path.exists():
    with out_path.open(encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f, delimiter="\t"))
for path_str in sys.argv[3:]:
    path = Path(path_str)
    if not path.exists():
        continue
    with path.open(encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f, delimiter="\t"))
order = {task: idx for idx, task in enumerate(task_order)}
rows.sort(key=lambda row: (order.get(row["task"], len(order)), row["task"]))
fieldnames = ["status", "task", "exit_code", "eval_time", "elapsed_sec", "result_path", "log_path", "message"]
with out_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key, "") for key in fieldnames})
PY

  write_summaries
  if [[ "${overall_rc}" != "0" ]]; then
    return 1
  fi
}

if [[ "${#ACTIVE_TASKS[@]}" -eq 0 ]]; then
  write_summaries
  echo "[smoke_all_tasks] no tasks left to run"
  exit 0
fi

if [[ "${execution_mode}" == "server" ]]; then
  echo "[smoke_all_tasks] mode=server workers=${#POLICY_GPU_WORKERS[@]} weights=embedded"
  if [[ "${#POLICY_GPU_WORKERS[@]}" -gt 1 ]]; then
    ensure_task_count_for_workers "${#POLICY_GPU_WORKERS[@]}"
    run_parallel_tasks
  else
    run_sequential_tasks
  fi
elif parallel_mode_enabled; then
  echo "[smoke_all_tasks] mode=parallel workers=${#POLICY_GPU_WORKERS[@]} weights=embedded"
  ensure_task_count_for_workers "${#POLICY_GPU_WORKERS[@]}"
  run_parallel_tasks
else
  echo "[smoke_all_tasks] mode=sequential policy_gpu=${policy_gpu} env_gpu=${env_gpu}"
  run_sequential_tasks
fi

write_summaries
fail_count="$(count_failures)"
echo "[smoke_all_tasks] complete: ${summary_path}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
