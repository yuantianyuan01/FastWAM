#!/usr/bin/env bash
set -euo pipefail

# Ensure project root is on PYTHONPATH so first-party imports like `env`, `task`, and `utils` work.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/XPolicyLab:${PYTHONPATH:-}"
echo "[INFO] PYTHONPATH=${PYTHONPATH}"

# Usage:
#   bash eval_policy.sh <task_name> <env_cfg_type> <device_id> <policy_name> <port> [extra-args...]

if [[ $# -lt 5 ]]; then
  echo "usage: bash eval_policy.sh <task_name> <env_cfg_type> <device_id> <policy_name> <port> [extra-args...]" >&2
  exit 1
fi

root_dir=""
task_name=""
env_cfg_type=""
device_id=""
policy_name=""
port=""
eval_batch=""
additional_info=""
seed=""
host="localhost"
protocol=""
policy_server_url=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root_dir|--task_name|--env_cfg_type|--device_id|--policy_name|--port|--eval_batch|--additional_info|--seed|--host|--protocol|--policy_server_url)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "[ERROR] Missing value for argument: $1"
        exit 1
      fi

      case "$1" in
        --root_dir)    root_dir="$2" ;;
        --task_name)   task_name="$2" ;;
        --env_cfg_type)     env_cfg_type="$2" ;;
        --device_id)   device_id="$2" ;;
        --policy_name) policy_name="$2" ;;
        --port)        port="$2" ;;
        --eval_batch)  eval_batch="$2" ;;
        --additional_info)  additional_info="$2" ;;
        --seed)  seed="$2" ;;
        --host) host="$2" ;;
        --protocol) protocol="$2" ;;
        --policy_server_url) policy_server_url="$2" ;;
      esac
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

for var_name in root_dir task_name env_cfg_type device_id policy_name port; do
  if [[ -z "${!var_name}" ]]; then
    echo "[ERROR] Missing required argument: --${var_name}"
    exit 1
  fi
done

cd "$root_dir" || exit 1
# Config file path
cfg_file="./env_cfg/${env_cfg_type}.yml"
if [[ ! -f "$cfg_file" ]]; then
  echo "[ERROR] Config file not found: $cfg_file" >&2
  exit 1
fi

# Resolve sim config file from cfg_file: env_cfg/<name>.yml -> config.sim
sim_cfg_name="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['config']['sim'])" "$cfg_file")"
sim_cfg_file="./env_cfg/sim/${sim_cfg_name}.yml"
if [[ ! -f "$sim_cfg_file" ]]; then
  echo "[ERROR] Sim config file not found: $sim_cfg_file" >&2
  exit 1
fi

if [[ -z "${protocol}" ]]; then
  policy_deploy_file="./XPolicyLab/policy/${policy_name}/deploy.yml"
  if [[ -f "$policy_deploy_file" ]]; then
    protocol="$(python3 -c "import sys,yaml;print((yaml.safe_load(open(sys.argv[1])) or {}).get('protocol','ws'))" "$policy_deploy_file")"
  else
    protocol="ws"
  fi
fi
if [[ "${protocol}" == "ws" && -z "${policy_server_url}" ]]; then
  policy_server_url="ws://${host}:${port}"
fi
if [[ "${protocol}" == "ws" ]]; then
  echo "[INFO] policy transport = WebSocket (protocol: ws)"
else
  echo "[INFO] policy transport = ${protocol}"
fi
if [[ -n "${policy_server_url}" ]]; then
  echo "[INFO] policy_server_url = ${policy_server_url}"
fi

if [[ -n "${device_id}" ]]; then
  export CUDA_VISIBLE_DEVICES="${device_id}"
  echo "[INFO] device_id = ${device_id} → CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

if [[ -n "${eval_batch}" ]]; then
  echo "[INFO] eval_batch     = ${eval_batch}"
fi

# Read render_interval / env.num_envs from yaml (fallback if missing)
render_interval="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1])).get('render_interval',1))" "$sim_cfg_file")"
num_envs="$(python3 -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1])).get('scene',{}).get('num_envs',1))" "$sim_cfg_file")"

echo "[INFO] render_interval = ${render_interval}"
echo "[INFO] num_envs        = ${num_envs}"

extra_args=()

KIT_ENABLE_EXTS=(
  "isaacsim.replicator.behavior"
  "isaacsim.sensors.camera"
)

KIT_ARGS=""
for ext in "${KIT_ENABLE_EXTS[@]}"; do
  KIT_ARGS+=" --enable ${ext}"
done

# Generated once per eval invocation. Carries the same identity through
# os.execv inside main.py and bash-level retries below. Append $$ to
# defuse same-second collisions when the same task/config is launched
# in parallel.
if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
  export ROBODOJO_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
fi
echo "[eval_policy] ROBODOJO_RUN_ID=${ROBODOJO_RUN_ID}"

MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
while : ; do
  set +e
  python -u src/eval_client/main.py \
    --task_name "$task_name" \
    --env_cfg_type "$env_cfg_type" \
    --num_envs "$num_envs" \
    --enable_cameras \
    --kit_args "$KIT_ARGS" \
    --device_id "$device_id" \
    --policy_name "$policy_name" \
    --port "$port" \
    --protocol "$protocol" \
    --policy_server_url "$policy_server_url" \
    --additional_info "$additional_info" \
    --seed "$seed" \
    --host "$host" \
    --headless \
    "${extra_args[@]}" \
    "$@"
  rc=$?
  set -e
  case "$rc" in
    0)
      exit 0
      ;;
    99|134|139)
      # 99  - clean fatal-restart from main.py (in-process cap reached)
      # 134 - SIGABRT (PhysX C++ aborted before main.py could re-exec)
      # 139 - SIGSEGV (CUDA driver segfault)
      attempt=$((attempt + 1))
      if [[ "$attempt" -ge "$MAX_BASH_RETRIES" ]]; then
        echo "[eval_policy] giving up after $attempt restart(s) (rc=$rc, run_id=${ROBODOJO_RUN_ID})" >&2
        exit "$rc"
      fi
      echo "[eval_policy] python exited rc=$rc, restarting ($attempt/$MAX_BASH_RETRIES, run_id=${ROBODOJO_RUN_ID})..." >&2
      sleep 5
      ;;
    *)
      exit "$rc"
      ;;
  esac
done
