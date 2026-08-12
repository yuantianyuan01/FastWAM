#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Helpers
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

# Hugging Face dataset repo. The remote data folders are stored under:
#   hf://datasets/RoboDojo-Benchmark/RoboDojo/data/
HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPO_URL="${HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO_ID}}"
MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-master}"
MODELSCOPE_REPO_URL="${MODELSCOPE_REPO_URL:-https://modelscope.cn/datasets/${MODELSCOPE_REPO_ID}}"
MODELSCOPE_DATA_ROOT="${MODELSCOPE_DATA_ROOT:-data}"

SOURCE="${1:-}"
DATA_TYPE="${2:-}"
DATA_ROOT="${ROBO_DOJO_DATA_ROOT:-${CURRENT_DIR}/data}"

usage() {
  cat <<EOF
RoboDojo data downloader

Usage:
  bash scripts/RoboDojo/download_data.sh <source> <format>

Sources:
  huggingface
  modelscope

Examples:
  bash scripts/RoboDojo/download_data.sh huggingface lerobot_v3.0
  bash scripts/RoboDojo/download_data.sh modelscope hdf5

Available data formats:
  lerobot_v3.0  120GB  LeRobot v3.0 format. State/action contain joint-only
                     values and do not include end-effector (ee) values.
  lerobot_v3.0_ee  120GB  LeRobot v3.0 format. State/action include
                          end-effector (ee) values. Availability depends on source.
  lerobot_v2.1   64GB  LeRobot v2.1 format. State/action contain joint-only
                     values and do not include end-effector (ee) values.
  hdf5          523GB  HDF5 format. Contains the full RoboDojo data, including
                     all available state/action fields.
  depth          4.5TB  HDF5 with depth observations. Folder: RoboDojo_depth.
                     Availability depends on source.
  real           273GB  Real-world dataset for testing and evaluation.

Environment overrides:
  HF_REPO_ID, HF_REPO_URL, HF_REVISION
  MODELSCOPE_REPO_ID, MODELSCOPE_REPO_URL, MODELSCOPE_REVISION
  MODELSCOPE_DATA_ROOT (default: data)
  ROBO_DOJO_DATA_ROOT
EOF
}

resolve_source() {
  case "${SOURCE,,}" in
    huggingface)
      SOURCE="huggingface"
      REPO_ID="${HF_REPO_ID}"
      REPO_URL="${HF_REPO_URL}"
      REPO_REVISION="${HF_REVISION}"
      REMOTE_DATA_ROOT="data"
      ;;
    modelscope)
      SOURCE="modelscope"
      REPO_ID="${MODELSCOPE_REPO_ID}"
      REPO_URL="${MODELSCOPE_REPO_URL}"
      REPO_REVISION="${MODELSCOPE_REVISION}"
      REMOTE_DATA_ROOT="${MODELSCOPE_DATA_ROOT#/}"
      REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT%/}"
      ;;
    *)
      error "Invalid source: ${SOURCE}. Expected 'huggingface' or 'modelscope'."
      ;;
  esac
}

check_download_tools() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda env list | grep -q "^RoboDojo "; then
      info "Activating conda environment 'RoboDojo'..."
      source "$HOME/miniconda3/bin/activate" RoboDojo 2>/dev/null || conda activate RoboDojo
    fi
  fi

  if ! command -v git >/dev/null 2>&1; then
    error "git not found. Please install git first."
  fi

  if ! git lfs version >/dev/null 2>&1; then
    error "git-lfs not found. Please install git-lfs first."
  fi
}

resolve_data_type() {
  case "${DATA_TYPE}" in
    lerobot_v3.0)
      DATA_SIZE="120GB"
      DATA_DESCRIPTION="LeRobot v3.0, joint-only state/action, no ee values"
      DATA_DIR_NAME="RoboDojo_lerobot_v30_video"
      ;;
    lerobot_v3.0_ee)
      DATA_SIZE="120GB"
      DATA_DESCRIPTION="LeRobot v3.0, state/action include ee values"
      DATA_DIR_NAME="RoboDojo_ee_lerobot_v30_video"
      ;;
    lerobot_v2.1)
      DATA_SIZE="64GB"
      DATA_DESCRIPTION="LeRobot v2.1, joint-only state/action, no ee values"
      DATA_DIR_NAME="RoboDojo_lerobot_v21_video"
      ;;
    hdf5)
      DATA_SIZE="523GB"
      DATA_DESCRIPTION="HDF5, full RoboDojo data with all available fields"
      DATA_DIR_NAME="RoboDojo"
      ;;
    depth|RoboDojo_depth|hdf5_w_depth)
      DATA_TYPE="depth"
      DATA_SIZE="4.5TB"
      DATA_DESCRIPTION="HDF5 with depth observations"
      DATA_DIR_NAME="RoboDojo_depth"
      ;;
    real)
      DATA_SIZE="273GB"
      DATA_DESCRIPTION="Real-world dataset for testing and evaluation"
      DATA_DIR_NAME="RoboDojo_real"
      ;;
    *)
      error "Invalid data format: ${DATA_TYPE}. Run without arguments to show available formats."
      ;;
  esac

  if [[ -n "${REMOTE_DATA_ROOT}" ]]; then
    REMOTE_DIR="${REMOTE_DATA_ROOT}/${DATA_DIR_NAME}"
  else
    REMOTE_DIR="${DATA_DIR_NAME}"
  fi
  TARGET_DIR="${DATA_ROOT}/${DATA_DIR_NAME}"
  DATA_CACHE_DIR="${CURRENT_DIR}/.cache/robodojo_data_${SOURCE}_${DATA_TYPE}_repo"
}

data_ready() {
  [[ -d "${TARGET_DIR}" && -f "${TARGET_DIR}/.download_complete" ]]
}

clone_data_repo() {
  info "Cloning sparse data repo into '${DATA_CACHE_DIR}'..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse --branch "${REPO_REVISION}" \
    "${REPO_URL}" "${DATA_CACHE_DIR}"
}

archive_path() {
  local path="$1"
  local partial_path="${path}.partial.$(date +%Y%m%d_%H%M%S)"
  warn "Moving existing path to '${partial_path}'."
  mv "${path}" "${partial_path}"
}

download_data() {
  info "Repo root: ${CURRENT_DIR}"
  info "Data target: ${TARGET_DIR}"
  info "Source: ${SOURCE}"
  info "Repository: ${REPO_ID} (revision=${REPO_REVISION})"
  info "Data format: ${DATA_TYPE} (${DATA_SIZE})"
  info "${DATA_DESCRIPTION}"

  if data_ready; then
    warn "'${TARGET_DIR}' already exists and is marked complete, skipping..."
    return 0
  fi

  mkdir -p "${DATA_ROOT}" "$(dirname "${DATA_CACHE_DIR}")"

  if [[ -e "${TARGET_DIR}" || -L "${TARGET_DIR}" ]]; then
    warn "'${TARGET_DIR}' exists but is not marked complete."
    archive_path "${TARGET_DIR}"
  fi

  if [[ ! -d "${DATA_CACHE_DIR}/.git" ]]; then
    clone_data_repo
  else
    if [[ -n "$(git -C "${DATA_CACHE_DIR}" config --get remote.origin.promisor || true)" ]]; then
      warn "Existing cache was created as a partial clone and may hit Hugging Face promisor fetch errors."
      archive_path "${DATA_CACHE_DIR}"
      clone_data_repo
    else
      info "Updating sparse data repo cache..."
      if ! GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" fetch --depth 1 origin "${REPO_REVISION}"; then
        warn "Failed to update existing data cache."
        archive_path "${DATA_CACHE_DIR}"
        clone_data_repo
      fi
    fi
  fi

  info "Configuring sparse checkout for ${REMOTE_DIR}/** (without downloading LFS objects)..."
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" sparse-checkout set "${REMOTE_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" \
    -c advice.detachedHead=false checkout --quiet --force --detach FETCH_HEAD 2>/dev/null || \
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" checkout --quiet --force "${REPO_REVISION}"

  if ! git -C "${DATA_CACHE_DIR}" cat-file -e "HEAD:${REMOTE_DIR}" 2>/dev/null; then
    error "Remote folder '${REMOTE_DIR}' is not available from source '${SOURCE}' in ${REPO_ID}."
  fi

  info "Pulling only ${REMOTE_DIR}/** LFS objects..."
  git -C "${DATA_CACHE_DIR}" lfs install --local >/dev/null
  git -C "${DATA_CACHE_DIR}" lfs pull --include="${REMOTE_DIR}/**" --exclude=""

  if [[ ! -d "${DATA_CACHE_DIR}/${REMOTE_DIR}" ]]; then
    error "Remote folder '${REMOTE_DIR}' was not found in ${REPO_ID}."
  fi

  ln -s "${DATA_CACHE_DIR}/${REMOTE_DIR}" "${TARGET_DIR}"
  cat > "${TARGET_DIR}/.download_complete" <<EOF
source=${SOURCE}
repo_id=${REPO_ID}
revision=${REPO_REVISION}
remote_dir=${REMOTE_DIR}
data_type=${DATA_TYPE}
data_dir_name=${DATA_DIR_NAME}
size=${DATA_SIZE}
EOF
}

verify_data() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    error "Expected '${TARGET_DIR}' after download, but it was not created."
  fi

  if [[ ! -f "${TARGET_DIR}/.download_complete" ]]; then
    error "Expected '${TARGET_DIR}/.download_complete' after download, but it was not created."
  fi
}

if [[ -z "${SOURCE}" && -z "${DATA_TYPE}" ]]; then
  usage
  exit 0
fi

if [[ "${SOURCE}" == "-h" || "${SOURCE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 2 ]]; then
  usage
  exit 1
fi

resolve_source
resolve_data_type
check_download_tools
download_data
verify_data

info "Data directory is ready: ${TARGET_DIR}"
