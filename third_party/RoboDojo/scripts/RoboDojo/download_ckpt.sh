#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m[WARNING] $*\e[0m" >&2; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }

# Checkpoints uploaded by huggingface/upload_ckpt.py live under:
#   hf://datasets/RoboDojo-Benchmark/RoboDojo/ckpt/RoboDojo/<POLICY>/
HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPO_URL="${HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO_ID}}"
MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-master}"
MODELSCOPE_REPO_URL="${MODELSCOPE_REPO_URL:-https://modelscope.cn/datasets/${MODELSCOPE_REPO_ID}}"
MODELSCOPE_CKPT_ROOT="${MODELSCOPE_CKPT_ROOT:-ckpt/RoboDojo}"

SOURCE="${1:-}"
POLICY_INPUT="${2:-}"
POLICY_ROOT="${ROBO_DOJO_POLICY_ROOT:-${CURRENT_DIR}/XPolicyLab/policy}"

usage() {
  cat <<EOF
RoboDojo policy checkpoint downloader

Usage:
  bash scripts/RoboDojo/download_ckpt.sh <source> <policy>

Examples:
  bash scripts/RoboDojo/download_ckpt.sh huggingface Pi_0
  bash scripts/RoboDojo/download_ckpt.sh modelscope pi_0  # case-insensitive match

The selected policy is downloaded to:
  XPolicyLab/policy/<POLICY>/checkpoints
EOF
}

resolve_source() {
  case "${SOURCE,,}" in
    huggingface)
      SOURCE="huggingface"
      REPO_ID="${HF_REPO_ID}"
      REPO_URL="${HF_REPO_URL}"
      REPO_REVISION="${HF_REVISION}"
      REMOTE_CKPT_ROOT="ckpt/RoboDojo"
      ;;
    modelscope)
      SOURCE="modelscope"
      REPO_ID="${MODELSCOPE_REPO_ID}"
      REPO_URL="${MODELSCOPE_REPO_URL}"
      REPO_REVISION="${MODELSCOPE_REVISION}"
      REMOTE_CKPT_ROOT="${MODELSCOPE_CKPT_ROOT#/}"
      REMOTE_CKPT_ROOT="${REMOTE_CKPT_ROOT%/}"
      [[ -n "${REMOTE_CKPT_ROOT}" ]] || error "MODELSCOPE_CKPT_ROOT cannot be empty."
      ;;
    *)
      error "Invalid source: ${SOURCE}. Expected 'huggingface' or 'modelscope'."
      ;;
  esac

  CKPT_CACHE_DIR="${ROBO_DOJO_CKPT_CACHE:-${CURRENT_DIR}/.cache/robodojo_ckpt_${SOURCE}_repo}"
  if [[ "${CKPT_CACHE_DIR}" != /* ]]; then
    CKPT_CACHE_DIR="${PWD}/${CKPT_CACHE_DIR}"
  fi
}

check_download_tools() {
  command -v git >/dev/null 2>&1 || error "git not found. Please install git first."
  git lfs version >/dev/null 2>&1 || error "git-lfs not found. Please install git-lfs first."
}

archive_path() {
  local path="$1"
  local partial_path="${path}.partial.$(date +%Y%m%d_%H%M%S)"
  warn "Moving existing path to '${partial_path}'."
  mv "${path}" "${partial_path}"
}

clone_ckpt_repo() {
  info "Cloning sparse checkpoint repo into '${CKPT_CACHE_DIR}'..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse --branch "${REPO_REVISION}" \
    "${REPO_URL}" "${CKPT_CACHE_DIR}"
}

prepare_ckpt_repo() {
  mkdir -p "$(dirname "${CKPT_CACHE_DIR}")"

  if [[ ! -d "${CKPT_CACHE_DIR}/.git" ]]; then
    clone_ckpt_repo
    return
  fi

  if [[ -n "$(git -C "${CKPT_CACHE_DIR}" config --get remote.origin.promisor || true)" ]]; then
    warn "Existing cache is a partial clone and may fail while fetching LFS objects."
    archive_path "${CKPT_CACHE_DIR}"
    clone_ckpt_repo
    return
  fi

  info "Updating checkpoint repo cache..."
  if ! GIT_LFS_SKIP_SMUDGE=1 git -C "${CKPT_CACHE_DIR}" fetch --quiet --depth 1 origin "${REPO_REVISION}"; then
    warn "Failed to update existing checkpoint cache."
    archive_path "${CKPT_CACHE_DIR}"
    clone_ckpt_repo
    return
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "${CKPT_CACHE_DIR}" \
    -c advice.detachedHead=false checkout --quiet --force --detach FETCH_HEAD
}

normalize_policy_name() {
  printf '%s' "${1,,}" | tr -cd '[:alnum:]'
}

# Mapping for remote ckpt/RoboDojo directory names that differ from their
# XPolicyLab/policy directory names. Keys are normalized by
# normalize_policy_name, so case, underscores, hyphens and dots do not matter.
declare -A POLICY_NAME_MAP=(
  [a1]="A1"
  [act]="ACT"
  [abotm0]="Abot_M0"
  [dm0]="Dexbotic_DM0"
  [dexora]="Dexora_1B"
  [eventvla]="EventVLA"
  [fastwam]="FastWAM"
  [go1]="GO1"
  [gr00tn17]="GR00T_N17"
  [galaxeavla]="GalaxeaVLA"
  [gigaworldpolicy]="GigaWorldPolicy"
  [hrdt]="H_RDT"
  [internvlaa1]="InternVLA_A1"
  [lda1b]="LDA_1B"
  [lingbotva]="LingBot_VA"
  [molmoact2]="MolmoACT2"
  [openvlaoft]="OpenVLA_OFT"
  [pi0]="Pi_0"
  [pi05]="Pi_05"
  [rdt1b]="RDT_1B"
  [smolvla]="SmolVLA"
  [spiritv15]="Spirit_v15"
  [starvlaalpha]="starVLA"
  [xvla]="X_VLA"
  [xwam]="X_WAM"
  [xiaomirobotics0]="Xiaomi_Robotics_0"
  [ahawam]="AHA_WAM"
  [hyvla]="Hy_Embodied_05_VLA"
)

resolve_local_policy() {
  local remote_policy="$1"
  local remote_key expected entry name
  local -a matches=()

  remote_key="$(normalize_policy_name "${remote_policy}")"
  expected="${POLICY_NAME_MAP[${remote_key}]:-${remote_policy}}"

  while IFS= read -r entry; do
    name="${entry##*/}"
    if [[ "$(normalize_policy_name "${name}")" == "$(normalize_policy_name "${expected}")" ]]; then
      matches+=("${name}")
    fi
  done < <(find "${POLICY_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | LC_ALL=C sort)

  if (( ${#matches[@]} == 0 )); then
    error "Remote checkpoint policy '${remote_policy}' maps to local '${expected}', but policy adapter '${POLICY_ROOT}/${expected}' does not exist. Install the adapter before downloading its checkpoints."
  fi
  (( ${#matches[@]} == 1 )) || error \
    "Mapped remote '${remote_policy}' to local '${expected}', but multiple matching policy directories exist under ${POLICY_ROOT}."
  printf '%s\n' "${matches[0]}"
}

resolve_remote_policy() {
  local requested="$1"
  local requested_key path name remote_key local_name tree_output
  local -a matches=()

  requested_key="$(normalize_policy_name "${requested}")"

  if ! tree_output="$(git -C "${CKPT_CACHE_DIR}" ls-tree -d --name-only "HEAD:${REMOTE_CKPT_ROOT}" 2>/dev/null)"; then
    error "Checkpoint root '${REMOTE_CKPT_ROOT}' was not found in ${REPO_ID} at revision ${REPO_REVISION}."
  fi

  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    name="${path#${REMOTE_CKPT_ROOT}/}"
    [[ "${name}" != */* ]] || continue
    remote_key="$(normalize_policy_name "${name}")"
    local_name="${POLICY_NAME_MAP[${remote_key}]:-}"
    if [[ "${remote_key}" == "${requested_key}" || ( -n "${local_name}" && "$(normalize_policy_name "${local_name}")" == "${requested_key}" ) ]]; then
      matches+=("${name}")
    fi
  done <<< "${tree_output}"

  if (( ${#matches[@]} == 0 )); then
    warn "${SOURCE} has no checkpoint policy matching '${requested}' under ${REMOTE_CKPT_ROOT}; nothing was downloaded."
    return 1
  fi
  if (( ${#matches[@]} > 1 )); then
    error "Ambiguous remote policy '${requested}': ${matches[*]}"
  fi
  printf '%s\n' "${matches[0]}"
}

download_ckpt() {
  local local_policy remote_policy remote_dir target_dir

  if ! remote_policy="$(resolve_remote_policy "${POLICY_INPUT}")"; then
    return 1
  fi
  local_policy="$(resolve_local_policy "${remote_policy}")"
  remote_dir="${REMOTE_CKPT_ROOT}/${remote_policy}"
  target_dir="${POLICY_ROOT}/${local_policy}/checkpoints"

  info "Matched policy: input='${POLICY_INPUT}', local='${local_policy}', remote='${remote_policy}'"
  info "Pulling only ${remote_dir}/** LFS objects..."

  # Keep previously downloaded policies checked out so their checkpoint
  # symlinks remain valid when another policy is downloaded.
  GIT_LFS_SKIP_SMUDGE=1 git -C "${CKPT_CACHE_DIR}" sparse-checkout add "${remote_dir}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${CKPT_CACHE_DIR}" checkout --quiet --force HEAD
  git -C "${CKPT_CACHE_DIR}" lfs install --local >/dev/null
  git -C "${CKPT_CACHE_DIR}" lfs pull --include="${remote_dir}/**" --exclude=""

  [[ -d "${CKPT_CACHE_DIR}/${remote_dir}" ]] || \
    error "Remote checkpoint folder '${remote_dir}' was not found after checkout."

  if [[ -e "${target_dir}" || -L "${target_dir}" ]]; then
    if [[ -L "${target_dir}" && "$(readlink -f "${target_dir}")" == "$(readlink -f "${CKPT_CACHE_DIR}/${remote_dir}")" ]]; then
      info "Checkpoint link already points to the downloaded policy; refresh completed."
      return
    fi
    archive_path "${target_dir}"
  fi

  ln -s "${CKPT_CACHE_DIR}/${remote_dir}" "${target_dir}"
  info "Checkpoints are ready: ${target_dir}"
}

if [[ -z "${SOURCE}" && -z "${POLICY_INPUT}" ]]; then
  usage
  exit 0
fi
[[ "$#" -eq 2 ]] || { usage; exit 1; }

resolve_source
check_download_tools
[[ -d "${POLICY_ROOT}" ]] || error "Policy root does not exist: ${POLICY_ROOT}"
prepare_ckpt_repo
download_ckpt
