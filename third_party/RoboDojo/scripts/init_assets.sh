#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Helpers
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

# Hugging Face dataset repo. The remote Assets folder is stored at repo root:
#   hf://datasets/RoboDojo-Benchmark/RoboDojo/Assets/
HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPO_URL="${HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO_ID}}"

TARGET_DIR="${CURRENT_DIR}/Assets"
ASSET_CACHE_DIR="${CURRENT_DIR}/.cache/robodojo_assets_repo"
REQUIRED_ASSET_SUBDIRS=(Robots Object Material Eval_Layout)

assets_ready() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    return 1
  fi

  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    if [[ ! -d "${TARGET_DIR}/${subdir}" ]]; then
      return 1
    fi
  done
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

clone_asset_repo() {
  info "Cloning sparse asset repo into '${ASSET_CACHE_DIR}'..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse "${HF_REPO_URL}" "${ASSET_CACHE_DIR}"
}

archive_asset_cache() {
  local cache_dir="$1"
  local partial_dir="${cache_dir}.partial.$(date +%Y%m%d_%H%M%S)"
  warn "Moving existing asset cache to '${partial_dir}'."
  mv "${cache_dir}" "${partial_dir}"
}

download_assets() {
  info "Repo root: ${CURRENT_DIR}"
  info "Assets target: ${TARGET_DIR}"
  info "HF repo: ${HF_REPO_ID} (revision=${HF_REVISION})"

  if assets_ready; then
    warn "'${TARGET_DIR}' already exists, skipping..."
    return 0
  fi

  if [[ -d "${TARGET_DIR}" ]]; then
    local partial_dir="${TARGET_DIR}.partial.$(date +%Y%m%d_%H%M%S)"
    warn "'${TARGET_DIR}' exists but is incomplete."
    warn "Moving incomplete directory to '${partial_dir}' before recreating Assets."
    mv "${TARGET_DIR}" "${partial_dir}"
  fi

  mkdir -p "$(dirname "${ASSET_CACHE_DIR}")"

  if [[ ! -d "${ASSET_CACHE_DIR}/.git" ]]; then
    clone_asset_repo
  else
    if [[ -n "$(git -C "${ASSET_CACHE_DIR}" config --get remote.origin.promisor || true)" ]]; then
      warn "Existing cache was created as a partial clone and may hit Hugging Face promisor fetch errors."
      archive_asset_cache "${ASSET_CACHE_DIR}"
      clone_asset_repo
    else
      info "Updating sparse asset repo cache..."
      if ! git -C "${ASSET_CACHE_DIR}" fetch --depth 1 origin "${HF_REVISION}"; then
        warn "Failed to update existing asset cache."
        archive_asset_cache "${ASSET_CACHE_DIR}"
        clone_asset_repo
      fi
    fi
  fi

  git -C "${ASSET_CACHE_DIR}" sparse-checkout set Assets
  git -C "${ASSET_CACHE_DIR}" checkout "${HF_REVISION}"

  info "Pulling only Assets/** LFS objects..."
  git -C "${ASSET_CACHE_DIR}" lfs install --local >/dev/null
  git -C "${ASSET_CACHE_DIR}" lfs pull --include="Assets/**" --exclude=""

  ln -s "${ASSET_CACHE_DIR}/Assets" "${TARGET_DIR}"
}

verify_assets() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    error "Expected '${TARGET_DIR}' after download, but it was not created."
  fi

  local subdir
  for subdir in "${REQUIRED_ASSET_SUBDIRS[@]}"; do
    if [[ ! -d "${TARGET_DIR}/${subdir}" ]]; then
      error "Asset subdir '${subdir}' missing under ${TARGET_DIR}; download incomplete."
    fi
  done
}

check_download_tools
download_assets
verify_assets

info "Assets directory is ready: ${TARGET_DIR}"
