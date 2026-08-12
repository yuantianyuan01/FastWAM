#!/usr/bin/env bash
# Download the RoboDojo dataset from Hugging Face to XPolicyLab/data/
#
# Usage:
#   bash scripts/RoboDojo/download_robodojo_data.sh <type>
#
# Example:
#   bash scripts/RoboDojo/download_robodojo_data.sh hdf5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # XPolicyLab root directory
DATA_ROOT="${PROJECT_ROOT}/../data"

DATA_TYPE="${1:-}"   # Data format: lerobot_v3.0 / lerobot_v2.1 / hdf5 / demo / real

HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/Robodojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-16}"
HF_RETRY_WAIT="${HF_RETRY_WAIT:-60}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-10}"
HF_DOWNLOAD_TIMEOUT="${HF_DOWNLOAD_TIMEOUT:-300}"

usage() {
	cat <<'EOF'
Usage: bash scripts/RoboDojo/download_robodojo_data.sh <type>

Types:
  lerobot_v3.0   -> RoboDojo_lerobot_v30_video
  lerobot_v2.1   -> RoboDojo_lerobot_v21_video
  hdf5           -> RoboDojo
  demo           -> demo
  real           -> RoboDojo_real

Example:
  bash scripts/RoboDojo/download_robodojo_data.sh hdf5
  bash scripts/RoboDojo/download_robodojo_data.sh real

Environment:
  HF_REPO_ID            default: RoboDojo-Benchmark/Robodojo
  HF_REVISION           default: main
  HF_MAX_WORKERS        parallel file downloads (default: 16)
  HF_RETRY_WAIT         seconds to wait before retry (default: 60)
  HF_MAX_RETRIES        max retry attempts (default: 10)
  HF_DOWNLOAD_TIMEOUT   per-file timeout in seconds (default: 300)
  HF_FORCE_DOWNLOAD     set to 1 to re-download even if complete
EOF
}

if [[ -z "${DATA_TYPE}" ]]; then
	usage
	exit 1
fi

mkdir -p "${DATA_ROOT}"

resolve_data_paths() {
	case "${DATA_TYPE}" in
		lerobot_v3.0)
			DATA_DIR_NAME="RoboDojo_lerobot_v30_video"
			;;
		lerobot_v2.1)
			DATA_DIR_NAME="RoboDojo_lerobot_v21_video"
			;;
		hdf5)
			DATA_DIR_NAME="RoboDojo"
			;;
		demo)
			DATA_DIR_NAME="demo"
			;;
		real|RoboDojo_real)
			DATA_DIR_NAME="RoboDojo_real"
			;;
		*)
			echo "Invalid type: ${DATA_TYPE}" >&2
			return 1
			;;
	esac

	REMOTE_DIR="data/${DATA_DIR_NAME}"
	TARGET_DIR="${DATA_ROOT}/${DATA_DIR_NAME}"
}

ensure_hf_deps() {
	if ! command -v python3 >/dev/null 2>&1; then
		echo "python3 not found" >&2
		exit 1
	fi

	python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
		name for name in ("huggingface_hub", "hf_transfer")
		if importlib.util.find_spec(name) is None
]
if missing:
		subprocess.check_call([
				sys.executable,
				"-m",
				"pip",
				"install",
				"-U",
				"huggingface_hub",
				"hf_transfer",
		])
PY
}

# HuggingFace: snapshot_download + allow_patterns + hf_transfer parallel batch download for large files
download_hf_folder() {
	local repo_id="$1"
	local remote_dir="$2"
	local target_dir="$3"
	local complete_marker="${target_dir}/.download_complete"

	if [[ -f "${complete_marker}" && "${HF_FORCE_DOWNLOAD:-0}" != "1" ]]; then
		echo "==> Target already exists, skip: ${target_dir}"
		return 0
	fi

	if [[ -d "${target_dir}" && "${HF_FORCE_DOWNLOAD:-0}" != "1" ]]; then
		echo "==> Resuming partial download: ${target_dir}"
	elif [[ -d "${target_dir}" ]]; then
		echo "==> Force re-download: ${target_dir}"
		rm -rf "${target_dir}"
	fi

	ensure_hf_deps

	# hf_transfer must be enabled before importing huggingface_hub
	export HF_HUB_ENABLE_HF_TRANSFER=1
	export HF_HUB_DOWNLOAD_TIMEOUT="${HF_DOWNLOAD_TIMEOUT}"
	export HF_TRANSFER_MAX_CONCURRENT_DOWNLOADS="${HF_MAX_WORKERS}"

	echo "==> Downloading ${remote_dir} from HuggingFace ${repo_id}"
	echo "==> Parallel workers: ${HF_MAX_WORKERS}, timeout: ${HF_DOWNLOAD_TIMEOUT}s"

	HF_REPO_ID="${repo_id}" \
	HF_REVISION="${HF_REVISION}" \
	REMOTE_DIR="${remote_dir}" \
	DATA_ROOT="${DATA_ROOT}" \
	TARGET_DIR="${target_dir}" \
	HF_MAX_WORKERS="${HF_MAX_WORKERS}" \
	HF_RETRY_WAIT="${HF_RETRY_WAIT}" \
	HF_MAX_RETRIES="${HF_MAX_RETRIES}" \
	python3 - <<'PY'
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

repo_id = os.environ["HF_REPO_ID"]
revision = os.environ["HF_REVISION"]
remote_dir = os.environ["REMOTE_DIR"]
data_root = Path(os.environ["DATA_ROOT"])
target_dir = Path(os.environ["TARGET_DIR"])
complete_marker = target_dir / ".download_complete"
max_workers = int(os.environ["HF_MAX_WORKERS"])
retry_wait = int(os.environ["HF_RETRY_WAIT"])
max_retries = int(os.environ["HF_MAX_RETRIES"])

data_root.mkdir(parents=True, exist_ok=True)
download_root = data_root.parent

def is_retryable(exc: BaseException) -> bool:
	err = str(exc).lower()
	if isinstance(exc, HfHubHTTPError) and exc.response is not None:
		if exc.response.status_code in (408, 429, 500, 502, 503, 504):
			return True
	return any(
			token in err
			for token in (
					"429",
					"too many requests",
					"rate limit",
					"timeout",
					"timed out",
					"connection reset",
					"connection aborted",
					"temporary failure",
					"503",
					"502",
			)
	)

attempt = 0
while True:
	attempt += 1
	try:
		print(f"==> Download attempt {attempt}/{max_retries} (max_workers={max_workers})")
		snapshot_download(
				repo_id=repo_id,
				repo_type="dataset",
				revision=revision,
				local_dir=str(download_root),
				allow_patterns=[f"{remote_dir}/**"],
				max_workers=max_workers,
		)
		break
	except Exception as exc:
		if attempt >= max_retries or not is_retryable(exc):
			raise
		print(
				f"==> Transient error, retry in {retry_wait}s: {exc}",
				file=sys.stderr,
		)
		time.sleep(retry_wait)

if not target_dir.is_dir():
	print(f"Remote folder not found: {remote_dir}", file=sys.stderr)
	sys.exit(1)

complete_marker.write_text(f"repo_id={repo_id}\nrevision={revision}\n")
print(f"==> Saved to {target_dir}")
PY

	if [[ ! -d "${target_dir}" ]]; then
		echo "Remote folder not found: ${remote_dir}" >&2
		exit 1
	fi

	echo "==> Download complete: ${target_dir}"
}

if ! resolve_data_paths; then
	usage
	exit 1
fi

download_hf_folder "${HF_REPO_ID}" "${REMOTE_DIR}" "${TARGET_DIR}"
