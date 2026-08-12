#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOWNLOAD_ROOT="${PROJECT_ROOT}/.hf_download_cache/robodojo_demo"

REPO_ID="${HF_REPO_ID:-DaMiTian/RoboDojo_demo_data}"
REMOTE_SUBDIR="${HF_REMOTE_SUBDIR:-archives/robodojo_tmp}"
HF_REVISION="${HF_REVISION:-main}"

DATA_ROOT="${PROJECT_ROOT}/data"
DATASET_NAME="${ROBODOJO_DEMO_DATASET:-RoboDojo_demo}"
TARGET_DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"
TARGET_ENV_CFG_DIR="${PROJECT_ROOT}/env_cfg"

echo "==> Project root: ${PROJECT_ROOT}"
echo "==> Demo data dir: data/${DATASET_NAME}"
echo "==> Repo: ${REPO_ID}"
echo "==> Remote subdir: ${REMOTE_SUBDIR}"

mkdir -p "${DOWNLOAD_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 未找到" >&2
	exit 1
fi

echo "==> Ensuring Python dependencies"
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

echo "==> Downloading archive parts from Hugging Face"
REPO_ID="${REPO_ID}" REMOTE_SUBDIR="${REMOTE_SUBDIR}" HF_REVISION="${HF_REVISION}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
remote_subdir = os.environ["REMOTE_SUBDIR"].strip("/")
revision = os.environ["HF_REVISION"]
download_root = Path(os.environ["DOWNLOAD_ROOT"])

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

patterns = [f"{remote_subdir}/*"] if remote_subdir else ["*"]

snapshot_download(
		repo_id=repo_id,
		repo_type="dataset",
		revision=revision,
		local_dir=str(download_root),
		local_dir_use_symlinks=False,
		resume_download=True,
		allow_patterns=patterns,
)

print(f"Downloaded to: {download_root}")
PY

ARCHIVE_DIR="${DOWNLOAD_ROOT}/${REMOTE_SUBDIR}"
if [[ ! -d "${ARCHIVE_DIR}" ]]; then
	echo "未找到下载目录: ${ARCHIVE_DIR}" >&2
	exit 1
fi

echo "==> Verifying downloaded files"
if [[ -f "${ARCHIVE_DIR}/SHA256SUMS" ]]; then
	(
		cd "${ARCHIVE_DIR}"
		sha256sum -c SHA256SUMS
	)
else
	echo "警告: 未发现 SHA256SUMS，跳过校验"
fi

shopt -s nullglob
parts=("${ARCHIVE_DIR}"/*.part-*)
shopt -u nullglob

if [[ ${#parts[@]} -eq 0 ]]; then
	echo "未找到分片文件: ${ARCHIVE_DIR}/*.part-*" >&2
	exit 1
fi

first_part="$(basename "${parts[0]}")"
archive_name="${first_part%.part-*}"
archive_path="${ARCHIVE_DIR}/${archive_name}"
extract_root="${DOWNLOAD_ROOT}/extracted"

echo "==> Reassembling archive: ${archive_name}"
cat "${ARCHIVE_DIR}"/*.part-* > "${archive_path}"

rm -rf "${extract_root}"
mkdir -p "${extract_root}"

echo "==> Extracting selected paths"
case "${archive_name}" in
	*.tar.zst)
		tar -I zstd -xf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*.tar.gz)
		tar -xzf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*.tar)
		tar -xf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*)
		echo "不支持的压缩格式: ${archive_name}" >&2
		exit 1
		;;
esac

if [[ ! -d "${extract_root}/tmp/data" || ! -d "${extract_root}/tmp/env_cfg" ]]; then
	echo "压缩包内未找到 tmp/data 或 tmp/env_cfg" >&2
	exit 1
fi

echo "==> Restoring demo data -> data/${DATASET_NAME} (not data/RoboDojo)"
src_data="${extract_root}/tmp/data"
mkdir -p "${DATA_ROOT}"
rm -rf "${TARGET_DATA_DIR}"

# Archives often contain tmp/data/RoboDojo/...; install uniformly to data/RoboDojo_demo/ to avoid conflicts with the full RoboDojo dataset
if [[ -d "${src_data}/${DATASET_NAME}" ]]; then
	mv "${src_data}/${DATASET_NAME}" "${TARGET_DATA_DIR}"
elif [[ -d "${src_data}/RoboDojo" ]]; then
	mv "${src_data}/RoboDojo" "${TARGET_DATA_DIR}"
else
	mv "${src_data}" "${TARGET_DATA_DIR}"
fi

if [[ -e "${TARGET_ENV_CFG_DIR}" ]]; then
	echo "env_cfg 已存在，为避免覆盖请先手动处理: ${TARGET_ENV_CFG_DIR}" >&2
	exit 1
fi
mv "${extract_root}/tmp/env_cfg" "${TARGET_ENV_CFG_DIR}"

echo "==> Done"
du -sh "${TARGET_DATA_DIR}" "${TARGET_ENV_CFG_DIR}"