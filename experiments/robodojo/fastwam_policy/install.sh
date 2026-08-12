#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-fastwam}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${FASTWAM_ROOT:-}" ]]; then
    FASTWAM_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${FASTWAM_ROOT}" || ! -d "${FASTWAM_ROOT}/src/fastwam" ]]; then
    echo "FASTWAM_ROOT must point to the FastWAM repository root." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

conda install -c conda-forge "ffmpeg>=6,<7" -y
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install opencv-python
pip install -e "${FASTWAM_ROOT}"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("XPolicyLab") else 1)
PY
then
    echo "[FastWAM] XPolicyLab is already importable."
else
    echo "[FastWAM] Install XPolicyLab separately in this environment before RoboDojo evaluation:"
    echo "  pip install -e third_party/RoboDojo/XPolicyLab"
fi
