#!/usr/bin/env bash
set -e
export PIP_USER=0
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_RL_FRAMEWORK="${ISAACLAB_RL_FRAMEWORK:-none}"
# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }
pip_install() {
  python -m pip install "$@"
}
pip_install_with_isaac_constraints() {
  local constraints
  constraints="$(mktemp)"
  cat >"$constraints" <<'EOF'
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
EOF
  if pip_install "$@" --constraint "$constraints"; then
    rm -f "$constraints"
  else
    local status=$?
    rm -f "$constraints"
    return "$status"
  fi
}
ensure_torch_cuda_stack() {
  if ! python -m pip show torchaudio &>/dev/null; then
    warn "    torchaudio missing, installing PyTorch cu128 stack..."
    pip_install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
  fi
}
pin_runtime_deps() {
  pip_install \
    "numpy==1.26.0" \
    "packaging==23.0" \
    "typing_extensions==4.12.2" \
    "filelock==3.13.1" \
    "websockets==12.0" \
    "click==8.1.7" \
    "psutil==5.9.8" \
    "wheel==0.45.1" \
    "starlette==0.45.3" \
    "scipy==1.15.3" \
    "warp-lang==1.11.0" \
    "onnx>=1.18,<1.22" \
    "ipython<9" \
    "virtualenv==20.30.0"
  # virtualenv 21.x pulled this in; virtualenv 20.x doesn't need it and it
  # requires filelock>=3.15.4 (conflicts with isaacsim-core's filelock==3.13.1)
  python -m pip uninstall -y python-discovery 2>/dev/null || true
}
# ── Step functions ─────────────────────────────────────────────────────────────
setup_system() {
  local missing=()
  for pkg in cmake build-essential ffmpeg; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    info "[0/7] Installing system dependencies: ${missing[*]}..."
    if [ "$(id -u)" -ne 0 ]; then
      warn "    You need sudo privileges to install system packages. Please enter your password."
      sudo apt-get update && sudo apt-get install -y "${missing[@]}"
    else
      apt-get update && apt-get install -y "${missing[@]}"
    fi
  else
    warn "[0/7] System dependencies already installed, skipping..."
  fi
}
setup_conda() {
  if ! command -v conda &>/dev/null; then
    info "[1/7] Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -q
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    rm -f Miniconda3-latest-Linux-x86_64.sh
    "$HOME/miniconda3/bin/conda" init bash
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    info "    Accepting Anaconda Terms of Service..."
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  else
    warn "[1/7] Conda already installed, skipping..."
    eval "$(conda shell.bash hook)"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  fi
  if ! conda env list | grep -q "^RoboDojo "; then
    info "    Creating conda environment 'RoboDojo' (Python 3.11)..."
    conda create -n RoboDojo python=3.11 -y
  else
    warn "    Conda environment 'RoboDojo' already exists, skipping..."
  fi
  info "    Activating environment 'RoboDojo'..."
  source "$HOME/miniconda3/bin/activate" RoboDojo 2>/dev/null || conda activate RoboDojo
  [[ "$CONDA_DEFAULT_ENV" == "RoboDojo" ]] || error "Failed to activate conda environment 'RoboDojo'"
}
setup_base_deps() {
  info "[2/7] Installing base pip dependencies..."
  pip_install -r "$CURRENT_DIR/scripts/requirements.txt"
  pip_install opencv-python-headless==4.11.0.86 pillow matplotlib "scipy==1.15.3" scikit-learn
  pip_install numpy==1.26.0
}
setup_submodules() {
  cd "$CURRENT_DIR" || exit 1
  local subs=(third_party/IsaacLab third_party/curobo XPolicyLab)
  info "[3/7] Syncing and updating submodules from remote..."
  git submodule sync "${subs[@]}"
  for sub in "${subs[@]}"; do
    info "    Updating ${sub} from remote..."
    git submodule update --init --remote --progress "$sub" || {
      [ "$sub" = "XPolicyLab" ] && error "Failed to clone XPolicyLab. Ensure HTTPS auth (e.g. gh auth login)."
      error "Failed to update $sub."
    }
  done
  [ -f "XPolicyLab/client_server/ws/model_client.py" ] \
    || error "XPolicyLab init failed. Check repo access."
}
setup_isaacsim() {
  if ! python -m pip show isaacsim 2>/dev/null | grep -q "5.1.0"; then
    info "[4/7] Installing PyTorch + IsaacSim 5.1..."
    pip_install --upgrade pip
    pip_install "numpy==1.26.0" "typing_extensions==4.12.2" "filelock==3.13.1"
    pip_install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
    pip_install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    pin_runtime_deps
  else
    warn "[4/7] IsaacSim 5.1.0 already installed, skipping..."
    ensure_torch_cuda_stack
  fi
}
setup_isaaclab() {
  cd "$CURRENT_DIR" || exit 1
  if ! python -m pip show isaaclab &>/dev/null; then
    info "[5/7] Installing IsaacLab (rl-framework: ${ISAACLAB_RL_FRAMEWORK})..."
    cd third_party/IsaacLab || error "third_party/IsaacLab not found"
    export OMNI_KIT_ACCEPT_EULA=YES
    # isaaclab.sh calls `tabs`; fails when TERM=dumb (CI / piped shells)
    export TERM=xterm-256color
    ./isaaclab.sh --install "$ISAACLAB_RL_FRAMEWORK"
    cd "$CURRENT_DIR"
    ensure_torch_cuda_stack
    pin_runtime_deps
  else
    warn "[5/7] IsaacLab already installed, skipping..."
  fi
}
setup_curobo() {
  cd "$CURRENT_DIR" || exit 1
  local need_install=1
  if python -m pip show nvidia-curobo &>/dev/null; then
    if python - <<'PY' &>/dev/null
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
PY
    then
      need_install=0
    fi
  fi
  if [ "$need_install" -eq 1 ]; then
    info "[6/7] Installing CuRobo..."
    cd third_party/curobo || error "third_party/curobo not found"
    python -m pip uninstall -y nvidia-curobo curobo 2>/dev/null || true
    pip_install_with_isaac_constraints -e ".[cu12]" --no-build-isolation
    cd "$CURRENT_DIR"
    pin_runtime_deps
  else
    warn "[6/7] CuRobo v2 already installed and importable, skipping..."
    pin_runtime_deps
  fi
}
# ── Entry point ────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 [-i | --from <step>]"
  echo "  -i, --install          Full install (all steps)"
  echo "  --from <step>          Resume from a specific step:"
  echo "                           system | conda | base_deps | submodules | isaacsim | isaaclab | curobo"
  echo "  ISAACLAB_RL_FRAMEWORK  IsaacLab RL extras to install (default: none; use all/sb3/skrl/etc. if needed)"
  echo "  -h, --help             Show this help"
}
run_from() {
  local from="$1"
  local steps=(system conda base_deps submodules isaacsim isaaclab curobo)
  local start=0
  local found=0
  for i in "${!steps[@]}"; do
    if [ "${steps[$i]}" == "$from" ]; then
      start=$i
      found=1
      break
    fi
  done
  [ "$found" -eq 1 ] || error "Unknown step '$from'. Valid: ${steps[*]}"
  if [ "$from" != "system" ] && [ "$from" != "conda" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    source "$HOME/miniconda3/bin/activate" RoboDojo 2>/dev/null || conda activate RoboDojo 2>/dev/null || true
  fi
  for i in "${!steps[@]}"; do
    if [ "$i" -ge "$start" ]; then
      "setup_${steps[$i]}"
    fi
  done
}
case "${1:-}" in
  -h|--help)
    usage
    ;;
  -i|--install)
    run_from system
    info "Develop environment setup completed."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  --from)
    [ -n "${2:-}" ] || { echo "Error: --from requires a step name"; usage; exit 1; }
    run_from "$2"
    info "Resumed from '$2' — done."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  *)
    usage
    exit 1
    ;;
esac

cd "$CURRENT_DIR" || exit