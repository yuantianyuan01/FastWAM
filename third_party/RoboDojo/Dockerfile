# syntax=docker/dockerfile:1
# RoboDojo simulator / evaluation image (Ubuntu 22.04 + CUDA 12.8).
#
# Scope: this image contains ONLY the RoboDojo simulation-evaluation side
# (Isaac Sim, IsaacLab, CuRobo, the RoboDojo Python stack) plus the lightweight
# XPolicyLab client_server + policy deploy adapters used by src/eval_client.
# It deliberately does NOT install any policy-specific environment, dependency
# set, or checkpoints (e.g. GR00T_N17). The policy server runs OUTSIDE this
# container and is reached over TCP (see docker/README.md).
#
# Build:
#   docker build -t robodojo:cuda12.8 .
#
# Run (Linux host, policy server already listening on the host):
#   docker run --rm -it --gpus all --network host --ipc host \
#     -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
#     -v "$PWD/eval_result:/workspace/RoboDojo/eval_result" \
#     robodojo:cuda12.8 \
#     bash scripts/robodojo.sh client --task stack_bowls \
#       --policy-name GR00T_N17 --policy-host 127.0.0.1 --policy-port 9999 --eval-num 1

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/miniconda3/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    TERM=xterm-256color \
    PIP_USER=0 \
    PYTHONNOUSERSITE=1 \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# ── Optional China mirrors (build args; defaults = official upstreams) ─────────
# The image is unchanged for normal builds. On a China network, enable mirrors:
#   docker build \
#     --build-arg UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     --build-arg MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     --build-arg CONDA_CHANNEL_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda \
#     -t robodojo:cuda12.8 .
# (docker/smoke_docker.sh sets all of these when ROBODOJO_CN_MIRRORS=1.)
ARG UBUNTU_MIRROR=""
ARG PIP_INDEX_URL="https://pypi.org/simple"
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG CONDA_CHANNEL_MIRROR=""
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# ── System dependencies ──────────────────────────────────────────────────────
# setup_system() from scripts/install.sh (cmake, build-essential, ffmpeg) plus
# the headless OpenGL/EGL/Vulkan runtime libraries Isaac Sim needs, and
# netcat for the policy-server connectivity check documented in docker/README.md.
RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
        sed -i "s@http://archive.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g; s@http://security.ubuntu.com/ubuntu@http://${UBUNTU_MIRROR}/ubuntu@g" /etc/apt/sources.list 2>/dev/null || true; \
        rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        git-lfs \
        wget \
        curl \
        ca-certificates \
        ffmpeg \
        netcat-openbsd \
        libgl1 \
        libglu1-mesa \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        libegl1 \
        libvulkan1 \
        vulkan-tools \
        mesa-utils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Headless EGL / Vulkan ICD (NVIDIA) ───────────────────────────────────────
# Point the GL/Vulkan loaders at the NVIDIA driver that the NVIDIA Container
# Toolkit injects at runtime, so Isaac Sim can render cameras headless.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.242"
    }
}
EOF

WORKDIR /workspace/RoboDojo

# ── Miniconda + RoboDojo env ─────────────────────────────────────────────────
# setup_conda() from scripts/install.sh: Miniconda under $HOME/miniconda3 and a
# Python 3.11 env named `RoboDojo` (the simulator conda env used by all scripts).
RUN wget -q "${MINICONDA_URL}" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm -f /tmp/miniconda.sh && \
    conda config --set always_yes yes && \
    if [ -n "${CONDA_CHANNEL_MIRROR}" ]; then \
        conda config --add default_channels "${CONDA_CHANNEL_MIRROR}/pkgs/main" && \
        conda config --add default_channels "${CONDA_CHANNEL_MIRROR}/pkgs/r" && \
        conda config --set show_channel_urls yes && \
        conda config --set channel_alias "${CONDA_CHANNEL_MIRROR}/cloud"; \
    fi && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true) && \
    conda create -n RoboDojo python=3.11 -y && \
    conda clean -afy

# Run every subsequent build step inside the RoboDojo env.
SHELL ["conda", "run", "--no-capture-output", "-n", "RoboDojo", "/bin/bash", "-c"]

# ── Base pip dependencies ────────────────────────────────────────────────────
# setup_base_deps() from scripts/install.sh. requirements.txt is copied on its
# own so this layer caches independently of the rest of the source tree.
COPY scripts/requirements.txt /workspace/RoboDojo/scripts/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r scripts/requirements.txt && \
    python -m pip install opencv-python-headless==4.11.0.86 pillow matplotlib "scipy==1.15.3" scikit-learn && \
    python -m pip install numpy==1.26.0

# ── PyTorch (cu128) + Isaac Sim 5.1 ──────────────────────────────────────────
# setup_isaacsim() from scripts/install.sh.
RUN python -m pip install "numpy==1.26.0" "typing_extensions==4.12.2" "filelock==3.13.1" && \
    python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# ── IsaacLab + CuRobo (built from the vendored submodules) ───────────────────
# setup_isaaclab() + setup_curobo() + repin_after_curobo() from
# scripts/install.sh. The submodules are COPYed in; their `.git` submodule pointer
# files are stripped later (see below) so we never touch git at build/run time.
# Only third_party/ is copied here so a source edit does not invalidate these
# expensive build layers.
COPY third_party/ /workspace/RoboDojo/third_party/
# `--install none` installs all core IsaacLab extensions but skips the RL learning
# frameworks (rl_games/rsl_rl/skrl/sb3). RoboDojo eval does not use them, and
# rl_games is pulled from git+github (unreachable/timeouts on some networks).
RUN cd third_party/IsaacLab && ./isaaclab.sh --install none
RUN cd third_party/curobo && \
    python -m pip uninstall -y nvidia-curobo curobo 2>/dev/null || true && \
    python -m pip install -e ".[cu12]" --no-build-isolation
RUN python -m pip install \
        "numpy==1.26.0" \
        "packaging==23.0" \
        "typing_extensions==4.12.2" \
        "filelock==3.13.1" \
        "websockets==12.0" \
        "scipy==1.15.3" \
        "warp-lang==1.11.0"

# ── RoboDojo source + XPolicyLab client/adapter code ─────────────────────────
# Explicit copies (not `COPY .`) so we do not clobber the compiled artifacts
# already built under third_party/. Heavy policy source, weights, checkpoints,
# assets, and runtime outputs are excluded via .dockerignore.
COPY env/ /workspace/RoboDojo/env/
COPY env_cfg/ /workspace/RoboDojo/env_cfg/
COPY task/ /workspace/RoboDojo/task/
COPY src/ /workspace/RoboDojo/src/
COPY utils/ /workspace/RoboDojo/utils/
COPY scripts/ /workspace/RoboDojo/scripts/
COPY XPolicyLab/ /workspace/RoboDojo/XPolicyLab/
COPY pyproject.toml README.md LICENSE /workspace/RoboDojo/

# ── Strip broken submodule .git pointers ─────────────────────────────────────
# Vendored submodules are copied with their `.git` files (a submodule `.git` is a
# FILE, so `.dockerignore`'s `.git/` directory rule misses it). curobo detects its
# version via setuptools_scm whenever a `.git` exists, which raises LookupError at
# import time in-image (no real git repo behind the pointer) and crashes eval.
# Removing them makes curobo fall back to the installed package version. This is a
# late, cheap layer so the expensive IsaacLab/curobo build layers stay cached.
RUN find /workspace/RoboDojo/third_party -name .git -prune -exec rm -rf {} + 2>/dev/null; true

# ── Headless RTX render deps + single Vulkan ICD (late, cheap layers) ─────────
SHELL ["/bin/bash", "-c"]
# libXt.so.6 is required by Isaac Sim's MaterialX render libs
# (libusd_usdBakeMtlx.so, libMaterialXRender*.so). Without it they fail to load at
# startup ("libXt.so.6: cannot open shared object file"). Its deps libsm6/libice6
# are already installed above. Kept as a late layer so the expensive
# IsaacLab/curobo build layers stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends libxt6 \
    && rm -rf /var/lib/apt/lists/*
# Force a SINGLE Vulkan ICD. This image bakes /usr/share/vulkan/icd.d/nvidia_icd.json
# AND the NVIDIA Container Toolkit injects /etc/vulkan/icd.d/nvidia_icd.json at run
# time; the loader then sees two ICDs for the same GPU ("multiple installable client
# drivers") and RTX device init wedges on the first rendered frame (tiled cameras
# hang right after buffer allocation). Pinning the toolkit-injected, driver-matched
# ICD makes headless RTX rendering start reliably. --gpus with
# NVIDIA_DRIVER_CAPABILITIES=all guarantees that path exists at run time.
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# ── Entrypoint ───────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PYTHONPATH=/workspace/RoboDojo

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
