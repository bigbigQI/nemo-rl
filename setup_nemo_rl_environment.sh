#!/bin/bash

# Setup script to replicate NeMo RL Docker environment
# Based on docker/Dockerfile.ngc_pytorch
# Usage: ./setup_nemo_rl_environment.sh

set -exu -o pipefail

# Configuration variables (can be overridden)
NRL_GIT_REF=${NRL_GIT_REF:-main}
UV_VERSION=${UV_VERSION:-0.7.2}
MAX_JOBS=${MAX_JOBS:-16}
NVTE_BUILD_THREADS_PER_JOB=${NVTE_BUILD_THREADS_PER_JOB:-}
BASE_DIR=${BASE_DIR:-/opt}
NEMO_RL_DIR=${NEMO_RL_DIR:-/opt/nemo-rl}
VENV_DIR=${VENV_DIR:-/opt/nemo_rl_venv}
UV_CACHE_DIR=${UV_CACHE_DIR:-/opt/uv_cache}

echo "=== Setting up NeMo RL Environment ==="
echo "NRL_GIT_REF: $NRL_GIT_REF"
echo "UV_VERSION: $UV_VERSION"
echo "MAX_JOBS: $MAX_JOBS"
echo "BASE_DIR: $BASE_DIR"
echo "NEMO_RL_DIR: $NEMO_RL_DIR"
echo "VENV_DIR: $VENV_DIR"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script should be run as root for system package installation"
   echo "Run with: sudo $0"
   exit 1
fi

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
export RAY_USAGE_STATS_ENABLED=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_PROJECT_ENVIRONMENT=$VENV_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_LINK_MODE=copy
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export VLLM_USE_STANDALONE_COMPILE=0

# Step 1: Install system packages
echo "=== Installing system packages ==="
apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    rsync \
    wget \
    less \
    vim \
    python3-dev \
    python3-pip \
    build-essential

apt-get clean
rm -rf /var/lib/apt/lists/*

# Step 2: Install UV package manager
echo "=== Installing UV package manager ==="
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | XDG_BIN_HOME=/usr/local/bin sh

# Add UV to PATH
export PATH="/usr/local/bin:$PATH"

# Step 3: Clone NeMo RL repository
echo "=== Cloning NeMo RL repository ==="
mkdir -p $BASE_DIR
cd $BASE_DIR

if [ -d "nemo-rl" ]; then
    echo "NeMo RL directory already exists, removing..."
    rm -rf nemo-rl
fi

git clone --depth 1 --branch ${NRL_GIT_REF} https://github.com/NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl
git submodule update --init --recursive

# Step 4: Build vLLM from source
echo "=== Building vLLM from source ==="
cd $BASE_DIR

# Extract vLLM version from uv.lock
VLLM_VERSION=$(grep -A 1 'name = "vllm"' $BASE_DIR/nemo-rl/uv.lock | grep 'version =' | sed 's/version = "\(.*\)"/\1/')
echo "Building vLLM version: $VLLM_VERSION"

if [ -d "vllm" ]; then
    echo "vLLM directory already exists, removing..."
    rm -rf vllm
fi

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v$VLLM_VERSION
python3 use_existing_torch.py
pip install -r requirements/build.txt
pip wheel --no-deps --no-build-isolation -v .

# Step 5: Setup Python virtual environment
echo "=== Setting up Python virtual environment ==="
cd $NEMO_RL_DIR

# Create virtual environment with system site packages
uv venv --system-site-packages ${VENV_DIR}

# Define packages to exclude (already available in PyTorch base image)
UV_NO_INSTALL_PACKAGES="--no-install-package torch --no-install-package torchvision --no-install-package triton --no-install-package nvidia-cublas-cu12 --no-install-package nvidia-cuda-cupti-cu12 --no-install-package nvidia-cuda-nvrtc-cu12 --no-install-package nvidia-cuda-runtime-cu12 --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cufft-cu12 --no-install-package nvidia-cufile-cu12 --no-install-package nvidia-curand-cu12 --no-install-package nvidia-cusolver-cu12 --no-install-package nvidia-cusparse-cu12 --no-install-package nvidia-cusparselt-cu12 --no-install-package nvidia-nccl-cu12 --no-install-package vllm --no-install-package flash-attn --no-install-package transformer-engine --no-install-package transformer-engine-cu12 --no-install-package transformer-engine-torch --no-install-package numpy"

export UV_NO_INSTALL_PACKAGES="${UV_NO_INSTALL_PACKAGES}"

# Install built vLLM wheel
uv pip install --no-cache-dir --no-deps $BASE_DIR/vllm/vllm*.whl

# Install NeMo RL dependencies
echo "=== Installing NeMo RL dependencies ==="
uv sync --link-mode symlink --locked --inexact --extra vllm --extra mcore --extra automodel --all-groups --no-install-project $UV_NO_INSTALL_PACKAGES

# Step 6: Final setup of NeMo RL
echo "=== Final NeMo RL setup ==="
export UV_NO_SYNC=1
UV_LINK_MODE=symlink uv sync --locked --inexact $UV_NO_INSTALL_PACKAGES

# Step 7: Update PATH
echo "=== Updating PATH ==="
export PATH="${VENV_DIR}/bin:$PATH"

# Step 8: Create activation script
echo "=== Creating activation script ==="
cat > /opt/activate_nemo_rl.sh << 'EOF'
#!/bin/bash
export RAY_USAGE_STATS_ENABLED=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_CACHE_DIR=/opt/uv_cache
export UV_LINK_MODE=copy
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export VLLM_USE_STANDALONE_COMPILE=0
export UV_NO_SYNC=1
export PATH="/opt/nemo_rl_venv/bin:$PATH"

echo "NeMo RL environment activated!"
echo "Working directory: /opt/nemo-rl"
cd /opt/nemo-rl
EOF

chmod +x /opt/activate_nemo_rl.sh

echo "=== Setup Complete ==="
echo ""
echo "To activate the NeMo RL environment, run:"
echo "source /opt/activate_nemo_rl.sh"
echo ""
echo "The environment is now ready for use at: $NEMO_RL_DIR"
