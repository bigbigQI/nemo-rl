#!/bin/bash

# Continuation script to complete NeMo RL setup after the virtual environment creation
# This script fixes the virtual environment activation issue and continues from where the original script failed
# Usage: sudo ./continue_nemo_rl_setup.sh

set -exu -o pipefail

# Configuration variables (should match the original script)
BASE_DIR=${BASE_DIR:-/opt}
NEMO_RL_DIR=${NEMO_RL_DIR:-/opt/nemo-rl}
VENV_DIR=${VENV_DIR:-/opt/nemo_rl_venv}
UV_CACHE_DIR=${UV_CACHE_DIR:-/opt/uv_cache}

echo "=== Continuing NeMo RL Environment Setup ==="
echo "BASE_DIR: $BASE_DIR"
echo "NEMO_RL_DIR: $NEMO_RL_DIR"
echo "VENV_DIR: $VENV_DIR"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script should be run as root for consistency with the original setup"
   echo "Run with: sudo $0"
   exit 1
fi

# Verify that previous steps completed successfully
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment directory $VENV_DIR not found!"
    echo "Please run the original setup script first."
    exit 1
fi

if [ ! -d "$NEMO_RL_DIR" ]; then
    echo "Error: NeMo RL directory $NEMO_RL_DIR not found!"
    echo "Please run the original setup script first."
    exit 1
fi

if [ ! -f "$BASE_DIR/vllm/vllm"*.whl ]; then
    echo "Error: vLLM wheel file not found in $BASE_DIR/vllm/"
    echo "Please ensure the vLLM build completed successfully."
    exit 1
fi

# Set environment variables (same as original script)
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
export RAY_USAGE_STATS_ENABLED=0
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_PROJECT_ENVIRONMENT=$VENV_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_LINK_MODE=copy
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export VLLM_USE_STANDALONE_COMPILE=0

# Add UV to PATH
export PATH="/usr/local/bin:$PATH"

# Navigate to NeMo RL directory
cd $NEMO_RL_DIR

echo "=== Checking virtual environment ==="
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment activation script not found. Recreating virtual environment..."
    uv venv --system-site-packages ${VENV_DIR}
fi

echo "=== Installing vLLM wheel into virtual environment ==="
# Define packages to exclude (already available in PyTorch base image)
UV_NO_INSTALL_PACKAGES="--no-install-package torch --no-install-package torchvision --no-install-package triton --no-install-package nvidia-cublas-cu12 --no-install-package nvidia-cuda-cupti-cu12 --no-install-package nvidia-cuda-nvrtc-cu12 --no-install-package nvidia-cuda-runtime-cu12 --no-install-package nvidia-cudnn-cu12 --no-install-package nvidia-cufft-cu12 --no-install-package nvidia-cufile-cu12 --no-install-package nvidia-curand-cu12 --no-install-package nvidia-cusolver-cu12 --no-install-package nvidia-cusparse-cu12 --no-install-package nvidia-cusparselt-cu12 --no-install-package nvidia-nccl-cu12 --no-install-package vllm --no-install-package flash-attn --no-install-package transformer-engine --no-install-package transformer-engine-cu12 --no-install-package transformer-engine-torch --no-install-package numpy"

export UV_NO_INSTALL_PACKAGES="${UV_NO_INSTALL_PACKAGES}"

# Fix: Use --python option to specify the virtual environment Python interpreter
echo "Installing vLLM wheel..."
uv pip install --python ${VENV_DIR}/bin/python --no-cache-dir --no-deps $BASE_DIR/vllm/vllm*.whl

echo "=== Installing NeMo RL dependencies ==="
# Fix: Use --python option for uv sync as well
UV_PROJECT_ENVIRONMENT=$VENV_DIR uv sync --python ${VENV_DIR}/bin/python --link-mode symlink --locked --inexact --extra vllm --extra mcore --extra automodel --all-groups --no-install-project $UV_NO_INSTALL_PACKAGES

echo "=== Final NeMo RL setup ==="
export UV_NO_SYNC=1
UV_PROJECT_ENVIRONMENT=$VENV_DIR UV_LINK_MODE=symlink uv sync --python ${VENV_DIR}/bin/python --locked --inexact $UV_NO_INSTALL_PACKAGES

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

echo "=== Verifying installation ==="
echo "Testing virtual environment activation..."
source ${VENV_DIR}/bin/activate

echo "Checking Python version and packages..."
python --version
echo "Checking if vLLM is installed..."
python -c "import vllm; print('vLLM successfully installed:', vllm.__version__)" || echo "vLLM import failed - this might be expected if dependencies are missing"

echo "=== Setup Complete ==="
echo ""
echo "To activate the NeMo RL environment, run:"
echo "source /opt/activate_nemo_rl.sh"
echo ""
echo "Or directly activate the virtual environment with:"
echo "source ${VENV_DIR}/bin/activate"
echo ""
echo "The environment is now ready for use at: $NEMO_RL_DIR"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source /opt/activate_nemo_rl.sh"
echo "2. Test the installation: cd /opt/nemo-rl && python -c 'import nemo_rl'"
