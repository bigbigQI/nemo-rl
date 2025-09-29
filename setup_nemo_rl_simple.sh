#!/bin/bash

# Simplified NeMo RL Environment Setup Script
# This script provides a more user-friendly interface with customizable options

set -e

# Default values
DEFAULT_NRL_GIT_REF="main"
DEFAULT_UV_VERSION="0.7.2"
DEFAULT_MAX_JOBS="16"
DEFAULT_BASE_DIR="/opt"
DEFAULT_VENV_DIR="/opt/nemo_rl_venv"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== NeMo RL Environment Setup ===${NC}"
echo "This script will set up the NeMo RL environment based on the Docker configuration."
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root${NC}"
   echo "Please run with: sudo $0"
   exit 1
fi

# Interactive configuration
read -p "Enter NeMo RL Git Reference [${DEFAULT_NRL_GIT_REF}]: " NRL_GIT_REF
NRL_GIT_REF=${NRL_GIT_REF:-$DEFAULT_NRL_GIT_REF}

read -p "Enter UV Version [${DEFAULT_UV_VERSION}]: " UV_VERSION
UV_VERSION=${UV_VERSION:-$DEFAULT_UV_VERSION}

read -p "Enter Maximum Build Jobs [${DEFAULT_MAX_JOBS}]: " MAX_JOBS
MAX_JOBS=${MAX_JOBS:-$DEFAULT_MAX_JOBS}

read -p "Enter Base Directory [${DEFAULT_BASE_DIR}]: " BASE_DIR
BASE_DIR=${BASE_DIR:-$DEFAULT_BASE_DIR}

read -p "Enter Virtual Environment Directory [${DEFAULT_VENV_DIR}]: " VENV_DIR
VENV_DIR=${VENV_DIR:-$DEFAULT_VENV_DIR}

echo ""
echo -e "${YELLOW}Configuration Summary:${NC}"
echo "NeMo RL Git Reference: $NRL_GIT_REF"
echo "UV Version: $UV_VERSION"
echo "Max Jobs: $MAX_JOBS"
echo "Base Directory: $BASE_DIR"
echo "Virtual Environment: $VENV_DIR"
echo ""

read -p "Continue with installation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Export variables for the main script
export NRL_GIT_REF
export UV_VERSION
export MAX_JOBS
export BASE_DIR
export VENV_DIR
export NEMO_RL_DIR="$BASE_DIR/nemo-rl"
export UV_CACHE_DIR="$BASE_DIR/uv_cache"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the main setup script
echo -e "${GREEN}Starting installation...${NC}"
bash "$SCRIPT_DIR/setup_nemo_rl_environment.sh"

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo ""
echo "To use the NeMo RL environment:"
echo "1. Source the activation script: source /opt/activate_nemo_rl.sh"
echo "2. Navigate to the NeMo RL directory: cd $NEMO_RL_DIR"
echo "3. Start using NeMo RL!"
