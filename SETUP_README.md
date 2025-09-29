# NeMo RL Environment Setup Scripts

These scripts replicate the Docker build environment from `docker/Dockerfile.ngc_pytorch` for local installation.

## Files

- `setup_nemo_rl_environment.sh` - Main setup script that replicates the Docker build
- `setup_nemo_rl_simple.sh` - Interactive wrapper script with user-friendly options
- `SETUP_README.md` - This documentation file

## Prerequisites

- Ubuntu/Debian-based Linux system
- Root access (sudo)
- NVIDIA PyTorch base environment (if you want exact Docker parity)
- Python 3.8+
- Git

## Quick Start

### Option 1: Interactive Setup (Recommended)

```bash
sudo ./setup_nemo_rl_simple.sh
```

This will prompt you for configuration options and guide you through the setup.

### Option 2: Direct Setup with Defaults

```bash
sudo ./setup_nemo_rl_environment.sh
```

This runs with default settings matching the Docker configuration.

### Option 3: Custom Configuration

You can override any configuration by setting environment variables:

```bash
sudo NRL_GIT_REF=r0.3.0 UV_VERSION=0.7.2 BASE_DIR=/custom/path ./setup_nemo_rl_environment.sh
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `NRL_GIT_REF` | `main` | NeMo RL git branch/tag to checkout |
| `UV_VERSION` | `0.7.2` | UV package manager version |
| `MAX_JOBS` | `16` | Maximum parallel build jobs |
| `BASE_DIR` | `/opt` | Base installation directory |
| `NEMO_RL_DIR` | `/opt/nemo-rl` | NeMo RL source directory |
| `VENV_DIR` | `/opt/nemo_rl_venv` | Python virtual environment path |
| `UV_CACHE_DIR` | `/opt/uv_cache` | UV cache directory |

## What the Script Does

1. **System Package Installation**: Installs required system dependencies (git, curl, build tools, etc.)
2. **UV Installation**: Downloads and installs the UV Python package manager
3. **NeMo RL Repository**: Clones the NeMo RL repository with submodules
4. **vLLM Build**: Builds vLLM from source matching the version in uv.lock
5. **Python Environment**: Creates virtual environment with all dependencies
6. **Environment Setup**: Configures all necessary environment variables

## After Installation

### Activate the Environment

```bash
source /opt/activate_nemo_rl.sh
```

### Verify Installation

```bash
cd /opt/nemo-rl
python -c "import nemo_rl; print('NeMo RL successfully installed!')"
```

## Key Differences from Docker

- Runs on your host system instead of in a container
- Uses your system's CUDA installation (if available)
- Preserves your existing Python packages (system-site-packages enabled)
- Creates persistent installation that survives reboots

## Troubleshooting

### Permission Issues
Make sure to run with `sudo` for system package installation.

### CUDA Issues
Ensure you have compatible NVIDIA drivers and CUDA toolkit installed.

### Build Failures
Check that you have sufficient disk space and memory for the vLLM build process.

### Dependency Conflicts
The script excludes PyTorch and NVIDIA packages assuming they're already installed. If you encounter issues, you may need to adjust the `UV_NO_INSTALL_PACKAGES` variable.

## Environment Variables Set

The activation script sets these important environment variables:

```bash
RAY_USAGE_STATS_ENABLED=0
NEMO_RL_VENV_DIR=/opt/ray_venvs
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
NEMO_RL_PY_EXECUTABLES_SYSTEM=1
VLLM_USE_STANDALONE_COMPILE=0
```

## Cleanup

To remove the installation:

```bash
sudo rm -rf /opt/nemo-rl /opt/nemo_rl_venv /opt/uv_cache /opt/vllm /opt/activate_nemo_rl.sh
```

## Support

This setup replicates the Docker environment as closely as possible. For issues specific to NeMo RL functionality, refer to the [NeMo RL documentation](https://github.com/NVIDIA-NeMo/RL).
