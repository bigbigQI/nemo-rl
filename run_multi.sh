#!/bin/bash

# Parse command line arguments
DEPENDENT=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dependent)
            DEPENDENT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-d|--dependent]"
            echo "  -d, --dependent    Run as dependent job (requires JOBID environment variable)"
            exit 1
            ;;
    esac
done

# Check if JOBID is set when dependent mode is requested
if [ "$DEPENDENT" = true ]; then
    if [ -z "$JOBID" ]; then
        echo "Error: JOBID environment variable must be set when using --dependent option"
        echo "Usage: export JOBID=xxxxx && $0 -d"
        exit 1
    fi
    echo "Running as dependent job with JOBID: $JOBID"
fi

PROJECT="Nemo-RL-qwen3-8b-base"
RUN="nemo-fp8-qwen3-8b-base"
CKPTDIR="results/${PROJECT}_${RUN}"

set -x
NUM_ACTOR_NODES=4
export HYDRA_FULL_ERROR=1
#export TRANSFORMERS_OFFLINE=1
export NCCL_AVOID_RECORD_STREAMS=1
export ENROOT_ROOTFS_WRITABLE=1
export PYTHONUNBUFFERED=1

export WANDB_API_KEY=149737fd3c4537b349a37aab90b6fff96f385ebc
export HF_HOME=/lustre/fs1/portfolios/coreai/users/larkz/.cache
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/users/larkz/.uv-cache

export RAY_LOG_SYNC_FREQUENCY=10

SBATCH_OPTIONS="--nodes=${NUM_ACTOR_NODES} --account=coreai_dlalgo_nemorl --job-name=${RUN} --partition=batch --time=3:59:00 --gres=gpu:8"
# Build sbatch command with optional dependency
if [[ "$DEPENDENT" == true ]]; then
   SBATCH_OPTIONS="$SBATCH_OPTIONS --dependency=afterany:$JOBID"
fi

cd /lustre/fs1/portfolios/coreai/users/larkz/RL

# CONTAINER="/lustre/fsw/portfolios/coreai/users/larkz/docker_images/nemorl.sqsh" \

COMMAND="cd /lustre/fs1/portfolios/coreai/users/larkz/RL && uv run ./examples/run_grpo_math.py --config examples/configs/recipes/llm/grpo-qwen3-base-8b-4n8g-megatron-fp8-dapo.yaml" \
CONTAINER="/lustre/fsw/portfolios/coreai/users/shuangy/images/nemo-rl-main-250825.sqsh" \
MOUNTS="/lustre/fs1/portfolios/coreai/users/larkz/:/lustre/fs1/portfolios/coreai/users/larkz/" \
sbatch $SBATCH_OPTIONS ray.sub

set +x
 
