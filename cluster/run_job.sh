#!/bin/bash
# run_job.sh
# Wrapper script for running a single discretax training job with HTCondor.

set -e

# shellcheck disable=SC1091
source /etc/profile.d/modules.sh

module load cuda/12.9
unset LD_LIBRARY_PATH  # Unset to avoid JAX CUDA device-not-found errors
export HOME=/home/jboyer

SWEEP_NAME=$1
PROCESS_ID=$2

SWEEP_DIR="sweeps/${SWEEP_NAME}"
CONFIG_FILE="${SWEEP_DIR}/configs/config_${PROCESS_ID}.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: configuration file $CONFIG_FILE not found"
    exit 1
fi

TRAIN_ARGS=$(cat "$CONFIG_FILE")

echo "=========================================="
echo "HTCondor Job Information"
echo "=========================================="
echo "Sweep:      $SWEEP_NAME"
echo "Process ID: $PROCESS_ID"
echo "Hostname:   $(hostname)"
echo "Working Dir: $(pwd)"
echo "Date:       $(date)"
echo "=========================================="
echo "Training args:"
echo "$TRAIN_ARGS"
echo "=========================================="

# shellcheck disable=SC2086
uv run python -m discretax.training $TRAIN_ARGS

echo "=========================================="
echo "Job completed successfully"
echo "=========================================="
