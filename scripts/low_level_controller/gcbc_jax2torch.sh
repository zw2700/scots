#!/bin/bash

set -e

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="${SCRIPT_DIR}/gcbc_jax2torch.py"

RESTORE_EPOCH=1000000
RESULTS_BASE_DIR="${SCRIPT_DIR}/../../results"

ENVS=(
    "pointmaze-medium-stitch-v0" "pointmaze-large-stitch-v0" "pointmaze-giant-stitch-v0"
    "antmaze-medium-stitch-v0" "antmaze-large-stitch-v0" "antmaze-giant-stitch-v0"
    "antmaze-medium-explore-v0" "antmaze-large-explore-v0"
)

for env_name in "${ENVS[@]}"; do
    save_dir_path="${RESULTS_BASE_DIR}/GCBC/${env_name}"

    python "$PYTHON_SCRIPT" \
        --env_name "$env_name" \
        --restore_epoch "$RESTORE_EPOCH" \
        --save_dir "$save_dir_path" \

    echo "Finished run for Env=$env_name."
    echo
done
