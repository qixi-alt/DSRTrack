#!/usr/bin/env bash
set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash sh/test.sh /path/to/dsrtrack.bin [output_dir] [gpu_ids]"
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

WEIGHT_PATH="$1"
OUTPUT_DIR="${2:-$PROJECT_ROOT/outputs/evaluation}"
CUDA_VISIBLE_DEVICES="${3:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES

GPU_NUM=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
python3 main.py DSRTrack dinov2 \
  --eval \
  --mixin_config evaluation \
  --distributed_nproc_per_node "$GPU_NUM" \
  --distributed_do_spawn_workers \
  --weight_path "$WEIGHT_PATH" \
  --device cuda \
  --disable_wandb \
  --output_dir "$OUTPUT_DIR"
