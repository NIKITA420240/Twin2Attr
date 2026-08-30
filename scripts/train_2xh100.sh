#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME before launching training}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  run.py train "$@"
