#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-booml}"
  conda activate "${CONDA_ENV_NAME}"
fi

# CUDA_VISIBLE_DEVICES=0 TASK=walker-run SEED=1 ./run_train.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TASK="${TASK:-dog-run}"
SEED="${SEED:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-worldmodel}"
RES_COEF="${RES_COEF:-100}"
STEPS="${STEPS:-1000000}"
USE_SCORE_FUNCTION="${USE_SCORE_FUNCTION:-true}"
TRAIN_SCORE_FUNCTION="${TRAIN_SCORE_FUNCTION:-true}"
MPC="${MPC:-true}"
DISABLE_WANDB="${DISABLE_WANDB:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGV=()
if [ -n "${EXTRA_ARGS}" ]; then
  # EXTRA_ARGS="horizon=5 num_samples=256" ./run_train.sh
  read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS}"
fi

export CUDA_VISIBLE_DEVICES

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

echo "Launching training with:"
echo "  task=${TASK}"
echo "  seed=${SEED}"
echo "  steps=${STEPS}"
echo "  res_coef=${RES_COEF}"
echo "  use_score_function=${USE_SCORE_FUNCTION}"
echo "  train_score_function=${TRAIN_SCORE_FUNCTION}"
echo "  mpc=${MPC}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

python -m booml.train \
  task="${TASK}" \
  seed="${SEED}" \
  steps="${STEPS}" \
  res_coef="${RES_COEF}" \
  use_score_function="${USE_SCORE_FUNCTION}" \
  train_score_function="${TRAIN_SCORE_FUNCTION}" \
  mpc="${MPC}" \
  wandb_project="${WANDB_PROJECT}" \
  disable_wandb="${DISABLE_WANDB}" \
  "${EXTRA_ARGV[@]}"
