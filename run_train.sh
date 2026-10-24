#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-booml}"
  conda activate "${CONDA_ENV_NAME}"
fi

# 直接在这里改默认值，也可以在命令行前面临时覆盖：
# CUDA_VISIBLE_DEVICES=0 TASK=walker-run SEED=1 ./run_train.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TASK="${TASK:-dog-run}"
SEED="${SEED:-2}"
EXP_NAME="${EXP_NAME:-booml}"
WANDB_PROJECT="${WANDB_PROJECT:-worldmodel}"
RES_COEF="${RES_COEF:-100}"
STEPS="${STEPS:-1000000}"
USE_SCORE_FUNCTION="${USE_SCORE_FUNCTION:-true}"
TRAIN_SCORE_FUNCTION="${TRAIN_SCORE_FUNCTION:-true}"
MPC="${MPC:-true}"
ADAPTIVE_LARO_TRIGGER="${ADAPTIVE_LARO_TRIGGER:-variance}"
DISABLE_WANDB="${DISABLE_WANDB:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ARGV=()
if [ -n "${EXTRA_ARGS}" ]; then
  # 额外 hydra 覆盖项，按空格拆分即可，例如：
  # EXTRA_ARGS="horizon=5 num_samples=256" ./run_train.sh
  read -r -a EXTRA_ARGV <<< "${EXTRA_ARGS}"
fi

export CUDA_VISIBLE_DEVICES

# 集群上无本地代理时取消代理，避免 wandb 连本地代理失败
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

echo "Launching training with:"
echo "  task=${TASK}"
echo "  seed=${SEED}"
echo "  steps=${STEPS}"
echo "  exp_name=${EXP_NAME}"
echo "  res_coef=${RES_COEF}"
echo "  use_score_function=${USE_SCORE_FUNCTION}"
echo "  train_score_function=${TRAIN_SCORE_FUNCTION}"
echo "  mpc=${MPC}"
echo "  adaptive_laro_trigger=${ADAPTIVE_LARO_TRIGGER}"
echo "  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

python -m booml.train \
  task="${TASK}" \
  seed="${SEED}" \
  steps="${STEPS}" \
  exp_name="${EXP_NAME}" \
  res_coef="${RES_COEF}" \
  use_score_function="${USE_SCORE_FUNCTION}" \
  train_score_function="${TRAIN_SCORE_FUNCTION}" \
  mpc="${MPC}" \
  adaptive_laro_trigger="${ADAPTIVE_LARO_TRIGGER}" \
  wandb_project="${WANDB_PROJECT}" \
  disable_wandb="${DISABLE_WANDB}" \
  "${EXTRA_ARGV[@]}"
