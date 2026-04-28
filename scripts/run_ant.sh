#!/bin/bash
# JaxGCRL CRL baseline — Ant
# Goal: reproduce paper Fig (Ant CRL ~0.95 success).
# Config from scripts/train.sh + RunConfig defaults (paper-faithful).
#
# Usage:
#   bash scripts/run_ant.sh                    # seed 1, 50M steps default
#   SEED=2 bash scripts/run_ant.sh
#   STEPS=10000000 bash scripts/run_ant.sh     # quick check
#   GROUP=baseline-ant-v1 bash scripts/run_ant.sh

set -euo pipefail

cd "$(dirname "$0")/.."

SEED=${SEED:-1}
STEPS=${STEPS:-50000000}      # paper convergence; ~30-40 min wall on A10
NUM_ENVS=${NUM_ENVS:-1024}     # paper sweep config (scripts/sweep.yml)
EPISODE_LEN=${EPISODE_LEN:-1001}
UNROLL=${UNROLL:-62}
BATCH=${BATCH:-256}
MIN_REPLAY=${MIN_REPLAY:-1000}
MAX_REPLAY=${MAX_REPLAY:-10000}
DISCOUNT=${DISCOUNT:-0.99}
LOSS_FN=${LOSS_FN:-bwd_infonce}
ENERGY_FN=${ENERGY_FN:-norm}
NUM_EVALS=${NUM_EVALS:-50}

DATE=$(date +%Y-%m-%d)
PROJECT=${WANDB_PROJECT:-jaxgcrl}
GROUP=${GROUP:-baseline-ant-${DATE}}
EXP_NAME=${EXP_NAME:-crl-ant-seed${SEED}-${DATE}}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export MUJOCO_GL=${MUJOCO_GL:-egl}

echo "=== JaxGCRL CRL Ant ==="
echo "seed=$SEED steps=$STEPS num_envs=$NUM_ENVS"
echo "loss=$LOSS_FN energy=$ENERGY_FN"
echo "group=$GROUP exp=$EXP_NAME"
echo "======================="

.venv/bin/python run.py \
    --env ant \
    --seed "$SEED" \
    --total-env-steps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --episode-length "$EPISODE_LEN" \
    --num-evals "$NUM_EVALS" \
    --exp-name "$EXP_NAME" \
    --no-log-wandb \
    crl \
    --batch-size "$BATCH" \
    --discounting "$DISCOUNT" \
    --unroll-length "$UNROLL" \
    --min-replay-size "$MIN_REPLAY" \
    --max-replay-size "$MAX_REPLAY" \
    --contrastive-loss-fn "$LOSS_FN" \
    --energy-fn "$ENERGY_FN" \
    --train-step-multiplier 1
