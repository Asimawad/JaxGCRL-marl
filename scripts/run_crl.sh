#!/bin/bash
# Generic JaxGCRL CRL baseline runner.
# Usage:
#   ENV=ant STEPS=20000000 LOGFILE=/tmp/crl_ant.log bash scripts/run_crl.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV=${ENV:?must set ENV}
SEED=${SEED:-1}
STEPS=${STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-512}
EPISODE_LEN=${EPISODE_LEN:-1001}
NUM_EVALS=${NUM_EVALS:-30}
LOSS_FN=${LOSS_FN:-bwd_infonce}
ENERGY_FN=${ENERGY_FN:-norm}
DISCOUNT=${DISCOUNT:-0.99}
UNROLL=${UNROLL:-62}
BATCH=${BATCH:-256}
MIN_REPLAY=${MIN_REPLAY:-1000}
MAX_REPLAY=${MAX_REPLAY:-10000}
LOGFILE=${LOGFILE:-/tmp/crl_${ENV}.log}
EXP_NAME=${EXP_NAME:-crl-${ENV}-seed${SEED}}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}
export MUJOCO_GL=${MUJOCO_GL:-egl}

echo "[CRL] env=$ENV steps=$STEPS num_envs=$NUM_ENVS seed=$SEED log=$LOGFILE"

if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--log-wandb --wandb-project-name ${WANDB_PROJECT:-jaxgcrl} --wandb-group ${WANDB_GROUP:-crl-${ENV}}"
else
    WANDB_FLAGS="--no-log-wandb"
fi

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
[ -x "$PYTHON_BIN" ] || PYTHON_BIN=python

$PYTHON_BIN run.py \
    --env "$ENV" \
    --seed "$SEED" \
    --total-env-steps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --episode-length "$EPISODE_LEN" \
    --num-evals "$NUM_EVALS" \
    --exp-name "$EXP_NAME" \
    $WANDB_FLAGS \
    crl \
    --batch-size "$BATCH" \
    --discounting "$DISCOUNT" \
    --unroll-length "$UNROLL" \
    --min-replay-size "$MIN_REPLAY" \
    --max-replay-size "$MAX_REPLAY" \
    --contrastive-loss-fn "$LOSS_FN" \
    --energy-fn "$ENERGY_FN" \
    --train-step-multiplier 1 2>&1 | tee "$LOGFILE"
