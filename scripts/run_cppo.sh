#!/bin/bash
# Generic CPPO runner with winning Reacher config + adaptive entropy.
# Usage:
#   ENV=ant STEPS=20000000 LOGFILE=/tmp/cppo_ant.log bash scripts/run_cppo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV=${ENV:?must set ENV}
SEED=${SEED:-1}
STEPS=${STEPS:-20000000}
NUM_ENVS=${NUM_ENVS:-256}
EPISODE_LEN=${EPISODE_LEN:-1001}
NUM_EVALS=${NUM_EVALS:-30}

ROLLOUT=${ROLLOUT:-128}
NUM_EPOCHS=${NUM_EPOCHS:-8}
BATCH=${BATCH:-256}
NUM_MC=${NUM_MC:-32}
GAMMA=${GAMMA:-0.9999}
ACTOR_LR=${ACTOR_LR:-3e-4}
Q_LR=${Q_LR:-3e-4}
CLIP_EPS=${CLIP_EPS:-0.15}
ENT_COEF=${ENT_COEF:-0.01}
ENT_END=${ENT_END:-0.01}
LOSS_FN=${LOSS_FN:-fwd_infonce}
ENERGY_FN=${ENERGY_FN:-dot}
TEMP=${TEMP:-2.5}
LSE_PEN=${LSE_PEN:-0.1}
GRAD_CLIP=${GRAD_CLIP:-1.0}
H_DIM=${H_DIM:-512}
N_HIDDEN=${N_HIDDEN:-4}
LSMAX=${LSMAX:-0.0}
LSMIN=${LSMIN:--5.0}

ADAPTIVE=${ADAPTIVE:-1}
TARGET_ENT=${TARGET_ENT:-4.0}
ALPHA_LR=${ALPHA_LR:-3e-4}
ALPHA_MIN=${ALPHA_MIN:-0.0001}
ALPHA_MAX=${ALPHA_MAX:-0.5}

LOGFILE=${LOGFILE:-/tmp/cppo_${ENV}.log}
EXP_NAME=${EXP_NAME:-cppo-${ENV}-seed${SEED}}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}
export MUJOCO_GL=${MUJOCO_GL:-egl}

echo "[CPPO] env=$ENV steps=$STEPS num_envs=$NUM_ENVS adaptive=$ADAPTIVE log=$LOGFILE"

ADAPT_FLAGS=""
if [ "$ADAPTIVE" = "1" ]; then
    ADAPT_FLAGS="--use-adaptive-entropy --target-entropy $TARGET_ENT --alpha-lr $ALPHA_LR --log-alpha-clip-min $ALPHA_MIN --log-alpha-clip-max $ALPHA_MAX"
fi

# Wandb on if WANDB_API_KEY set, off otherwise. Override with WANDB=1 / WANDB=0.
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB:-1}" = "1" ]; then
    WANDB_FLAGS="--log-wandb --wandb-project-name ${WANDB_PROJECT:-jaxgcrl} --wandb-group ${WANDB_GROUP:-cppo-${ENV}}"
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
    cppo \
    --rollout-length "$ROLLOUT" \
    --unroll-length "$ROLLOUT" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH" \
    --num-mc-samples "$NUM_MC" \
    --discounting "$GAMMA" \
    --actor-lr "$ACTOR_LR" \
    --q-lr "$Q_LR" \
    --clip-eps "$CLIP_EPS" \
    --ent-coef "$ENT_COEF" \
    --ent-coef-end "$ENT_END" \
    --contrastive-loss-fn "$LOSS_FN" \
    --energy-fn "$ENERGY_FN" \
    --contrastive-temperature "$TEMP" \
    --logsumexp-penalty-coeff "$LSE_PEN" \
    --max-grad-norm "$GRAD_CLIP" \
    --log-std-min "$LSMIN" --log-std-max "$LSMAX" \
    --h-dim "$H_DIM" --n-hidden "$N_HIDDEN" --use-layer-norm \
    --skip-connections "${SKIP_CONN:-0}" \
    --sa-state-mode "${SA_STATE_MODE:-obs_full}" \
    --actor-input-mode "${ACTOR_INPUT_MODE:-obs_full_ach}" \
    $([ "${TERMINATE_ON_SUCCESS:-0}" = "1" ] && echo --terminate-on-success || echo --no-terminate-on-success) \
    --use-achieved-goal \
    $ADAPT_FLAGS ${EXTRA_FLAGS:-} 2>&1 | tee "$LOGFILE"
