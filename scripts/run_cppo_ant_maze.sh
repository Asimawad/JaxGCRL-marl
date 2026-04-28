#!/bin/bash
# CPPO (PPO + CRL contrastive critic) on ant_u_maze.
# Goal: ~0.5 success_any. Config from winning Reacher CPPO sweep.
#
# Usage:
#   bash scripts/run_cppo_ant_maze.sh
#   SEED=2 STEPS=30000000 bash scripts/run_cppo_ant_maze.sh

set -euo pipefail

cd "$(dirname "$0")/.."

ENV=${ENV:-ant_u_maze}
SEED=${SEED:-1}
STEPS=${STEPS:-50000000}
NUM_ENVS=${NUM_ENVS:-256}
EPISODE_LEN=${EPISODE_LEN:-1001}
NUM_EVALS=${NUM_EVALS:-50}

# Winning Reacher config
ROLLOUT=${ROLLOUT:-128}
NUM_EPOCHS=${NUM_EPOCHS:-8}
BATCH=${BATCH:-256}
NUM_MC=${NUM_MC:-32}
GAMMA=${GAMMA:-0.9999}
ACTOR_LR=${ACTOR_LR:-3e-4}
Q_LR=${Q_LR:-3e-4}
CLIP_EPS=${CLIP_EPS:-0.15}
ENT_COEF=${ENT_COEF:-0.01}
ENT_END=${ENT_END:-0.001}
LOSS_FN=${LOSS_FN:-fwd_infonce}
ENERGY_FN=${ENERGY_FN:-dot}
TEMP=${TEMP:-2.5}
LSE_PEN=${LSE_PEN:-0.1}
GRAD_CLIP=${GRAD_CLIP:-1.0}
H_DIM=${H_DIM:-512}
N_HIDDEN=${N_HIDDEN:-4}

DATE=$(date +%Y-%m-%d)
GROUP=${GROUP:-cppo-${ENV}-${DATE}}
EXP_NAME=${EXP_NAME:-cppo-${ENV}-seed${SEED}-${DATE}}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export MUJOCO_GL=${MUJOCO_GL:-egl}

echo "=== CPPO ${ENV} ==="
echo "seed=$SEED steps=$STEPS num_envs=$NUM_ENVS"
echo "loss=$LOSS_FN energy=$ENERGY_FN temp=$TEMP"
echo "actor_lr=$ACTOR_LR q_lr=$Q_LR ent=$ENT_COEF gamma=$GAMMA"
echo "net=${H_DIM}x${N_HIDDEN} +LN"
echo "exp=$EXP_NAME"
echo "==================="

.venv/bin/python run.py \
    --env "$ENV" \
    --seed "$SEED" \
    --total-env-steps "$STEPS" \
    --num-envs "$NUM_ENVS" \
    --episode-length "$EPISODE_LEN" \
    --num-evals "$NUM_EVALS" \
    --exp-name "$EXP_NAME" \
    --no-log-wandb \
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
    --log-std-min -5.0 --log-std-max 2.0 \
    --h-dim "$H_DIM" --n-hidden "$N_HIDDEN" --use-layer-norm \
    --use-achieved-goal
