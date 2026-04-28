#!/bin/bash
# Simple experiment launcher.
#
# Usage:
#   bash scripts/launch.sh <method> <env> [steps] [extra env vars...]
#
# Examples:
#   bash scripts/launch.sh cppo ant_u_maze
#   bash scripts/launch.sh crl ant 50000000
#   STEPS=20000000 bash scripts/launch.sh cppo humanoid
#   SEED=42 LSMAX=0 bash scripts/launch.sh cppo reacher
#
# Smart defaults:
#   - mem fraction auto-computed from current GPU usage
#   - tmux session named <method>_<env>[_<seed>]
#   - log written to logs/overnight/<method>_<env>[_<seed>].log
#
# Once started:
#   tmux attach -t <method>_<env>     # attach
#   tmux ls                           # list
#   tail -f logs/overnight/<method>_<env>.log | grep success_any

set -euo pipefail
cd "$(dirname "$0")/.."

METHOD=${1:?usage: launch.sh <method> <env> [steps]}
ENV=${2:?usage: launch.sh <method> <env> [steps]}
STEPS=${STEPS:-${3:-15000000}}
SEED=${SEED:-1}

if [ "$METHOD" != "cppo" ] && [ "$METHOD" != "crl" ]; then
    echo "method must be cppo or crl, got: $METHOD"
    exit 1
fi

# Auto memory fraction: divide remaining headroom equally + slack.
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
FREE_MB=$((TOTAL_MB - USED_MB))
# Reserve ~1GB safety. Take 80% of free.
ALLOC_MB=$(( (FREE_MB - 1024) * 8 / 10 ))
if [ "$ALLOC_MB" -lt 1500 ]; then
    echo "Not enough free GPU memory (have ${FREE_MB}MB, need >=2.5GB)."
    exit 1
fi
MEM_FRAC=$(awk "BEGIN { printf \"%.2f\", $ALLOC_MB / $TOTAL_MB }")

# Recipe defaults (best CPPO so far)
if [ "$METHOD" = "cppo" ]; then
    LSMAX=${LSMAX:--0.5}
    LOSS_FN=${LOSS_FN:-fwd_infonce}
    ENERGY_FN=${ENERGY_FN:-dot}
    TEMP=${TEMP:-2.5}
    H_DIM=${H_DIM:-512}
    N_HIDDEN=${N_HIDDEN:-4}
    SKIP_CONN=${SKIP_CONN:-4}
    SA_STATE_MODE=${SA_STATE_MODE:-state_only}
    ACTOR_INPUT_MODE=${ACTOR_INPUT_MODE:-obs_full_ach}
    NUM_ENVS=${NUM_ENVS:-256}
    ACTOR_LR=${ACTOR_LR:-3e-4}
    Q_LR=${Q_LR:-3e-4}
    ENT_COEF=${ENT_COEF:-0.01}
    ENT_END=${ENT_END:-0.001}
    ADAPTIVE=${ADAPTIVE:-1}
    TARGET_ENT=${TARGET_ENT:-4.0}
    SCRIPT=scripts/run_cppo.sh
else
    NUM_ENVS=${NUM_ENVS:-512}
    LOSS_FN=${LOSS_FN:-bwd_infonce}
    ENERGY_FN=${ENERGY_FN:-norm}
    SCRIPT=scripts/run_crl.sh
fi

# Naming
DATE=$(date +%Y-%m-%d)
TAG="${METHOD}_${ENV}_seed${SEED}"
SESS="${TAG}"
LOG="logs/overnight/${TAG}.log"
EXP_NAME="${METHOD}-${ENV}-seed${SEED}-${DATE}"

mkdir -p logs/overnight

if tmux has-session -t "$SESS" 2>/dev/null; then
    echo "tmux session $SESS already exists. attach with: tmux attach -t $SESS"
    echo "or kill: tmux kill-session -t $SESS"
    exit 1
fi

# Build env-var prefix passed to the launcher script
ENV_PREFIX="ENV=$ENV STEPS=$STEPS SEED=$SEED LOGFILE=$LOG EXP_NAME=$EXP_NAME XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRAC NUM_ENVS=$NUM_ENVS"

if [ "$METHOD" = "cppo" ]; then
    ENV_PREFIX="$ENV_PREFIX LSMAX=$LSMAX LOSS_FN=$LOSS_FN ENERGY_FN=$ENERGY_FN TEMP=$TEMP H_DIM=$H_DIM N_HIDDEN=$N_HIDDEN SKIP_CONN=$SKIP_CONN SA_STATE_MODE=$SA_STATE_MODE ACTOR_INPUT_MODE=$ACTOR_INPUT_MODE ACTOR_LR=$ACTOR_LR Q_LR=$Q_LR ENT_COEF=$ENT_COEF ENT_END=$ENT_END ADAPTIVE=$ADAPTIVE TARGET_ENT=$TARGET_ENT"
else
    ENV_PREFIX="$ENV_PREFIX LOSS_FN=$LOSS_FN ENERGY_FN=$ENERGY_FN"
fi

echo "=========================================="
echo "method      : $METHOD"
echo "env         : $ENV"
echo "steps       : $STEPS"
echo "seed        : $SEED"
echo "mem frac    : $MEM_FRAC ($ALLOC_MB MB of $TOTAL_MB MB)"
echo "tmux session: $SESS"
echo "log         : $LOG"
echo "=========================================="
echo "  attach: tmux attach -t $SESS"
echo "  watch : tail -f $LOG | grep success_any"
echo "=========================================="

tmux new-session -d -s "$SESS" "$ENV_PREFIX bash $SCRIPT"
echo "launched."
