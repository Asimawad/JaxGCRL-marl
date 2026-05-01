#!/bin/bash
# Ant CRL — paper-faithful config (Figure 7 of JaxGCRL paper, arXiv 2408.11052)
# Reproduces ~0.85-0.90 success rate by ~30M steps, plateau through 300M.
#
# Key parameters derived from paper Table 2 / Appendix A.6:
#   sym_infonce + l2 energy + h_dim=512 + n_hidden=4 + use_ln + policy_lr=6e-4
#
# Usage:
#   SEEDS="0 1 2" bash scripts/run_ant_paper.sh        # run 3 seeds sequentially
#   bash scripts/run_ant_paper.sh                       # default: seeds 0 1 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

unset WANDB_API_KEY
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.85   # h100 80gb: leave headroom for large-arch scan

SEEDS=${SEEDS:-"0 1 2"}
TOTAL_STEPS=${TOTAL_STEPS:-300000000}
NUM_EVALS=${NUM_EVALS:-100}

for SEED in $SEEDS; do
    echo "========================================"
    echo " CRL Ant  seed=$SEED  steps=$TOTAL_STEPS"
    echo "========================================"
    python -u run.py \
        --env ant \
        --seed "${SEED}" \
        --total-env-steps "${TOTAL_STEPS}" \
        --num-envs 512 \
        --num-eval-envs 512 \
        --episode-length 1000 \
        --num-evals "${NUM_EVALS}" \
        --exp-name "crl-ant-paper-seed${SEED}" \
        --no-log-wandb \
        crl \
        --h-dim 512 \
        --n-hidden 4 \
        --repr-dim 64 \
        --unroll-length 62 \
        --min-replay-size 1000 \
        --max-replay-size 10000 \
        --batch-size 256 \
        --discounting 0.99 \
        --policy-lr 6e-4 \
        --critic-lr 3e-4 \
        --contrastive-loss-fn sym_infonce \
        --energy-fn l2 \
        --use-ln \
        --logsumexp-penalty-coeff 0.1 \
        --train-step-multiplier 1 \
        "$@"
    echo "=== seed=$SEED done ==="
done
