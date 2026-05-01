#!/bin/bash
set -euo pipefail

unset WANDB_API_KEY
export WANDB_MODE=disabled
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

SEEDS=${SEEDS:-"0 1 2"}

for SEED in $SEEDS; do
    echo "=== seed=$SEED ==="
    python -u run.py \
        --env ant \
        --seed "${SEED}" \
        --total-env-steps 50000000 \
        --num-envs 1024 \
        --num-eval-envs 1024 \
        --episode-length 1000 \
        --num-evals 50 \
        --exp-name "crl-ant-paper-sym-seed${SEED}" \
        --no-log-wandb \
        crl \
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
        "$@" 2>&1 | tee "/tmp/jaxgcrl_paper_seed${SEED}.log"
    echo "=== seed=$SEED done ==="
done
