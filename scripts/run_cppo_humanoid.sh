#!/bin/bash
# CPPO on JaxGCRL Humanoid — same recipe that hit 0.623 peak on Ant.
# Paper CRL @ 50M: ~0.50  (paper notes humanoid benefits from larger arch + more compute).
# 17-dim action space — hardest env. May plateau low without bigger nets / more steps.

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"
source .venv/bin/activate

DATE=$(date +%Y-%m-%d)
GROUP="cppo-humanoid-${DATE}"
WANDB_PROJECT=${WANDB_PROJECT:-cppo-brax}

SEEDS=${SEEDS:-"0"}
STEPS=${STEPS:-100000000}
NUM_ENVS=${NUM_ENVS:-512}
NUM_MC=${NUM_MC:-32}

for SEED in $SEEDS; do
    EXP_NAME="cppo-humanoid-seed${SEED}-${DATE}"
    echo "=== seed=$SEED  steps=$STEPS  num_envs=$NUM_ENVS ==="
    .venv/bin/python run.py \
        --env=humanoid \
        --seed="$SEED" \
        --total-env-steps="$STEPS" \
        --num-envs="$NUM_ENVS" \
        --num-eval-envs=512 \
        --episode-length=1001 \
        --num-evals=50 \
        --exp-name="$EXP_NAME" \
        --log-wandb \
        --wandb-project-name="$WANDB_PROJECT" \
        --wandb-group="$GROUP" \
        cppo \
        --rollout-length=128 \
        --unroll-length=128 \
        --num-epochs=8 \
        --batch-size=256 \
        --num-mc-samples="$NUM_MC" \
        --discounting=0.9999 \
        --actor-lr=3e-4 \
        --q-lr=3e-4 \
        --clip-eps=0.15 \
        --max-grad-norm=1.0 \
        --ent-coef=0.0001 \
        --ent-coef-end=0.0001 \
        --no-use-adaptive-entropy \
        --contrastive-loss-fn=fwd_infonce \
        --energy-fn=norm \
        --contrastive-temperature=1.0 \
        --logsumexp-penalty-coeff=0.1 \
        --log-std-min=-5.0 \
        --log-std-max=2.0 \
        --h-dim=512 \
        --n-hidden=4 \
        --use-layer-norm \
        --skip-connections=4 \
        --sa-state-mode=state_only \
        --actor-input-mode=obs_full_ach \
        --use-achieved-goal \
        --terminate-on-success
done
