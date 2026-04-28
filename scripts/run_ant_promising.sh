#!/bin/bash
# CPPO on JaxGCRL Ant — params from our cppo_ant_D recipe (peak 0.25).
# Same code path we ran yesterday (root run.py + jaxgcrl.agents.cppo).

# Reads WANDB_API_KEY from your environment. Set via:
#   export WANDB_API_KEY=...   (in your shell or .env file, never committed)
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate

DATE=$(date +%Y-%m-%d)
GROUP="cppo-ant-promising-${DATE}"
WANDB_PROJECT=${WANDB_PROJECT:-cppo-brax}

# SEEDS=${SEEDS:-"0 1 2 3 4"}
# STEPS=${STEPS:-50000000}

SEEDS=0 STEPS=100000000 \
.venv/bin/python run.py \
    --env=ant --seed=0 --total-env-steps=50000000 \
    --num-envs=512 --num-eval-envs=512 \
    --episode-length=1001 --num-evals=30 \
    --exp-name=cppo-ant-purer-seed0 \
    --log-wandb --wandb-project-name=cppo-brax \
    --wandb-group=cppo-ant-purer-2026-04-28 \
    cppo \
    --rollout-length=128 --unroll-length=128 \
    --num-epochs=8 --batch-size=256 --num-mc-samples=64 \
    --discounting=0.9999 --actor-lr=3e-4 --q-lr=3e-4 \
    --clip-eps=0.15 --max-grad-norm=1.0 \
    --ent-coef=0.0001 --ent-coef-end=0.0001 \
    --no-use-adaptive-entropy \
    --contrastive-loss-fn=fwd_infonce --energy-fn=norm \
    --contrastive-temperature=1 --logsumexp-penalty-coeff=0.1 \
    --log-std-min=-5.0 --log-std-max=2 \
    --h-dim=512 --n-hidden=4 --use-layer-norm \
    --skip-connections=4 --sa-state-mode=state_only \
    --actor-input-mode=obs_full_ach \
    --use-achieved-goal --terminate-on-success  --lr_linear_decay=True

# for SEED in $SEEDS; do
#     EXP_NAME="cppo-ant-promising-seed${SEED}-${DATE}"
#     echo "=== seed=$SEED ==="
#     .venv/bin/python run.py \
#         --env=ant \
#         --seed="$SEED" \
#         --total-env-steps="$STEPS" \
#         --num-envs=512 \
#         --num-eval-envs=512 \
#         --episode-length=1001 \
#         --num-evals=30 \
#         --exp-name="$EXP_NAME" \
#         --log-wandb \
#         --wandb-project-name="$WANDB_PROJECT" \
#         --wandb-group="$GROUP" \
#         cppo \
#         --rollout-length=128 \
#         --unroll-length=128 \
#         --num-epochs=8 \
#         --batch-size=256 \
#         --num-mc-samples=32 \
#         --discounting=0.9999 \
#         --actor-lr=3e-4 \
#         --q-lr=3e-4 \
#         --clip-eps=0.15 \
#         --max-grad-norm=1.0 \
#         --ent-coef=0.01 \
#         --ent-coef-end=0.001 \
#         --use-adaptive-entropy \
#         --target-entropy=4.0 \
#         --alpha-lr=3e-4 \
#         --log-alpha-clip-min=0.0001 \
#         --log-alpha-clip-max=0.1 \
#         --contrastive-loss-fn=fwd_infonce \
#         --energy-fn=dot \
#         --contrastive-temperature=2.5 \
#         --logsumexp-penalty-coeff=0.1 \
#         --log-std-min=-5.0 \
#         --log-std-max=-0.5 \
#         --h-dim=512 \
#         --n-hidden=4 \
#         --use-layer-norm \
#         --skip-connections=4 \
#         --sa-state-mode=state_only \
#         --actor-input-mode=obs_full_ach \
#         --use-achieved-goal
#         # --terminate-on-success
# done

# /home/app/jaxgcrl/run.py --env=ant --seed=0 --total-env-steps=50000000 --num-envs=512 --num-eval-envs=512 --episode-length=1001 --num-evals=30 --exp-name=cppo-ant-purer-seed0 --log-wandb --wandb-project-name=cppo-brax --wandb-group=cppo-ant-purer-2026-04-28 cppo --rollout-length=128 --unroll-length=128 --num-epochs=8 --batch-size=256 --num-mc-samples=32 --discounting=0.9999 --actor-lr=3e-4 --q-lr=3e-4 --clip-eps=0.15 --max-grad-norm=1.0 --ent-coef=0.0001 --ent-coef-end=0.0001 --no-use-adaptive-entropy --contrastive-loss-fn=fwd_infonce --energy-fn=norm --contrastive-temperature=1 --logsumexp-penalty-coeff=0.1 --log-std-min=-5.0 --log-std-max=2 --h-dim=512 --n-hidden=4 --use-layer-norm --skip-connections=4 --sa-state-mode=state_only --actor-input-mode=obs_full_ach --use-achieved-goal --terminate-on-success