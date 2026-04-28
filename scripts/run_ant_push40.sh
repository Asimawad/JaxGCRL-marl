#!/bin/bash
# CPPO on JaxGCRL Ant — best-known pure-PPO + CRL recipe, scaled up.
#
# Recipe basis: the run that produced 0.598 peak / 0.566 final at 50M
# (cppo-ant-purer-seed0, wandb run e0mj8w9z). That recipe matches JaxGCRL CRL's
# choices on the three knobs that mattered:
#
#   energy_fn=norm                 (CRL paper default — Euclidean energy)
#   contrastive_temperature=1.0    (CRL paper has no temperature)
#   log_std_max=2.0                (CRL paper default — wide squashing)
#
# Plus pure-PPO actor (no SAC bits): constant ent_coef=0.0001, no adaptive
# log_alpha, no target entropy, no entropy schedule. The natural equilibrium
# of PPO ratio + small entropy bonus settles at entropy ~10–25 (pre-tanh)
# without any controller.
#
# Compute boost vs the 50M run:
#   STEPS 50M → 100M           (curve was still climbing at 50M)
#   num_envs 512 → 1024        (bigger contrastive batch in InfoNCE)
#   num_mc_samples 32 → 64     (lower-variance V baseline)
#   num_evals 30 → 50          (finer resolution, catch peaks earlier)
#
# If you OOM, override via env: NUM_ENVS=512 NUM_MC=32 bash scripts/run_ant_push40.sh

# Reads WANDB_API_KEY from your environment. Set via:
#   export WANDB_API_KEY=...   (in your shell or .env file, never committed)
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate

DATE=$(date +%Y-%m-%d)
GROUP="cppo-ant-push40-${DATE}"
WANDB_PROJECT=${WANDB_PROJECT:-cppo-brax}

SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-100000000}
NUM_ENVS=${NUM_ENVS:-512}
NUM_MC=${NUM_MC:-64}


for SEED in $SEEDS; do
    EXP_NAME="cppo-ant-push40-seed${SEED}-${DATE}"
    echo "=== seed=$SEED  steps=$STEPS  num_envs=$NUM_ENVS  num_mc=$NUM_MC ==="
    .venv/bin/python run.py \
        --env=ant \
        --seed="$SEED" \
        --total-env-steps="$STEPS" \
        --num-envs="$NUM_ENVS" \
        --num-eval-envs=512 \
        --episode-length=501 \
        --num-evals=50 \
        --exp-name="$EXP_NAME" \
        --log-wandb \
        --wandb-project-name="$WANDB_PROJECT" \
        --wandb-group="$GROUP" \
        cppo \
        --rollout-length=128 \
        --unroll-length=128 \
        --num-epochs=2 \
        --batch-size=256 \
        --num-mc-samples="$NUM_MC" \
        --discounting=0.9999 \
        --actor-lr=3e-4 \
        --q-lr=3e-4 \
        --clip-eps=0.15 \
        --max-grad-norm=1.0 \
        --ent-coef=0.001 \
        --ent-coef-end=0.0001 \
        --no-use-adaptive-entropy \
        --contrastive-loss-fn=fwd_infonce \
        --energy-fn=norm \
        --contrastive-temperature=1.0 \
        --logsumexp-penalty-coeff=0.1 \
        --log-std-min=-5.0 \
        --log-std-max=-0.5 \
        --h-dim=512 \
        --n-hidden=6 \
        --use-layer-l2 \
        --skip-connections=4 \
        --sa-state-mode=state_only \
        --actor-input-mode=obs_full_ach \
        --use-achieved-goal \
        --terminate-on-success
done

# for SEED in $SEEDS; do
#     EXP_NAME="cppo-ant-push40-seed${SEED}-${DATE}"
#     echo "=== seed=$SEED  steps=$STEPS  num_envs=$NUM_ENVS  num_mc=$NUM_MC ==="
#     .venv/bin/python run.py \
#         --env=ant \
#         --seed="$SEED" \
#         --total-env-steps="$STEPS" \
#         --num-envs="$NUM_ENVS" \
#         --num-eval-envs=512 \
#         --episode-length=1001 \
#         --num-evals=50 \
#         --exp-name="$EXP_NAME" \
#         --log-wandb \
#         --wandb-project-name="$WANDB_PROJECT" \
#         --wandb-group="$GROUP" \
#         cppo \
#         --rollout-length=128 \
#         --unroll-length=128 \
#         --num-epochs=8 \
#         --batch-size=256 \
#         --num-mc-samples="$NUM_MC" \
#         --discounting=0.9999 \
#         --actor-lr=3e-4 \
#         --q-lr=3e-4 \
#         --clip-eps=0.15 \
#         --max-grad-norm=1.0 \
#         --ent-coef=0.0001 \
#         --ent-coef-end=0.0001 \
#         --no-use-adaptive-entropy \
#         --contrastive-loss-fn=fwd_infonce \
#         --energy-fn=norm \
#         --contrastive-temperature=1.0 \
#         --logsumexp-penalty-coeff=0.1 \
#         --log-std-min=-5.0 \
#         --log-std-max=-1.0 \
#         --h-dim=512 \
#         --n-hidden=4 \
#         --use-layer-norm \
#         --skip-connections=4 \
#         --sa-state-mode=state_only \
#         --actor-input-mode=obs_full_ach \
#         --use-achieved-goal \
#         --terminate-on-success
# done
