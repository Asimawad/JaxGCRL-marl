#!/usr/bin/env bash
# CPPO SUB50 — IPPO-style actor + critic_subsample_fraction=0.5.
# Trains the critic on a random 50% subset of on-policy data per epoch.
# +45.8% SPS over baseline at SHORT runs.
#
# !!! WARNING !!!
# At the user's tuned baseline hparams (Run-9), this combo is at the edge of
# what those hparams handle. The 80M version of the closely-related sub25
# variant collapsed (entropy -> 0, win rate stuck at ~1%) because the actor
# moves too aggressively per update with the IPPO-style schedule.
#
# Before running this for real, RETUNE these:
#   - ent_coef:   0.01 -> 0.05 or 0.1   (more exploration pressure)
#   - actor_lr:   0.0005 -> 0.0001-0.0003 (smaller PPO steps)
#   - clip_eps:   maybe drop 0.2 -> 0.1 (tighter trust region)
# Verify on a short run (system.num_updates=200) that entropy doesn't crash
# below 0.3 in the first ~20 evals before committing to a full 80M sweep.

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="cppo_sub50-smacv2_5_units"

ACTOR_BATCH=163840

COMMON_ARGS="env=smax \
    env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 \
    system.num_updates=1250 \
    arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 \
    system.rollout_length=128 \
    system.gamma=0.99 \
    system.batch_size=256 \
    system.rep_size=64 \
    system.add_agent_id=1 \
    system.energy_fn=norm \
    system.contrastive_loss_fn=fwd_infonce \
    system.win_repeat_steps=5 \
    system.deterministic_reset=false \
    system.use_icrl=true \
    system.actor_lr=0.0005 \
    system.q_lr=0.0001 \
    system.ent_coef=0.01 \
    system.max_grad_norm=0.05 \
    system.clip_eps=0.2 \
    system.logsumexp_penalty_coeff=0.91 \
    system.lr_end=1e-07 \
    system.lr_decay_type=linear \
    system.ppo_epochs=2 \
    +system.actor_batch_size=${ACTOR_BATCH} \
    +system.actor_ppo_epochs=4 \
    +system.critic_ppo_epochs=1 \
    +system.critic_subsample_fraction=0.5 \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.group=${WANDB_GROUP}"

python -u -m mava.systems.icrl.anakin.ff_ppo_crl -m system.seed=0,1,2,3,4 \
    ${COMMON_ARGS} \
    logger.loggers.wandb.run_name=cppo_sub50-smacv2-5units-V1 \
    logger.loggers.wandb.tags=["cppo_sub50","ff_ppo_crl","smacv2-5units","ablation","needs-retune"]
