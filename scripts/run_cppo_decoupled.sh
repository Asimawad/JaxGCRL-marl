#!/usr/bin/env bash
# CPPO DECOUPLED — recommended optimized config (no GAE).
# +actor_batch_size=512 decouples actor from critic minibatch size.
# Confirmed at 80M / 512 envs: 76.51% peak win rate, 26.7 min wall-clock.
# Matches the user's tuned baseline (76.07%) at +33% throughput.

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="cppo_decoupled-smacv2_5_units"

COMMON_ARGS="env=smax \
    env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 \
    system.num_updates=1250 \
    arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 \
    system.rollout_length=128 \
    system.ppo_epochs=2 \
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
    +system.actor_batch_size=512 \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.group=${WANDB_GROUP}"

python -u -m mava.systems.icrl.anakin.ff_ppo_crl -m system.seed=0,1,2,3,4 \
    ${COMMON_ARGS} \
    logger.loggers.wandb.run_name=cppo_decoupled-smacv2-5units-V1 \
    logger.loggers.wandb.tags=["cppo_decoupled","ff_ppo_crl","smacv2-5units","optimized"]
