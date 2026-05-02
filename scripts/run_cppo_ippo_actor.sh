#!/usr/bin/env bash
# CPPO IPPO-style actor — actor does 4 PPO epochs over 2 huge minibatches
# (matching standard IPPO update structure). Critic still uses small batches
# because of the InfoNCE [B, B] matrix.
#
# WARNING: at the user's tuned baseline hparams (Run-9), this MAY push the
# actor too aggressively per update — entropy can collapse early and the PPO
# clip starts biting. If learning breaks, retune ent_coef (try 0.05-0.1) and
# possibly drop actor_lr.
#
# Per env step: ~8 actor SGD iters (vs 640+ in baseline). Speed gain mostly
# eaten by per-iter wall-clock cost; not as fast as critic-subsampling
# variants but algorithmically cleaner.

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="cppo_ippo_actor-smacv2_5_units"

# At num_envs=512, rollout_length=128, num_agents=5: N = 327680 transitions.
# actor_batch_size=N/2 -> 2 minibatches per actor epoch.
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
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.group=${WANDB_GROUP}"

python -u -m mava.systems.icrl.anakin.ff_ppo_crl -m system.seed=0,1,2,3,4 \
    ${COMMON_ARGS} \
    logger.loggers.wandb.run_name=cppo_ippo_actor-smacv2-5units-V1 \
    logger.loggers.wandb.tags=["cppo_ippo_actor","ff_ppo_crl","smacv2-5units","ablation"]
