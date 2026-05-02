#!/usr/bin/env bash
# CPPO SUB25 — IPPO-style actor + critic_subsample_fraction=0.25.
# Trains the critic on a random 25% subset of on-policy data per epoch.
# +88.7% SPS over baseline at SHORT runs.
#
# !!! WARNING !!!
# At the user's tuned baseline hparams (Run-9), this CONFIRMED FAILS at 80M:
#   - Wall-clock: 9.1 min (3x faster than non-GAE decoupled)
#   - Peak win rate: 0.98%   <-- BROKEN, never escapes random performance
#   - Entropy collapsed to 0.04 by mid-run (policy went deterministic)
#
# The critic at frac=0.25 sees only 25% of fresh on-policy data per update.
# Combined with IPPO-style actor (4 epochs over huge batches) at the
# baseline's actor_lr=0.0005 and ent_coef=0.01, the policy tunnels into a
# bad deterministic policy in the first few updates and PPO clip prevents
# recovery.
#
# To make this viable, RETUNE these (more aggressively than for sub50):
#   - ent_coef:   0.01 -> 0.1 or 0.2     (much more exploration)
#   - actor_lr:   0.0005 -> 0.00005-0.0001 (tiny PPO steps)
#   - clip_eps:   0.2 -> 0.05            (very tight trust region)
#   - actor_ppo_epochs: try dropping 4 -> 2 to slow actor evolution
# Verify on a short run that entropy stays > 0.5 for the first ~20 evals
# before committing to anything longer.
#
# Or: keep critic_subsample_fraction=0.25 but drop actor_ppo_epochs to 1
# and actor_batch_size to a smaller value (e.g. 1024). That alone may
# rescue learning at the cost of less wall-clock speedup.

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="cppo_sub25-smacv2_5_units"

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
    +system.critic_subsample_fraction=0.25 \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.group=${WANDB_GROUP}"

python -u -m mava.systems.icrl.anakin.ff_ppo_crl -m system.seed=0,1,2,3,4 \
    ${COMMON_ARGS} \
    logger.loggers.wandb.run_name=cppo_sub25-smacv2-5units-V1 \
    logger.loggers.wandb.tags=["cppo_sub25","ff_ppo_crl","smacv2-5units","ablation","BROKEN-needs-retune"]
