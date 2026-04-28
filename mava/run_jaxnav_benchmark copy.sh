#!/bin/bash
# JaxNav Benchmark: ff_ippo, ff_mappo, ff_icrl, ppo_crl_continuous
# Single-agent (num_agents=1) and multi-agent (num_agents=2) variants.
#
# All algorithms target ~200M environment timesteps:
#   PPO-based  : 256 envs × 256 rollout × 3051 updates = 200,015,872
#   ICRL (SAC) : 256 envs × 100 rollout × 7812 updates = 199,987,200
#
# Evaluation protocol (identical for all):
#   25 checkpoints, 1024 episodes per checkpoint, stochastic, no absolute metric

# Reads WANDB_API_KEY from env. Set via: export WANDB_API_KEY=...
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate

WANDB_PROJECT=jaxnav
WANDB_ENTITY=asim_osman-aimst-university
DATE=04-25-2026

# ══════════════════════════════════════════════════════════════════════════════
# SHARED ARCH / EVAL OVERRIDES  (same for every algorithm)
# ══════════════════════════════════════════════════════════════════════════════
EVAL_ARGS=(
    arch.num_envs=256
    arch.num_evaluation=25
    arch.num_eval_episodes=1024
    arch.evaluation_greedy=false
    arch.absolute_metric=false
    env=jaxnav
    env.goal_type=distance
    env.eval_metric=success
)

# ══════════════════════════════════════════════════════════════════════════════
# ██  MULTI-AGENT  (num_agents=2)
# ══════════════════════════════════════════════════════════════════════════════
    # system.seed=0,1,2,3,4 \
# ── FF-IPPO | multi-agent ─────────────────────────────────────────────────────
uv run mava/systems/ppo/anakin/ff_ippo.py \
    "${EVAL_ARGS[@]}" \
    env.kwargs.num_agents=2 \
    system.rollout_length=256 \
    system.num_updates=3051 \
    system.update_batch_size=1 \
    system.ppo_epochs=4 \
    system.num_minibatches=2 \
    system.actor_lr=2.5e-4 \
    system.critic_lr=2.5e-4 \
    system.gamma=0.99 \
    system.gae_lambda=0.95 \
    system.clip_eps=0.2 \
    system.ent_coef=0.01 \
    system.vf_coef=0.5 \
    system.max_grad_norm=0.5 \
    system.decay_learning_rates=false \
    system.add_agent_id=true \
    system.use_old_wrapper=true \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=$WANDB_PROJECT \
    logger.loggers.wandb.entity=$WANDB_ENTITY \
    'logger.loggers.wandb.tags=[jaxnav,ff_ippo,200M,benchmark,multi-agent]' \
    logger.loggers.wandb.group=jaxnav-benchmark-multiagent-$DATE \
    logger.loggers.wandb.run_name=ff_ippo-jaxnav-2agent-$DATE

# # ── FF-MAPPO | multi-agent ────────────────────────────────────────────────────
# uv run mava/systems/ppo/anakin/ff_mappo.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=2 \
#     system.rollout_length=256 \
#     system.num_updates=3051 \
#     system.update_batch_size=1 \
#     system.ppo_epochs=4 \
#     system.num_minibatches=2 \
#     system.actor_lr=2.5e-4 \
#     system.critic_lr=2.5e-4 \
#     system.gamma=0.99 \
#     system.gae_lambda=0.95 \
#     system.clip_eps=0.2 \
#     system.ent_coef=0.01 \
#     system.vf_coef=0.5 \
#     system.max_grad_norm=0.5 \
#     system.decay_learning_rates=false \
#     system.add_agent_id=true \
#     system.use_old_wrapper=true \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ff_mappo,200M,benchmark,multi-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-multiagent-$DATE \
#     logger.loggers.wandb.run_name=ff_mappo-jaxnav-2agent-$DATE

# # ── FF-ICRL (SAC-based) | multi-agent ────────────────────────────────────────
# # Config from run.sh (commented section). 200M: 256 envs × 100 rollout × 7812 updates.
# uv run mava/systems/icrl/anakin/ff_icrl.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=2 \
#     system.num_updates=7812 \
#     system.rollout_length=100 \
#     system.buffer_size=5000 \
#     system.batch_size=256 \
#     system.explore_steps=1000 \
#     system.policy_lr=2e-5 \
#     system.q_lr=2e-5 \
#     system.alpha_lr=5e-6 \
#     system.max_grad_norm=0.5 \
#     system.gamma=0.99 \
#     system.target_entropy_scale=3.0 \
#     system.init_alpha=0.1 \
#     system.logsumexp_penalty_coeff=0.1 \
#     system.add_agent_id=true \
#     system.use_old_wrapper=false \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ff_icrl,200M,benchmark,multi-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-multiagent-$DATE \
#     logger.loggers.wandb.run_name=ff_icrl-jaxnav-2agent-$DATE

# # ── PPO-CRL (CPPO) | multi-agent ─────────────────────────────────────────────
# # No-reward contrastive PPO — baseline from run.sh.
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=2 \
#     system.num_updates=3051 \
#     system.rollout_length=256 \
#     system.num_epochs=4 \
#     system.batch_size=512 \
#     system.actor_lr=3e-4 \
#     system.q_lr=3e-4 \
#     system.gamma=0.99 \
#     system.clip_eps=0.2 \
#     system.max_grad_norm=0.5 \
#     system.lr_linear_decay=false \
#     system.ent_coef=0.01 \
#     system.ent_coef_end=0.005 \
#     system.rep_size=64 \
#     system.num_mc_samples=32 \
#     system.logsumexp_penalty_coeff=0.1 \
#     system.energy_fn=norm \
#     system.contrastive_loss_fn=sym_infonce \
#     system.log_std_min=-1.0 \
#     system.num_critic_warmup_epochs=4 \
#     system.use_achieved_goal=true \
#     system.use_gae=false \
#     system.reward_advantage_coeff=0.0 \
#     system.use_reinforce=false \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ppo_crl_continuous,no-reward,200M,benchmark,multi-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-multiagent-$DATE \
#     logger.loggers.wandb.run_name=ppo_crl-jaxnav-2agent-$DATE

# # ══════════════════════════════════════════════════════════════════════════════
# # ██  SINGLE-AGENT  (num_agents=1)
# # ══════════════════════════════════════════════════════════════════════════════

# # ── FF-IPPO | single-agent ───────────────────────────────────────────────────
# uv run mava/systems/ppo/anakin/ff_ippo.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=1 \
#     system.rollout_length=256 \
#     system.num_updates=3051 \
#     system.update_batch_size=1 \
#     system.ppo_epochs=4 \
#     system.num_minibatches=2 \
#     system.actor_lr=2.5e-4 \
#     system.critic_lr=2.5e-4 \
#     system.gamma=0.99 \
#     system.gae_lambda=0.95 \
#     system.clip_eps=0.2 \
#     system.ent_coef=0.01 \
#     system.vf_coef=0.5 \
#     system.max_grad_norm=0.5 \
#     system.decay_learning_rates=false \
#     system.add_agent_id=false \
#     system.use_old_wrapper=true \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ff_ippo,200M,benchmark,single-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-singleagent-$DATE \
#     logger.loggers.wandb.run_name=ff_ippo-jaxnav-1agent-$DATE

# # ── FF-MAPPO | single-agent ──────────────────────────────────────────────────
# uv run mava/systems/ppo/anakin/ff_mappo.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=1 \
#     system.rollout_length=256 \
#     system.num_updates=3051 \
#     system.update_batch_size=1 \
#     system.ppo_epochs=4 \
#     system.num_minibatches=2 \
#     system.actor_lr=2.5e-4 \
#     system.critic_lr=2.5e-4 \
#     system.gamma=0.99 \
#     system.gae_lambda=0.95 \
#     system.clip_eps=0.2 \
#     system.ent_coef=0.01 \
#     system.vf_coef=0.5 \
#     system.max_grad_norm=0.5 \
#     system.decay_learning_rates=false \
#     system.add_agent_id=false \
#     system.use_old_wrapper=true \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ff_mappo,200M,benchmark,single-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-singleagent-$DATE \
#     logger.loggers.wandb.run_name=ff_mappo-jaxnav-1agent-$DATE

# # ── FF-ICRL (SAC-based) | single-agent ───────────────────────────────────────
# uv run mava/systems/icrl/anakin/ff_icrl.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=1 \
#     system.num_updates=7812 \
#     system.rollout_length=100 \
#     system.buffer_size=5000 \
#     system.batch_size=256 \
#     system.explore_steps=1000 \
#     system.policy_lr=2e-5 \
#     system.q_lr=2e-5 \
#     system.alpha_lr=5e-6 \
#     system.max_grad_norm=0.5 \
#     system.gamma=0.99 \
#     system.target_entropy_scale=3.0 \
#     system.init_alpha=0.1 \
#     system.logsumexp_penalty_coeff=0.1 \
#     system.add_agent_id=false \
#     system.use_old_wrapper=false \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ff_icrl,200M,benchmark,single-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-singleagent-$DATE \
#     logger.loggers.wandb.run_name=ff_icrl-jaxnav-1agent-$DATE

# # ── PPO-CRL (CPPO) | single-agent ────────────────────────────────────────────
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     system.seed=0,1,2,3,4 \
#     "${EVAL_ARGS[@]}" \
#     env.kwargs.num_agents=1 \
#     system.num_updates=3051 \
#     system.rollout_length=256 \
#     system.num_epochs=4 \
#     system.batch_size=512 \
#     system.actor_lr=3e-4 \
#     system.q_lr=3e-4 \
#     system.gamma=0.99 \
#     system.clip_eps=0.2 \
#     system.max_grad_norm=0.5 \
#     system.lr_linear_decay=false \
#     system.ent_coef=0.01 \
#     system.ent_coef_end=0.005 \
#     system.rep_size=64 \
#     system.num_mc_samples=32 \
#     system.logsumexp_penalty_coeff=0.1 \
#     system.energy_fn=norm \
#     system.contrastive_loss_fn=sym_infonce \
#     system.log_std_min=-1.0 \
#     system.num_critic_warmup_epochs=4 \
#     system.use_achieved_goal=true \
#     system.use_gae=false \
#     system.reward_advantage_coeff=0.0 \
#     system.use_reinforce=false \
#     logger.loggers.wandb.enabled=true \
#     logger.loggers.wandb.project=$WANDB_PROJECT \
#     logger.loggers.wandb.entity=$WANDB_ENTITY \
#     'logger.loggers.wandb.tags=[jaxnav,ppo_crl_continuous,no-reward,200M,benchmark,single-agent]' \
#     logger.loggers.wandb.group=jaxnav-benchmark-singleagent-$DATE \
#     logger.loggers.wandb.run_name=ppo_crl-jaxnav-1agent-$DATE
