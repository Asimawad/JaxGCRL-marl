#!/bin/bash


# Reads WANDB_API_KEY from env. Set via: export WANDB_API_KEY=...
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate
# uv run mava/systems/icrl/anakin/ff_icrl.py -m system.seed=0,1,2,3,4 \
#     env=jaxnav \
#     arch.num_envs=256 \
#     arch.num_evaluation=100 \
#     arch.absolute_metric=false \
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
#     +system.use_old_wrapper=false \
#     env.goal_type=distance \
#     env.eval_metric=success \

uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m system.seed=0,1,2,3,4 \
    env=jaxnav \
    arch.num_envs=256 \
    arch.num_evaluation=25 \
    arch.num_eval_episodes=1024 \
    arch.evaluation_greedy=false \
    system.num_updates=3051 \
    system.rollout_length=256 \
    system.num_epochs=4 \
    system.batch_size=512 \
    system.actor_lr=3e-4 \
    system.q_lr=3e-4 \
    system.gamma=0.99 \
    system.clip_eps=0.2 \
    system.max_grad_norm=0.5 \
    system.lr_linear_decay=false \
    system.ent_coef=0.01 \
    system.ent_coef_end=0.005 \
    system.rep_size=64 \
    system.num_mc_samples=32 \
    system.logsumexp_penalty_coeff=0.1 \
    system.energy_fn=norm \
    system.contrastive_loss_fn=sym_infonce \
    system.log_std_min=-1.0 \
    system.num_critic_warmup_epochs=4 \
    system.use_achieved_goal=true \
    system.use_gae=false \
    system.reward_advantage_coeff=0.0 \
    system.use_reinforce=false \
    env.goal_type=distance \
    env.eval_metric=success \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=jaxnav \
    logger.loggers.wandb.entity=asim_osman-aimst-university \
    logger.loggers.wandb.tags=["jaxnav","ppo_crl_continuous","no-reward","200M"]' \
    logger.loggers.wandb.group="ppo_crl_continuous-sweep-04-24-2026" \
    logger.loggers.wandb.run_name="ppo_crl_continuous-jaxnav-sweep-04-24-2026"