#!/bin/bash
# JaxGCRL CRL off-policy baseline for Brax envs
# Runs from docs/JaxGCRL/ using tyro CLI (not Hydra)
#
# Usage: bash run_jaxgcrl_baseline.sh [ant|reacher|ant_u_maze]
# If no arg given, runs all three envs sequentially.

# Reads WANDB_API_KEY from env. Set via: export WANDB_API_KEY=...
: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running}"

source .venv/bin/activate

DATE=04-25-2026
WANDB_PROJECT=ant
WANDB_GROUP=brax-benchmark-jaxgcrl-crl-$DATE

ENVS=(ant)
if [[ -n "$1" ]]; then
    ENVS=("$1")
fi


WANDB_PROJECT=jaxnav
WANDB_ENTITY=asim_osman-aimst-university 
DATE=04-25-2026

# ══════════════════════════════════════════════════════════════════════════════
# SHARED CPPO ARGS
# ══════════════════════════════════════════════════════════════════════════════
# CPPO_ARGS=(
#     arch.num_envs=256
#     arch.num_evaluation=25
#     arch.num_eval_episodes=512
#     arch.evaluation_greedy=false
#     arch.absolute_metric=false
#     system.num_updates=3000
#     system.rollout_length=128
#     system.num_epochs=8
#     system.batch_size=256
#     system.num_mc_samples=32
#     system.rep_size=64
#     system.energy_fn=dot
#     system.contrastive_loss_fn=fwd_infonce
#     system.contrastive_temperature=2.5
#     system.logsumexp_penalty_coeff=0.1
#     system.num_critic_warmup_epochs=1
#     system.use_achieved_goal=true
#     system.use_gae=false
#     system.reward_advantage_coeff=0.0
#     system.use_reinforce=false
#     system.use_adaptive_entropy=false
#     system.target_entropy=4
#     system.lr_linear_decay=false
#     system.gamma=0.9999
#     system.actor_lr=1.5e-6
#     system.q_lr=4e-6
#     system.clip_eps=0.15
#     system.ent_coef=0.25
#     system.ent_coef_end=0.01
#     system.log_std_min=-5
#     system.log_std_max=2
#     system.max_grad_norm=1
#     system.gae_lambda=0.95
#     system.update_batch_size=1
#     system.add_agent_id=false
#     system.use_old_wrapper=false
#     logger.loggers.wandb.enabled=false
#     logger.loggers.wandb.project=$WANDB_PROJECT
#     logger.loggers.wandb.entity=$WANDB_ENTITY
#     logger.loggers.wandb.tags=["brax","ppo_crl_continuous","no-reward","benchmark"]
#     logger.loggers.wandb.group=brax-benchmark-cppo-$DATE
# )

# ══════════════════════════════════════════════════════════════════════════════
# ██  CPPO  ──  Ant
# ══════════════════════════════════════════════════════════════════════════════
uv run mava/systems/icrl/anakin/ppo_crl_continuous.py \
    --config-name ppo_crl_brax \
    env=brax_ant \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=$WANDB_PROJECT \
    logger.loggers.wandb.entity=$WANDB_ENTITY \
    logger.loggers.wandb.tags=["brax","ppo_crl_continuous","no-reward","benchmark"] \
    logger.loggers.wandb.group=brax-benchmark-cppo-$DATE \
    logger.loggers.wandb.run_name=cppo-brax-ant-$DATE

# ══════════════════════════════════════════════════════════════════════════════
# ██  CPPO  ──  Reacher
# ══════════════════════════════════════════════════════════════════════════════
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     --config-name ppo_crl_brax \
#     "${CPPO_ARGS[@]}" \
#     env=brax_reacher \
#     system.target_entropy=1 \
#     logger.loggers.wandb.run_name=cppo-brax-reacher-$DATE

# # ══════════════════════════════════════════════════════════════════════════════
# # ██  CPPO  ──  AntMaze (U-maze)
# # ══════════════════════════════════════════════════════════════════════════════
# uv run mava/systems/icrl/anakin/ppo_crl_continuous.py -m \
#     --config-name ppo_crl_brax \
#     "${CPPO_ARGS[@]}" \
#     env=brax_ant_maze \
#     logger.loggers.wandb.run_name=cppo-brax-ant_umaze-$DATE

# ══════════════════════════════════════════════════════════════════════════════
# ██  JaxGCRL CRL BASELINE  (off-policy, run from docs/JaxGCRL)
# Only executed when --jaxgcrl flag is passed
# ══════════════════════════════════════════════════════════════════════════════
# if [[ "$1" == "--jaxgcrl" ]]; then
#     echo "Running JaxGCRL CRL baseline..."
#     cd docs/JaxGCRL

#     for ENV in ant reacher ant_u_maze; do
#         for SEED in 0; do
#             uv run python run.py \
#                 run.env=$ENV \
#                 run.seed=$SEED \
#                 run.num_envs=256 \
#                 run.episode_length=1001 \
#                 run.num_steps=3000000 \
#                 run.wandb_project=$WANDB_PROJECT \
#                 run.wandb_entity=$WANDB_ENTITY \
#                 run.group=brax-benchmark-jaxgcrl-crl-$DATE \
#                 run.name=jaxgcrl-crl-$ENV-seed$SEED
#         done
#     done

#     cd ../..
# fi
