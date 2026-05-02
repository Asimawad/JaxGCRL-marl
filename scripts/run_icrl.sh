#!/usr/bin/env bash
# ICRL — off-policy contrastive RL with replay buffer (SAC backbone).
# Comparison baseline for the paper. Same env, same architecture depth as
# CPPO. Hyperparameters from configs/system/icrl/ff_icrl.yaml defaults.

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="icrl-smacv2_5_units"

python -u -m mava.systems.icrl.anakin.ff_icrl -m system.seed=0,1,2,3,4 \
    env=smax \
    env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 \
    system.num_updates=1250 \
    arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 \
    system.rollout_length=128 \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.tags=["icrl","ff_icrl","smacv2-5units","comparison"] \
    logger.loggers.wandb.run_name=icrl-smacv2-5units-V1 \
    logger.loggers.wandb.group=${WANDB_GROUP}
