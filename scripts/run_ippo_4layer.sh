#!/usr/bin/env bash
# IPPO with 4-layer torsos to match CPPO/ICRL depth — for fair speed comparison
# in the paper. The default IPPO config uses [512, 512] (2 layers); CPPO/ICRL
# use [512, 512, 512, 512] (4 layers). Without this matching, IPPO appears
# ~5x faster than CPPO purely because of architecture, not algorithm.
#
# At unified 4-layer depth, IPPO is still ~3.2x faster than CPPO_decoupled,
# which is the genuine algorithm-cost gap (no goal encoder, no [B,B] InfoNCE
# matrix, num_minibatches=2 vs ~640).

set -euo pipefail
source .venv/bin/activate

export WANDB_API_KEY=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
WANDB_GROUP="ippo_4layer-smacv2_5_units"

python -u -m mava.systems.ppo.anakin.ff_ippo -m system.seed=0,1,2,3,4 \
    arch.num_envs=512 \
    system.num_updates=1250 \
    arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 \
    system.rollout_length=128 \
    system.add_agent_id=true \
    env.scenario.task_name=smacv2_5_units \
    'network.actor_network.pre_torso.layer_sizes=[512,512,512,512]' \
    'network.critic_network.pre_torso.layer_sizes=[512,512,512,512]' \
    logger.loggers.wandb.enabled=true \
    logger.loggers.wandb.project=ssued \
    logger.loggers.wandb.tags=["ippo_4layer","ff_ippo","smacv2-5units","comparison"] \
    logger.loggers.wandb.run_name=ippo_4layer-smacv2-5units-V1 \
    logger.loggers.wandb.group=${WANDB_GROUP}
