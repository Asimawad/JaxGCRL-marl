#!/usr/bin/env bash
# Short SPS benchmark for the paper Table-1 — runs all systems sequentially
# in subprocess-isolated mode and dumps a JSON with steady-state SPS,
# compile time, peak GPU memory, and per-eval win rates.
#
# Each subprocess runs 12 updates / 6 evals (~3 min). Total ~30-40 min for
# all 5 systems. No wandb (sandbox-friendly); results land in
# benchmark_results.json and benchmark_logs/.
#
# IPPO is bumped to 4-layer torsos (matching CPPO/ICRL) for fair comparison.
# cppo_decoupled uses actor_batch_size=512.
# cppo_ippo_actor uses IPPO-style actor (4 epochs, batch=N/2).

set -euo pipefail
source .venv/bin/activate

# At num_envs=256, rollout_length=128, num_agents=5: N = 163,840.
# IPPO-style means 2 minibatches per actor epoch -> actor_batch = N/2 = 81920.
ACTOR_BATCH_IPPO_STYLE=81920

python scripts/benchmark_systems.py \
    --num_envs 256 \
    --num_updates 12 \
    --num_evaluation 6 \
    --rollout_length 128 \
    --num_eval_episodes 16 \
    --systems cppo_baseline cppo_decoupled cppo_ippo_actor icrl ippo \
    --per-system "cppo_decoupled:+system.actor_batch_size=512" \
    --per-system "cppo_ippo_actor:+system.actor_batch_size=${ACTOR_BATCH_IPPO_STYLE}" \
    --per-system "cppo_ippo_actor:+system.actor_ppo_epochs=4" \
    --per-system "cppo_ippo_actor:+system.critic_ppo_epochs=1" \
    --per-system "ippo:network.actor_network.pre_torso.layer_sizes=[512,512,512,512]" \
    --per-system "ippo:network.critic_network.pre_torso.layer_sizes=[512,512,512,512]" \
    --out benchmark_results.json
