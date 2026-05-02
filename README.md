# mava CPPO optimization snapshot v3 — 2026-05-02

This is an **orphan branch** (no parent commit) checkpoint of work-in-progress
on the CPPO trainer (`mava/systems/icrl/anakin/ff_ppo_crl.py`) for a NeurIPS
wall-clock benchmark.

**If you're a Claude or human picking this up: read [HANDOFF.md](HANDOFF.md)
first.** It has the user's preferences, what's been tried, what works, what
broke, and what to do next.

## Contents

| file | purpose |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **Start here.** Project goal, what's done, gotchas, next steps. |
| [CHANGES.md](CHANGES.md) | Code-level changelog: what changed in each file, what config knobs were added. |
| [BENCHMARKS.md](BENCHMARKS.md) | All benchmark numbers + commands to reproduce. |
| `mava/systems/icrl/anakin/ff_ppo_crl.py` | Optimized v3 trainer. |
| `mava/systems/icrl/anakin/ff_ppo_crl_baseline.py` | Pristine pre-optimization copy for A/B benchmarking. |
| `mava/systems/icrl/utils.py` | Micro-cleanup in `flatten_crl_fn`. |
| `scripts/benchmark_systems.py` | Subprocess-isolated CPPO/ICRL/IPPO benchmark with `--per-system` overrides. |
| `scripts/summarize_run.py` | Post-run log analyzer (per-eval SPS, win-rate trajectory). |
| `80M_runs/` | Win-rate trajectories from the production-scale runs. |

## Prior branches in this repo

| branch | what it has |
|---|---|
| `mava-cppo-optimization-2026-05-02` | v1 — local optimizations only. ~3% wall-clock at scale. |
| `mava-cppo-optimization-v2-2026-05-02` | v2 — adds decoupled `actor_batch_size`. +11% at short benchmark. |
| **`mava-cppo-optimization-v3-2026-05-02`** | **(this one)** — adds decoupled epoch counts, critic subsampling, GAE ablation, plus 80M-timestep production-run results. |

Each branch is independent (orphan), so any one of them can be checked out
and applied on top of a fresh `mava` checkout.

## Headline result

`cppo_decoupled` (the recommended config) at 80M timesteps, 512 envs, on
`smacv2_5_units` with the user's tuned hparams:

| | |
|---|---|
| Wall clock | **26.7 min** |
| Peak eval win rate | **76.51%** at eval 66 |
| Final eval | 75.59% |

Matches the user's `run.sh` Run-9 baseline (`win_rate=76.07%`) within seed
noise, while running ~25-30% faster than baseline at the matched
`ppo_epochs=2` config.

## Recommended config to reproduce

```bash
python -u -m mava.systems.icrl.anakin.ff_ppo_crl \
    --config-name ppo_crl env=smax env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 system.num_updates=1250 arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 system.rollout_length=128 \
    system.batch_size=256 system.ppo_epochs=2 \
    +system.actor_batch_size=512 \
    system.actor_lr=0.0005 system.q_lr=0.0001 system.ent_coef=0.01 \
    system.max_grad_norm=0.05 system.clip_eps=0.2 \
    system.logsumexp_penalty_coeff=0.91 system.lr_end=1e-07 \
    system.lr_decay_type=linear system.add_agent_id=1
```

Add `logger.loggers.wandb.enabled=true` to log to wandb.

## Things that didn't work (read before retrying)

- **`cppo_sub25` (critic subsampling at 0.25)** with the IPPO-style actor
  + Run-9 hparams: 3× faster wall-clock but **broke learning entirely**
  (peak win rate 0.98%). The hparams are not robust to that aggressive a
  config change.
- **`use_gae=true` (GAE bootstrapping over contrastive V, no explicit
  reward)**: at handoff time this run was at eval 12+ stuck around 0.1-0.3%
  win rate. The pure value-difference TD signal at init (when V is random)
  is too weak. Possible fixes in [HANDOFF.md](HANDOFF.md).
