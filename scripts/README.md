# Production scripts

Each script reproduces one of the runs we did. All assume `.venv/` exists at
repo root. All have the user's wandb key inlined (rotate it!).

## Quick reference

| script | what | wall (1 seed) | peak win-rate |
|---|---|---|---|
| `run_cppo_baseline.sh` | original `ff_ppo_crl_baseline.py`, no optimizations | ~35 min | 76.07% (`run.sh` Run-9) |
| **`run_cppo_decoupled.sh`** | **safe optimized config (recommended)** | **~27 min** | **76.51%** |
| **`run_cppo_decoupled_gae.sh`** | **adds GAE smooth_adv on top of decoupled** | **~27 min** | **79.88%** |
| `run_cppo_ippo_actor.sh` | IPPO-style actor (ablation) | ~25 min | TBD — may need hparam re-tune |
| `run_cppo_sub50_RETUNE_NEEDED.sh` | + critic_subsample=0.5 | ~18 min | BROKEN at prod hparams; retune |
| `run_cppo_sub25_RETUNE_NEEDED.sh` | + critic_subsample=0.25 | ~9 min | BROKEN at prod hparams; retune |
| `run_ippo_4layer.sh` | IPPO with 4-layer torsos for fair comparison | ~10 min | (reward-PPO baseline) |
| `run_icrl.sh` | off-policy contrastive (SAC + replay) | ~25 min | (off-policy baseline) |
| `run_benchmark_sps.sh` | short SPS benchmark, all 5 systems sequentially | ~30 min | (SPS only) |

All multi-seed (`-m system.seed=0,1,2,3,4`). Wandb on, project `ssued`.

## Recommended order

1. **`run_benchmark_sps.sh`** first — short, gives the SPS comparison table
   for paper Table-1.
2. **`run_cppo_decoupled.sh`** — confirms the optimized version matches the
   tuned baseline.
3. **`run_cppo_decoupled_gae.sh`** — confirms the GAE smooth_adv lift
   (~+3.4pp peak win-rate at the same wall-clock).
4. **`run_cppo_baseline.sh`** — reference run with the unoptimized code, for
   the head-to-head wall-clock vs win-rate comparison in the paper.
5. **`run_ippo_4layer.sh`** + **`run_icrl.sh`** — comparison baselines.

## Things to know

- **Win-rate is on a 0-100 scale** (`mava/utils/logger.py:82-83`). `Win
  rate: 76.51` means 76.51%, not 0.7651.
- **The first eval has JIT compile time baked in**. Eval 0 SPS will look
  artificially low (~few hundred / sec). Steady-state from eval 2 onward.
- **Eval 1 also has a residual ~30s warmup** (likely async eval bleeding
  into the next `learn()`). The benchmark script drops first 2 evals from
  the steady-state SPS calc.
- **The "RETUNE_NEEDED" scripts** are documented broken at the user's
  baseline hparams. The header of each lists what to retune.
- **WandB key is inlined** — rotate after running.
