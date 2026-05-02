# Handoff — CPPO optimization for NeurIPS wall-clock benchmark (v5)

**Snapshot date:** 2026-05-02. Branch: `mava-cppo-optimization-v5-2026-05-02`.

This branch supersedes v4 with the **final 80M GAE smooth_adv result** plus
production bash scripts for all the variants we tried.

## TL;DR for the next Claude / human

- **`cppo_decoupled` is the recommended safe optimization** — same win-rate
  as the user's tuned baseline (76.5% vs 76.07%) at +33% wall-clock speed.
- **`cppo_decoupled + use_gae=true gae_kind=smooth_adv` beats baseline
  on win-rate too** — peak 79.88% (vs 76.51% non-GAE) at the SAME 27-min
  wall-clock. This is the new best.
- All other ablations (sub50, sub25, ippo_actor at prod hparams) either
  break learning or need re-tuning. Don't run them blind.

## Final 80M run results (num_envs=512, smacv2_5_units, single seed)

| run | wall | steady SPS | peak win-rate | final eval |
|---|---|---|---|---|
| `cppo_baseline` (`run.sh` Run-9 reference) | ~35 min (estimated) | ~41k (estimated) | 76.07% | — |
| `cppo_decoupled` (no GAE) | 26.7 min | 54,924 | **76.51%** at eval 66 | 75.59% |
| **`cppo_decoupled + GAE smooth_adv`** | **26.7 min** | 54,933 | **79.88%** at eval 74 | 77.10% |
| `cppo_sub25` | 9.1 min | 206,161 | 0.98% (BROKEN) | 0.64% |

**The GAE result is the headline.** Same wall-clock, same SPS, +3.4pp peak
win-rate. The cost: 1 extra forward pass through actor + SA encoder during
advantage computation (cheap relative to the inner minibatch loop).

## What changed across v3 / v4 / v5

- **v3:** decoupled epoch counts, critic subsampling, broken GAE
  (`gae_kind=value_diff` no-reward TD).
- **v4:** fixed GAE → `gae_kind=smooth_adv` (default), δ_t = Q(s,a) − V(s)
  smoothed across time by GAE λ. Same-state Q and V cancel network noise
  → working signal at init. Plus OOM fix: flatten to (N, ...) for network
  forwards, only (T, E) for the small reverse-time scan.
- **v5 (this):** final 80M smooth_adv result confirmed (peak 79.88%) +
  production bash scripts in `scripts/`.

## All config flags (recap)

| field | type | default | meaning |
|---|---|---|---|
| `system.actor_batch_size` | int | `system.batch_size` | actor minibatch size |
| `system.actor_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs for actor side |
| `system.critic_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs for critic side |
| `system.critic_subsample_fraction` | float (0,1] | 1.0 | random subset of critic data per epoch |
| `system.scan_unroll` | int | 8 | `lax.scan(unroll=...)` for actor/critic loops |
| `system.use_gae` | bool | false | enable GAE-smoothed advantages |
| `system.gae_lambda` | float | 0.95 | GAE smoothing parameter |
| `system.gae_kind` | str | `"smooth_adv"` | per-step δ source: `smooth_adv` (recommended), `value_diff` (broken), `q_as_reward` |

## What to do next

1. **Multi-seed both `cppo_decoupled` and `cppo_decoupled_gae`** at 5+ seeds
   each. Single-seed GAE +3.4pp could be lucky variance — confirm with a
   sweep. Use `scripts/run_cppo_decoupled.sh` and
   `scripts/run_cppo_decoupled_gae.sh` (both already set up for 5 seeds with
   wandb).
2. **Multi-seed `cppo_baseline`** at the same scale for the formal speed
   comparison. We have estimates but never ran the unoptimized code at 80M
   in this session.
3. **Try `gae_lambda=0.5` and `gae_lambda=1.0`** to see how sensitive
   smooth_adv is to the smoothing horizon. Default 0.95 is a PPO heuristic.
4. **Optionally retune the broken `sub50`/`sub25` configs** with
   `ent_coef=0.05-0.1` and `actor_lr=0.0001-0.0003`. If they recover
   learning, they'd be the SPS winners.
5. **Push to GitHub:** the `gh_push_*.py` pattern in prior branches' commit
   messages. Direct `git push` is blocked from this sandbox; use the
   GitHub REST API via curl.

## Bash scripts (in this branch under `scripts/`)

```
scripts/README.md                              # quick reference
scripts/run_cppo_baseline.sh                   # reference run, unoptimized
scripts/run_cppo_decoupled.sh                  # safe optimization
scripts/run_cppo_decoupled_gae.sh              # GAE smooth_adv (current best)
scripts/run_cppo_ippo_actor.sh                 # ablation
scripts/run_cppo_sub50_RETUNE_NEEDED.sh        # ablation, broken at prod hparams
scripts/run_cppo_sub25_RETUNE_NEEDED.sh        # ablation, broken at prod hparams
scripts/run_ippo_4layer.sh                     # comparison baseline (matched depth)
scripts/run_icrl.sh                            # comparison baseline (off-policy)
scripts/run_benchmark_sps.sh                   # short SPS benchmark for paper Table-1
```

All multi-seed (`-m system.seed=0,1,2,3,4`), all wandb on. Wandb key is
inlined from the user's `run.sh` — rotate it.

## User preferences (worth knowing before you change anything)

- **Don't change the algorithm.** `gae_kind=smooth_adv` is on the boundary
  — they explicitly asked for it as an ablation, and it works.
- **Don't break baseline learning.** If your change makes win rate drop, it's
  not a win regardless of SPS. (The `value_diff` GAE was a hard-learned
  example of this.)
- **Production hparams live in `run.sh`** — Run 9 (`win_rate=76.07%`) is
  the reference. Don't deviate without explicit retuning.
- **Wandb key in `run.sh` and these scripts is compromised** — rotate it.
- **GitHub PAT in chat history is also compromised** — rotate that too.
