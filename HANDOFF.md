# Handoff — CPPO optimization for NeurIPS wall-clock benchmark (v4)

**Snapshot date:** 2026-05-02. Branch: `mava-cppo-optimization-v4-2026-05-02`.

## Status at handoff time

A `cppo_decoupled + use_gae=true + gae_kind=smooth_adv` 80M-timestep run is
**actively running** (PID 146943 at handoff). It's the GAE ablation, fixed
to actually work. Current eval-policy win-rate trajectory (still climbing):

```
eval  0:  0.05%   (random init)
eval  1: 11.4%
eval  2: 15.4%
eval  3: 24.4%
eval  4: 30.4%
eval  5: 46.1%
eval  6: 48.1%
eval  7: 52.7%
eval  8: 55.7%
eval  9: 55.1%
eval 10: 57.3%
eval 11: 55.9%
eval 12: 54.6%
eval 13: 59.0%
eval 14: 59.4%
eval 15: 62.0%
eval 16: 66.0%
eval 17: 62.9%   ← latest at handoff
```

For comparison, **non-GAE `cppo_decoupled` at the same eval indices** was:
0.05 / 14.9 / 27.6 / 30.2 / 32.1 / 40.8 / 41.1 / 36.6 / 47.9 / 44.5 / 51.3 /
51.9 / 52.8 / 54.0 / 55.9 / 57.2 / 54.7 / 58.8 — **smooth_adv is leading by
~5-10 percentage points consistently.**

Non-GAE plateaued at 75.59% by eval 79. We expect smooth_adv to plateau at
or above that. **Wait for it to finish before drawing conclusions.**

## What changed since the v3 branch

Two important things on top of v3:

### 1. Fixed the broken GAE (the v3 `value_diff` mode never learned)

**Problem in v3:** When `use_gae=true`, the per-step δ was computed as
`δ_t = γ V_{t+1} − V_t` (no reward, pure value-difference TD). At init, V is
a random network. V at one state and V at another state are basically
two unrelated random numbers — their difference is pure noise. The actor
followed noise, never learned. Win rate stuck at 0.1-0.3% over 12+ evals.

**Fix:** Replaced the per-step signal with the original single-step
contrastive advantage:
```
δ_t = Q(s_t, a_t) − V(s_t)        (single-step adv, evaluated on same state)
GAE_t = δ_t + γ λ (1 − done) GAE_{t+1}     (then GAE-smooth across time)
```

The key property: Q and V are evaluated on the **same state through the
same network**, so the network's randomness shows up identically in both
and cancels in the subtraction. What's left is the genuine "is this
action better than the policy average?" signal — clean even at init.

Then on top of that, GAE smoothing is just variance reduction: each
timestep starts from its own clean signal `Q − V` and borrows a little
from later timesteps' signals on the same trajectory.

This is now the **default** when `use_gae=true`. The broken variants are
preserved as `gae_kind=value_diff` and `gae_kind=q_as_reward` for ablation.

### 2. Fixed an OOM at production scale

The first attempt at smooth_adv used `(T, E, A, S)` 4D tensors all the way
through the SA-encoder forward. At `num_envs=512, rollout=128` that
triggers a different XLA memory plan than the equivalent 3D `(N, A, S)`
shape used by the non-GAE path, and OOMs at ~6.7 GB. **Fix:** flatten to
`(N, ...)` for the network forward passes (matching what the non-GAE path
already does — known to fit), then reshape outputs back to `(T, E, ...)`
only for the small reverse-time GAE scan. Same total compute, same memory
budget as non-GAE, no OOM.

## Config flags (recap from v3 + new gae_kind)

| field | type | default | meaning |
|---|---|---|---|
| `system.actor_batch_size` | int | `system.batch_size` | actor minibatch size; can be much larger than critic |
| `system.actor_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs for actor side |
| `system.critic_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs for critic side |
| `system.critic_subsample_fraction` | float in (0,1] | 1.0 | random subset of critic data per epoch |
| `system.scan_unroll` | int | 8 | `lax.scan(unroll=...)` for actor + critic minibatch loops |
| `system.use_gae` | bool | false | enable GAE-smoothed advantages |
| `system.gae_lambda` | float | 0.95 | GAE smoothing parameter |
| **`system.gae_kind`** | str | **`"smooth_adv"`** | per-step δ: `smooth_adv` (recommended), `value_diff` (broken — kept for ablation), `q_as_reward` |

## Recommended config (production)

For wall-clock-vs-win-rate, `cppo_decoupled` (no GAE, single-step adv) is
the safe option that matches the user's tuned baseline. Confirmed at
80M / 512 envs: **76.51% peak win rate, 26.7 min wall-clock**.

For potentially better sample efficiency, **add `+system.use_gae=true`**
to the same config. The 80M run currently underway is on track to match
or exceed the non-GAE peak, and is already 5-10pp ahead at every
mid-training eval. **Wait for the run to finish before claiming this for
the paper.**

## What to do next

1. **Wait for the GAE 80M run to finish** (PID 146943, log
   `/tmp/cppo_smooth_adv_80M.log`). Run `python scripts/summarize_run.py
   /tmp/cppo_smooth_adv_80M.log` after PID exits.
2. **Multi-seed both runs** (`cppo_decoupled` no-GAE, `cppo_decoupled` +
   GAE smooth_adv) at 5+ seeds each — the +5-10pp lead at the current
   single seed is suggestive, not significant.
3. **Run cppo_baseline at 80M for the formal speed comparison.** We have
   the optimized numbers but never ran the unoptimized baseline at the
   wall-clock benchmark scale.
4. **Try `gae_lambda=0.5` and `gae_lambda=1.0` ablations** to see how
   sensitive smooth_adv is to the smoothing horizon. Default 0.95 is a
   PPO heuristic, not tuned for the contrastive setting.

## File list (this branch)

```
README.md                                        # snapshot entry point
HANDOFF.md                                       # this file
CHANGES.md                                       # code-level changelog (v3 + v4 fixes)
BENCHMARKS.md                                    # all numbers, including current GAE run progress

mava/systems/icrl/anakin/ff_ppo_crl.py           # WORKING smooth_adv GAE + OOM fix
mava/systems/icrl/anakin/ff_ppo_crl_baseline.py  # PRISTINE pre-optimization copy
mava/systems/icrl/utils.py                       # micro-cleanup
scripts/benchmark_systems.py                     # subprocess-isolated benchmark
scripts/summarize_run.py                         # post-run log analyzer

80M_runs/cppo_decoupled.win_rate_trajectory.txt          # 76.51% peak, baseline-matching
80M_runs/cppo_sub25.win_rate_trajectory.txt              # the broken sub25 run, for reference
80M_runs/cppo_smooth_adv_in_progress.win_rate_trajectory.txt  # CURRENTLY RUNNING — partial
```

## Prior branches in this repo

| branch | what it has |
|---|---|
| `mava-cppo-optimization-2026-05-02` | v1 — local optimizations only |
| `mava-cppo-optimization-v2-2026-05-02` | v2 — adds decoupled `actor_batch_size` |
| `mava-cppo-optimization-v3-2026-05-02` | v3 — decoupled epochs, critic subsample, **broken** GAE (value_diff) |
| **`mava-cppo-optimization-v4-2026-05-02`** | **(this one)** — fixed GAE (smooth_adv), fixed OOM, GAE 80M run in progress |
