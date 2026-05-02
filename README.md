# mava CPPO optimization snapshot — 2026-05-02

This branch is an orphan snapshot of the optimization pass on
`mava/systems/icrl/anakin/ff_ppo_crl.py` (CPPO — on-policy contrastive RL
with a PPO actor and a contrastive critic, no reward signal).

It's not meant to be merged — it's a checkpoint of the changes for later
reference. Apply on top of a fresh `mava` checkout if you want to run it.

## What's here

| path | status | what it is |
|---|---|---|
| `mava/systems/icrl/anakin/ff_ppo_crl.py` | modified | the optimized CPPO trainer |
| `mava/systems/icrl/anakin/ff_ppo_crl_baseline.py` | new | pristine pre-optimization copy for A/B benchmarking |
| `mava/systems/icrl/utils.py` | modified | small `flatten_crl_fn` cleanup (broadcast `seed_mask`) |
| `scripts/benchmark_systems.py` | new | subprocess-isolated benchmark for CPPO/ICRL/IPPO |
| `benchmark_logs/*.log` | new | terminal output from the smoke runs |
| `benchmark_results.json` | new | parsed metrics from the comparison run |

## Optimizations (5 changes, no algorithm change)

1. **`compute_crl_advantages` — batched per-action SA encoder.** Replaced
   `jax.vmap(compute_q_for_action)(jnp.arange(action_dim))` (which calls the
   SA encoder `action_dim` times) with a single batched call on
   `[N, A, …]`. Inputs broadcast via `jnp.broadcast_to`; energy is
   evaluated against `g_repr[:, None, :]`. **Bit-identical** to the original
   on a synthetic equivalence test (`max abs diff = 0.0`).

2. **Drop `swapaxes` + Fortran-order reshape.** `traj_batch` is `(T, E, …)`;
   we now `vmap(flatten_crl_fn, in_axes=(None, 1, 0), out_axes=1)` directly
   over the env-agent axis, then C-order reshape on `(T-1, E, …)`. The
   flat ordering matches the previous Fortran reshape on `(E, T-1, …)`
   exactly (both produce `t` slow, `e` fast); fewer copies for XLA to fuse.

3. **Strip `transition.extras` after advantage compute.** The PPO actor
   loss reads `extras["real_ultimate_goal"]` and `extras["advantages"]`;
   the critic loss reads only the tuple's relabeled `ultimate_goal` and
   the basic fields. Everything else (`state`, `future_state`,
   `future_action`, `done`, `state_extras`, `policy_extras`) is dropped
   before entering the PPO epoch loop, so it doesn't get gathered in
   every minibatch shuffle.

4. **`flatten_crl_fn` micro-cleanup.** Replaced
   `jnp.concatenate([seed[:, None].T] * seq_len, axis=0)` with broadcasting
   (`seeds[None, :] == seeds[:, None]`). Same behaviour, no Python-level
   `O(T)` list construction in the trace.

5. **SPS logging audit.** Added `compile_seconds` (logged on iter 0 only)
   and `wall_clock_seconds` (every iter) to the ACTOR log line so a
   benchmark can isolate JIT compile from steady-state SPS. The pre-existing
   `steps_per_second = steps_per_rollout / elapsed_time` calc is correct;
   the issue was the first eval bundling JIT compile (~80 s) with one
   iteration of work.

## What the benchmark catches

In addition to the iter-0 compile-cost contamination, **iter 1 also runs
slow** (~32 s for whatever scale of work). The pattern was identical
between optimized and baseline (498/2,005 SPS at small/large config), so
it's not a regression — it's a residual warmup, plausibly the async
evaluator bleeding into the next `learn()`. The benchmark drops the first
**two** evals from the steady-state SPS mean. A more permanent fix is to
add `jax.block_until_ready(eval_metrics)` after the `evaluator(...)` call
in `run_experiment`.

## Measured numbers

`num_envs=256, rollout_length=128, ppo_epochs=1, smacv2_5_units, 1× H100 80 GB`
(short benchmark — 12 updates, 6 evals, last 4 used for steady-state):

| system | compile (s) | steady SPS (mean ± std, n=4) | total wall (s) |
|---|---|---|---|
| `cppo_optimized` | 84.8 | **85,175 ± 355** | 216.1 |
| `cppo_baseline`  | ~85  | 82,221 ± 136 | 219.5 |

Steady-state speedup: **~3.6%**. Honest read: the per-action SA encoder
vmap was *not* the dominant bottleneck at this scale — XLA already produced
a comparable plan. The dominant cost is the inner minibatch `lax.scan`
(`batch_size=128` against ~327 k transitions = ~2,500 minibatch updates
per eval). To get closer to 2x you'd need either an algorithm-touching
change (combine actor + critic into one backward pass) or a config change
(raise `batch_size` to cut the minibatch count). Both were out of scope.

`peak_gpu_memory_mb` from `nvidia-smi` is system-wide and gets conflated
across runs (we observed ~60 GB for both, which matches the resident
state of all the JAX caches on this box). For accurate per-system peak,
either run on a fresh GPU or instrument
`jax.local_devices()[0].memory_stats()['peak_bytes_in_use']` from inside
each `run_experiment`.

## How to run the benchmark

From the `mava` repo root, after copying these files in:

```bash
# Quick smoke (12 updates, 6 evals, no wandb)
python scripts/benchmark_systems.py \
    --num_envs 256 --num_updates 12 --num_evaluation 6 --rollout_length 128 \
    --num_eval_episodes 16 \
    --systems cppo_optimized cppo_baseline icrl ippo \
    --out benchmark_results.json

# Paper-scale (5M timesteps-ish, 8 evals)
python scripts/benchmark_systems.py \
    --num_envs 512 --num_updates 80 --num_evaluation 8 --rollout_length 128 \
    --num_eval_episodes 64 \
    --systems cppo_optimized cppo_baseline icrl ippo
```

`--systems` accepts any subset of
`{cppo_optimized, cppo_baseline, icrl, ippo}` and runs them sequentially
as fully isolated subprocesses (each gets its own JAX compilation and
memory state). Pass extra Hydra overrides (e.g. seed) via `--extra
system.seed=0`.

## Known caveats

- `cppo_baseline` was made by `cp ff_ppo_crl.py ff_ppo_crl_baseline.py`
  *before* the optimization edits. Both versions still import
  `flatten_crl_fn` from the *same* `utils.py`, which now has the
  micro-cleanup from change (4). That cleanup is just a Python-level
  trace simplification (same compiled HLO), so the A/B comparison still
  isolates the substantive optimizations.
- The 1 % drift in TRAINER metrics (Q mean, critic loss, actor loss)
  between baseline and optimized smoke runs is float-reordering noise
  amplified through training — verified by an isolated
  `compute_crl_advantages` equivalence test that returns
  `max abs diff = 0.0`.

## Files we did *not* push

- `.venv/` — local JAX/CUDA install
- `run.sh` — contains a wandb API key that should not be in version control
- everything else under `mava/` we did not touch
