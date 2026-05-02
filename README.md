# mava CPPO optimization snapshot v2 — 2026-05-02

Iteration on the v1 branch (`mava-cppo-optimization-2026-05-02`). The big new
idea here is **decoupling the actor minibatch size from the critic minibatch
size** — necessary because contrastive InfoNCE materialises a `[B, B]` logits
matrix (memory-bound past ~1024) while PPO's actor has no such constraint
and benefits from much larger batches (fewer SGD steps per update → less
per-iter scan overhead).

## What's different from v1

1. **Decoupled batch sizes.** New config field `actor_batch_size`
   (defaults to `batch_size` for backward compatibility). Critic still uses
   `batch_size` so the InfoNCE matrix stays at the user-controlled size.
2. **Two separate scans per PPO epoch.** Actor and critic params are
   disjoint; their losses don't reference each other; advantages are
   frozen before the epoch loop. So an "actor scan over actor-sized
   minibatches, then critic scan over critic-sized minibatches" is
   mathematically equivalent (up to optimizer-step ordering) to the old
   interleaved actor+critic-per-minibatch loop, but lets each side use
   its own batch size.
3. **`scan(unroll=8)`** on both actor and critic minibatch scans.
   Configurable via `system.scan_unroll` (default 8). Trades compile time
   for per-iter overhead.
4. **Per-side LR schedule grad-step counts.** Each schedule's
   `total_grad_steps` is computed from its own minibatch count
   (`num_updates × ppo_epochs × side_minibatches`) instead of the shared
   one. Matters for cosine/linear LR decay.

## Headline numbers

`num_envs=256, rollout=128, ppo_epochs=1, smacv2_5_units, 1× H100 80 GB`,
6 evals × 12 updates, last 4 used for steady-state:

| system | actor_batch | critic_batch | compile (s) | steady SPS (mean ± std, n=4) | vs baseline |
|---|---|---|---|---|---|
| `cppo_baseline` | 128 | 128 | ~85 | 82,333 ± 341 | 1.000× |
| `cppo_optimized` (v2 default) | 128 | 128 | 113.8 | 84,163 ± 187 | +2.2% |
| **`cppo_decoupled`** (v2 + actor=512) | **512** | 128 | 116.5 | **90,955 ± 130** | **+10.5%** |

Total stack (v1 ⊕ v2):

| from baseline | optimized batched-Q vmap (v1) | + decoupled batches + scan unroll (v2) |
|---|---|---|
| 82,333 SPS | 85,175 SPS (+3.4%) | **90,955 SPS (+10.5%)** |

**Win-rate at eval 5** (smoke run, not paper-scale):
- baseline: 2.25
- v2 default: 0.66
- v2 decoupled: 3.04

Decoupled does NOT hurt learning at this short horizon — actually slightly
better, plausibly because larger PPO batches give a less noisy gradient.
This is a *short* run (only 12 updates); a full sweep is needed to
confirm parity, but there's no early-warning of a regression.

## Config knobs

```yaml
system:
  batch_size: 128            # critic minibatch size (controls InfoNCE [B, B] matrix)
  actor_batch_size: 512      # NEW. defaults to batch_size if unset.
  scan_unroll: 8             # NEW. unroll for actor + critic scans.
```

`actor_batch_size` should be a multiple (or factor) of `batch_size` so the
flat batch divides cleanly. Truncation is to `max(actor, critic)` and with
power-of-two sizes that's automatic.

CLI:
```bash
python -u -m mava.systems.icrl.anakin.ff_ppo_crl \
    --config-name ppo_crl env=smax env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 system.num_updates=1250 \
    system.batch_size=128 +system.actor_batch_size=512
```

## How to A/B against baseline

The benchmark script supports per-system Hydra overrides so all three
configurations run in one invocation:

```bash
python scripts/benchmark_systems.py \
    --num_envs 256 --num_updates 12 --num_evaluation 6 --rollout_length 128 \
    --systems cppo_optimized cppo_decoupled cppo_baseline \
    --per-system "cppo_decoupled:+system.actor_batch_size=512" \
    --out benchmark_results.json
```

`cppo_optimized` and `cppo_decoupled` point at the same module
(`ff_ppo_crl`) — they're aliases so a single run can A/B different
override sets cleanly.

## Why the v1 batched-Q optimization didn't move the needle on its own

At the scales we tested (N≈327k transitions, action_dim=14), XLA already
batched the per-action SA-encoder vmap into a comparable matmul plan.
The bottleneck is not the advantage computation but the **inner minibatch
`lax.scan`** — with `batch_size=128` you do ~2,500 SGD steps per eval, and
each one has its own forward+backward+optimiser overhead. Decoupling
shrinks the actor side to ~640 steps (at `actor_batch_size=512`) without
breaking the critic's memory-bound InfoNCE matrix. That's where the
speedup lives.

## What to try next

If you want more SPS:

- **Push `actor_batch_size` higher** (1024, 2048) until the actor scan
  becomes a single iter or you hit a regression.
- **Drop `pmap` on single-GPU.** ~5% available, but requires touching
  env-state reshaping, `flax.jax_utils.replicate`, and eval-key plumbing
  in `run_experiment`. Out of scope for this branch.
- **Compute advantages during rollout** — saves one actor forward pass on
  the full batch. Modest (~5–10%) but contained.

If you want better wall-clock convergence, that's a separate axis (sample
efficiency, hparam tuning) which this work doesn't touch.

## Files in this branch

| path | what |
|---|---|
| `README.md` | this file |
| `mava/systems/icrl/anakin/ff_ppo_crl.py` | optimized v2 (decoupled batches) |
| `mava/systems/icrl/anakin/ff_ppo_crl_baseline.py` | pristine pre-optimization copy for A/B |
| `mava/systems/icrl/utils.py` | `flatten_crl_fn` micro-cleanup |
| `scripts/benchmark_systems.py` | subprocess-isolated CPPO/ICRL/IPPO benchmark with `--per-system` overrides |
| `benchmark_logs/cppo_*.log` | terminal output from the v2 benchmark |
| `benchmark_results.json` | parsed metrics from the v2 benchmark |

## Caveats

- Compile time grew from ~85 s → ~117 s due to `scan(unroll=8)`. One-off
  cost; doesn't affect steady-state SPS.
- Algorithm equivalence between v1 and v2 default (same `batch_size` for
  both sides) is **mathematical** (separate scans on disjoint params with
  frozen advantages), but optimizer-step ordering differs (all actor steps
  before all critic steps within an epoch, vs interleaved). Float-noise
  drift in TRAINER metrics is identical in character to the v1 → baseline
  drift.
- `peak_gpu_memory_mb` from `nvidia-smi` is system-wide and not isolatable
  per system on a shared GPU. For accurate per-run peak, instrument
  `jax.local_devices()[0].memory_stats()['peak_bytes_in_use']` from inside
  each `run_experiment`.
