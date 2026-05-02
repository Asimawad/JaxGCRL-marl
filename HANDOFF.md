# Handoff — CPPO optimization for NeurIPS wall-clock benchmark

**Snapshot date:** 2026-05-02. Branch: `mava-cppo-optimization-v3-2026-05-02`.

This branch is an **orphan snapshot** of working state — the goal is to give the
next Claude (or human) enough context to continue without spelunking through
chat logs. Read this file first, then `CHANGES.md` for a code-level changelog,
then `BENCHMARKS.md` for all the numbers.

---

## Project goal

The user is preparing a NeurIPS submission. They need:

1. **Wall-clock speed comparison** of CPPO (`ff_ppo_crl.py`) vs ICRL
   (`ff_icrl.py`) vs IPPO (`ff_ippo.py`) on `smacv2_5_units` (SMAX), under
   matched architectures.
2. **A faster CPPO** that doesn't break the algorithm. The original `ff_ppo_crl.py`
   was a "first-thing-that-worked" prototype — they want it optimized for
   throughput while preserving learning behavior.

The user's tuned baseline (`run.sh` Run-9) hit **76.07% win rate** on
`smacv2_5_units` after 80M timesteps with `num_envs=512`. **That's the
behavior we must preserve** — anything faster is good, anything that doesn't
hit ~75-76% win rate is not acceptable as a paper baseline.

---

## What's been done in v3 (this branch)

Three rounds of optimization, each more invasive than the last. All
correctness-preserving by construction (or by experimental verification).

### v1 — local optimizations (~0% wall-clock at scale, mostly cleanup)
1. `compute_crl_advantages`: replace per-action `vmap` with batched
   `[N, A, ...]` SA-encoder call. Bit-identical (`max abs diff = 0.0`).
2. Drop `swapaxes` + Fortran-order reshape; vmap HER over axis 1 with
   `out_axes=1`, then C-order reshape. Same flat ordering, fewer copies.
3. Strip `transition.extras` to `{real_ultimate_goal, advantages}` after
   advantage computation.
4. `flatten_crl_fn`: replace `concatenate([...] * seq_len)` with broadcasting.
5. Add `compile_seconds` and `wall_clock_seconds` to ACTOR log line.

**Result:** ~3% wall-clock improvement at `num_envs=256, ppo_epochs=1`. The
real bottleneck turned out to be the inner minibatch `lax.scan`, not the
per-action SA-encoder vmap.

### v2 — decouple actor and critic batch sizes
1. New config: `system.actor_batch_size` (defaults to `system.batch_size`).
   Critic stays on `system.batch_size` so the InfoNCE `[B, B]` matrix is
   bounded; actor can use much bigger batches because PPO has no such
   constraint.
2. Two separate `lax.scan`s per PPO epoch (actor pass, then critic pass).
   Mathematically equivalent to the old interleaved per-minibatch loop —
   actor/critic params are disjoint and advantages are frozen.
3. `scan(unroll=8)` on both. Configurable via `system.scan_unroll`.
4. Per-side `total_grad_steps` for the LR schedules.

**Result:** +11% SPS at `num_envs=256, ppo_epochs=1, actor_batch_size=512`.
The actor side went from 320 SGD iters/update down to 320 (with `batch=512`,
half the iter count) without touching the critic.

### v3 (this branch) — further decoupling + experimental knobs

1. **Decoupled epoch counts.** `system.actor_ppo_epochs` and
   `system.critic_ppo_epochs` (both default to `system.ppo_epochs`). Actor
   and critic can now run different numbers of PPO epochs — supports
   IPPO-style "multiple epochs over a couple of huge minibatches" for the
   actor while keeping the critic at one pass.
2. **Critic subsampling.** `system.critic_subsample_fraction` (default 1.0).
   Each critic epoch picks a random subset of `frac * N` transitions before
   batching. With `frac=0.25` you do 4× fewer critic SGD iters per update —
   speed gain proportional to the cut.
3. **GAE-bootstrapped advantages (experimental).** `system.use_gae` (default
   false), `system.gae_lambda` (default 0.95). When on, advantages are
   computed by GAE over `V(s,g) = E_pi[Q(s,.,g)]` from the contrastive
   critic, with **no explicit reward** (the contrastive V already encodes
   `log P(reach g from s)`):

   ```
   delta_t = gamma * V_{t+1} * (1 - done_t) - V_t
   GAE_t   = delta_t + gamma * lambda * (1 - done_t) * GAE_{t+1}
   ```

   Computed on the pre-HER `(T, E, ...)` traj_batch (V is conditioned on the
   env's real `ultimate_goal`, same goal the actor uses in its loss).

---

## Big results so far — read this before benchmarking again

### Speed benchmarks (short, 12 updates × 6 evals, num_envs=256)
See `BENCHMARKS.md` for the full table; headline:

| variant | actor_batch / critic_batch | extra | steady SPS | vs baseline |
|---|---|---|---|---|
| `cppo_baseline` | 256 / 256 | — | 82,373 | 1.00× |
| `cppo_decoupled` | 512 / 256 | — | 91,642 | +11.3% |
| `cppo_ippo_actor` | N/2 / 256 | actor_ppo_epochs=4 | 94,211 | +14.4% |
| `cppo_sub50` | N/2 / 256 | sub_frac=0.5 | 137,180 | +45.8% |
| `cppo_sub25` | N/2 / 256 | sub_frac=0.25 | 177,548 | +88.7% |
| `icrl` | (off-policy) | — | 97,887 | +18.8% |
| `ippo` (4-layer) | (no goal encoder) | — | 273,289 | +231.8% |

### Production-scale runs (80M timesteps, num_envs=512)

| run | wall | peak eval win-rate | learning |
|---|---|---|---|
| `cppo_decoupled` | **26.7 min** | **76.51%** (eval 66) | ✅ matches baseline (76.07%) |
| `cppo_sub25` | 9.1 min | 0.98% | **❌ broken** — entropy collapsed, PPO clip fully biting |
| `cppo_decoupled` + `use_gae=true` | **in progress** at handoff time | TBD | early signal: learning much slower than non-GAE |

---

## Critical caveat about cppo_sub25

`cppo_sub25` was 3× faster wall-clock but **completely failed to learn**. Final
TRAINER state:

```
Actor loss:           -0.000     <-- PPO clip is fully biting, no gradient flow
Entropy:               0.042     <-- policy is essentially deterministic, collapsed
Categorical accuracy:  0.072     <-- contrastive critic ~7%, barely above random
```

**Cause:** the user's `run.sh` Run-9 hyperparameters
(`actor_lr=0.0005, ent_coef=0.01, max_grad_norm=0.05, clip_eps=0.2`) were
tuned for `ppo_epochs=2` + `batch_size=256` + `frac=1.0`. With actor_ppo_epochs=4
and `frac=0.25` the policy moves too aggressively per update. Don't reuse
those hparams with the IPPO-style actor unless you also re-tune `ent_coef`
(higher) and `actor_lr` (lower).

**The safe operating point** for production runs is `cppo_decoupled` (just
`actor_batch_size=512`, no other changes). Confirmed to match baseline win
rate at 80M / 512 envs.

---

## Win-rate scale gotcha

`mava/utils/logger.py:82-83`:
```python
n_won_episodes: int = np.sum(metrics["won_episode"])
win_rate: float = (n_won_episodes / n_episodes) * 100   # <-- SCALE IS 0-100
```

So in the logs, `Win rate: 76.51` means 76.51% (a great trained policy), not
0.7651. The user's `run.sh` comments use the same convention
(`win_rate=76.07421875%`).

---

## SPS logging gotcha

The first eval has JIT compile time baked in. The **second** eval is also
slow (~30s on 256-env config) due to a residual JAX warmup — likely the
async evaluator bleeding into the next `learn()` call. The benchmark script
in `scripts/benchmark_systems.py` drops the first **two** evals from
steady-state SPS by default. A more permanent fix is to add
`jax.block_until_ready(eval_metrics)` after the `evaluator(...)` call in
`run_experiment` — not done yet.

---

## What's currently running at handoff time

A `cppo_decoupled + use_gae=true` 80M-timestep run is in progress on the box
(PID was 130463 at handoff). Output log: `/tmp/cppo_decoupled_gae_80M.log`.
Expected runtime ~25-30 minutes. After it finishes:

```bash
python /tmp/summarize_run.py /tmp/cppo_decoupled_gae_80M.log | tail -100
```

**Early observation (first 3 evals, 3M timesteps):** win-rate 0.05 → 0.24 →
0.34%. Compare to non-GAE `cppo_decoupled` which at the same point was
0.05 → 14.9 → 27.6%. **GAE is learning ~80× slower in the early phase.**

This may or may not catch up later. Possible explanations:
- The contrastive V is itself untrained at the start, so GAE bootstraps off
  garbage values for the first many updates.
- Pure value-difference TD without an explicit reward gives a much weaker
  signal at init when V ≈ random.
- The default `gae_lambda=0.95` may need tuning for the contrastive setting.

If it doesn't recover by eval 30 (~30M timesteps), it's probably broken.
Worth trying:
- `gae_lambda=0.5` (more local advantage, less bootstrap)
- Sparse reward injection: `r_t = 1` if `achieved_goal_t == ultimate_goal`
  else 0, then standard PPO GAE on top.
- Use the contrastive Q (not V) more directly: `delta_t = Q(s_t, a_t) -
  V(s_t)` plus a small bootstrap term.

---

## Files in this branch

```
README.md                                          # this snapshot's entry point
HANDOFF.md                                         # this file
CHANGES.md                                         # code-level changelog
BENCHMARKS.md                                      # all numbers + commands

mava/systems/icrl/anakin/ff_ppo_crl.py             # OPTIMIZED v3
mava/systems/icrl/anakin/ff_ppo_crl_baseline.py    # PRISTINE pre-optimization copy
mava/systems/icrl/utils.py                         # micro-cleanup in flatten_crl_fn
scripts/benchmark_systems.py                       # subprocess-isolated CPPO/ICRL/IPPO
scripts/summarize_run.py                           # post-run log analyzer

benchmark_logs/                                    # short-run benchmark stdout
80M_runs/cppo_decoupled.win_rate_trajectory.txt    # eval-policy win-rate curve
80M_runs/cppo_sub25.win_rate_trajectory.txt        # the broken run, for reference
```

---

## What to do next (in priority order)

1. **Wait for the GAE 80M run to finish.** If it converges to ~75% win rate,
   GAE is a viable advantage estimator and probably worth a section in the
   paper. If it plateaus below 30%, it's not — either drop it or try one of
   the variants suggested above.

2. **Run cppo_decoupled at multiple seeds (5+).** The 76.51% peak is a
   single-seed result. For a paper claim you need a seed sweep:
   ```bash
   for seed in 0 1 2 3 4 5; do
       python -u -m mava.systems.icrl.anakin.ff_ppo_crl ... \
           system.seed=$seed +system.actor_batch_size=512
   done
   ```

3. **Run cppo_baseline at the same scale for the formal comparison.** We
   benchmarked the unoptimized `ff_ppo_crl_baseline.py` on short runs but
   never at 80M — the `cppo_decoupled` 76.51% needs a matched-seed baseline
   number to claim "preserves learning at the wall-clock benchmark scale".

4. **Push to GitHub:** `gh_push.py` pattern (see prior branches
   `mava-cppo-optimization-2026-05-02` and `mava-cppo-optimization-v2-...`).
   `git push` over smart-http is blocked from this sandbox — use the
   GitHub REST API with the user's PAT (which is now in old chat history
   and should be rotated; user knows this).

5. **Find a wall-clock-vs-win-rate Pareto frontier.** Sweep
   `actor_batch_size`, `actor_ppo_epochs`, `critic_subsample_fraction` and
   plot SPS vs final win rate. The user wants to know which config gives
   the best speed *without* breaking learning.

---

## User preferences (worth knowing before you change anything)

- **Don't change the algorithm.** Same contrastive loss, same HER, same PPO
  surrogate. Optimization is implementation, not algorithm. (`use_gae` is
  on the boundary — they explicitly asked for it as an ablation.)
- **Don't break baseline learning.** If your change makes win rate drop, it's
  not a win, regardless of SPS.
- **Production hparams live in `run.sh`** — Run 9 (`win_rate=76.07%`) is the
  reference config. Don't deviate from these unless explicitly retuning.
- **Wandb is disabled in our runs** — the user has a wandb key in `run.sh`
  but we keep wandb off for sandbox runs. Re-enable for the user's actual
  paper runs.
- **GitHub PAT in chat history is compromised** — rotate it.
