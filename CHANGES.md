# Code-level changelog — v3 (relative to original `ff_ppo_crl.py`)

Read `HANDOFF.md` first for context. This file is a more granular pointer
into what the diff actually does, mapped to source lines.

## `mava/systems/icrl/anakin/ff_ppo_crl.py`

### New config knobs (all read at `learner_setup` / `get_learner_fn` time)

| field | type | default | meaning |
|---|---|---|---|
| `system.actor_batch_size` | int | `system.batch_size` | actor minibatch size; can be much larger than critic since no `[B,B]` matrix |
| `system.actor_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs over actor data; supports IPPO-style "few epochs over huge minibatches" |
| `system.critic_ppo_epochs` | int | `system.ppo_epochs` | PPO epochs over critic data; usually 1 is fine |
| `system.critic_subsample_fraction` | float in (0, 1] | 1.0 | each critic epoch picks `frac * N` random transitions; cuts critic SGD iters proportionally |
| `system.scan_unroll` | int | 8 | `lax.scan(unroll=...)` for both actor and critic minibatch loops |
| `system.use_gae` | bool | false | enable GAE-bootstrapped advantages over the contrastive V; default off — single-step `Q(s,a) - V(s)` is the original behavior |
| `system.gae_lambda` | float | 0.95 | only used when `use_gae=true` |

### Structural changes

1. **`compute_crl_advantages`** — original single-step advantage. Replaced
   per-action `vmap` with a single batched SA-encoder call on
   `[N, action_dim, ...]` broadcast tensors. Bit-identical numerics
   (`max abs diff = 0.0`).

2. **`compute_crl_gae_advantages`** (new). Operates on the pre-HER
   `(T, E, ...)` traj_batch, conditioned on the env's real
   `ultimate_goal`:
   ```
   V_t        = sum(softmax(actor_logits) * Q_values, axis=-1)   # [T, E]
   delta_t    = gamma * V_{t+1} * (1 - done_t) - V_t              # no reward
   GAE_t      = delta_t + gamma * lambda * (1 - done_t) * GAE_{t+1}
   advantages = (GAE - GAE.mean()) / (GAE.std() + 1e-8)
   ```
   Reverse-time `lax.scan`, output `(T, E)` reshaped C-order to `(T*E,)`
   to match the transition flattening order downstream.

3. **HER vmap order**: `vmap(flatten_crl_fn, in_axes=(None, 1, 0), out_axes=1)`
   directly on the `(T, E, ...)` traj_batch (no `swapaxes` copy). C-order
   reshape on the output gives the same flat ordering as the previous
   `swapaxes` + Fortran-order reshape.

4. **`extras` strip after advantage compute**: only `real_ultimate_goal`
   and `advantages` are kept. The 4-5 unused fields (state, future_state,
   future_action, done, state_extras, policy_extras) are dropped before
   entering the PPO epoch loop so they don't get gathered every minibatch.

5. **Two separate scans per PPO epoch** — `_actor_epoch` and
   `_critic_epoch`. Mathematically equivalent to the old interleaved
   actor+critic-per-minibatch loop because actor and critic params are
   disjoint and advantages are frozen pre-loop.

6. **Per-side `total_grad_steps`** in `learner_setup` — actor and critic
   schedules see their own `num_updates × ppo_epochs × minibatches` count.

7. **SPS logging** — adds `compile_seconds` (logged on iter 0 only) and
   `wall_clock_seconds` (every iter) to the ACTOR log line so a benchmark
   can isolate JIT compile from steady-state SPS.

## `mava/systems/icrl/utils.py`

`flatten_crl_fn` micro-cleanup: replaced
```python
single_trajectories = jnp.concatenate(
    [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
)
seed_mask = jnp.equal(single_trajectories, single_trajectories.T)
```
with
```python
seeds = transition.extras["state_extras"]["seed"]
seed_mask = jnp.equal(seeds[None, :], seeds[:, None])
```
Same behaviour, no Python-level `O(T)` list construction in the JIT trace.

## `scripts/benchmark_systems.py`

New file. Subprocess-isolated benchmark for CPPO / ICRL / IPPO. Highlights:
- Aliases (`cppo_optimized`, `cppo_decoupled`, `cppo_ippo_actor`,
  `cppo_sub50`, `cppo_sub25`) all point at `ff_ppo_crl` so a single
  benchmark run can A/B different config overrides.
- `--per-system` flag for per-alias Hydra overrides.
- Drops the first **two** evals from steady-state SPS (compile + warmup).
- Polls `nvidia-smi` for peak memory (system-wide; not isolatable per run
  on a shared GPU).
- Captures `eval_win_rates` per eval from `EVALUATOR` lines.

## `scripts/summarize_run.py`

New file. Standalone log analyzer used to post-hoc summarize the long
80M-timestep production runs. Extracts per-eval SPS, training-rollout win
rate, eval-policy win rate, compile time, total wall.

## `mava/systems/icrl/anakin/ff_ppo_crl_baseline.py`

Pristine copy of the **pre-optimization** `ff_ppo_crl.py`. Used by the
benchmark script as `cppo_baseline`. Don't edit this file — it's the
reference for "did the optimization actually speed things up?". Note:
both files import `flatten_crl_fn` from the **same** modified `utils.py`,
but the modification there is purely a Python-trace simplification (same
compiled HLO), so the A/B comparison still isolates the substantive
optimizations.
