# Code changelog — v4 (delta from v3)

Two changes on top of the v3 branch.

## 1. Working GAE: new `gae_kind` config + corrected default

### v3 (broken)

`use_gae=true` always used the per-step
```
δ_t = γ V_{t+1} (1 − done_t) − V_t      (no reward, value-difference TD)
```
which fails at init: V is a random network, V at successive states is
~unrelated random numbers, their difference is noise.

### v4

New config `system.gae_kind` (default `"smooth_adv"`). The function
`compute_crl_gae_advantages` now branches on this:

- **`smooth_adv`** (default, working):
  ```
  δ_t   = Q(s_t, a_t) − V(s_t)              (single-step contrastive adv)
  GAE_t = δ_t + γ λ (1 − done_t) GAE_{t+1}   (smooth across time)
  ```
  Q and V are evaluated on the same state through the same network, so
  the network's randomness cancels in the subtraction. λ=0 recovers the
  non-GAE single-step advantage exactly.

- **`q_as_reward`**:
  ```
  δ_t = Q_taken_t + γ V_{t+1} (1 − done_t) − V_t
  ```
  Treats Q as a per-step "reward" with V as bootstrap. Closer to the
  standard PPO formula but V isn't trained by Bellman regression in CRL,
  so semantically iffy. Kept for ablation.

- **`value_diff`** (the v3 default, kept for ablation):
  ```
  δ_t = γ V_{t+1} (1 − done_t) − V_t
  ```
  Empirically broken at init, as documented.

### File: `mava/systems/icrl/anakin/ff_ppo_crl.py`

- Added `gae_kind = str(config.system.get("gae_kind", "smooth_adv"))` in
  `get_learner_fn` config-read block.
- `compute_crl_gae_advantages`: renamed inner GAE recurrence to a
  conditional that picks one of three `_gae_step` implementations based
  on `gae_kind`. Same outer compute (Q, V, q_taken, done mask), same
  output (mean=0, std=1 normalized advantages, stop_gradient).

## 2. OOM fix: flatten before network forwards

### Problem

Original v3 implementation used 4D `(T, E, A, S)` shapes through the
SA-encoder forward:
```python
obs_tiled = jnp.broadcast_to(
    obs_achieved[..., None, :], (T_, E_, action_dim, S_)
)  # [T, E, A, S]
sa_repr = sa_encoder_apply_fn(params.sa_encoder, obs_tiled, actions_tiled)
```
At `num_envs=512, rollout=128, action_dim=14, S≈100`, total elements per
network input is the same as the 3D `(N=T·E, A, S)` shape used by the
non-GAE path — but XLA picks a different memory plan for 4D, and OOMs at
~6.7 GB intermediate Dense layer activations.

### Fix

Flatten to 2D / 3D before the SA-encoder, run all network forwards on the
flat shapes (matching what `compute_crl_advantages` already does), then
reshape outputs back to `(T, E)` only for the reverse-time GAE scan:

```python
T_, E_ = traj_batch.observation.shape[0], traj_batch.observation.shape[1]
obs_achieved_flat = jnp.concatenate(
    [traj_batch.observation, traj_batch.achieved_goal], axis=-1
).reshape(T_ * E_, -1)                    # [N, S]
ultimate_flat = traj_batch.ultimate_goal.reshape(T_ * E_, -1)
# ... network forwards on [N, A, S] / [N, ...] ...
values = values_flat.reshape(T_, E_)      # back to (T, E) for the scan
q_taken = q_taken_flat.reshape(T_, E_)
```

Same total compute, same memory pattern as non-GAE — fits at 80M scale.

## Other notes

- No public API changes: all v3 config flags still work as before.
- `gae_kind` default is `smooth_adv`; existing `use_gae=true` invocations
  silently get the working version. The broken default in v3 (`value_diff`)
  is now opt-in only.
- `compute_crl_advantages` (the non-GAE single-step path) is untouched.
  Default behaviour with `use_gae=false` (or unset) is identical to v3.
