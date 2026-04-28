# Goal Routing in Mava's PPO-CRL: Discrete vs Continuous

**Subject:** Comparison of two Mava PPO-CRL implementations to determine which
correctly routes the *real* env goal vs the *HER-relabeled* goal through the
training loop.

**TL;DR.** The discrete file [`example.py`](example.py) (Mava `ppo_crl.py`)
routes the goals correctly: the **HER-relabeled** goal is used **only** by
the contrastive critic; the **real env goal** is used by the rollout, the
advantage computation, and the PPO actor loss. The continuous file
[`mava/systems/icrl/anakin/ppo_crl_continuous.py`](mava/systems/icrl/anakin/ppo_crl_continuous.py)
gets this wrong when `use_gae=False`: the rollout records `log_prob` under the
**real** goal, but the actor and the advantage are computed under the
**relabeled** goal. This makes the PPO importance ratio compare two probabilities
conditioned on different inputs, which is mathematically meaningless and
empirically causes late-training collapse on harder envs.

---

## 1. The shared algorithm

Both files implement the same high-level loop:

1. **Rollout** with the current actor: `actor_input = [obs, achieved_goal, env_goal]`,
   record `(action, log_prob)`.
2. **HER relabel**: per-trajectory, replace the goal of each transition with a
   future-achieved goal sampled from the same trajectory.
3. **Critic loss**: contrastive InfoNCE between `phi(s, a)` and `psi(g)`, where
   `g` is the relabeled goal — this is the CRL signal.
4. **Advantage**: `A = Q(s, a, g) − E_{a'~pi}[Q(s, a', g)]` (MC) or via GAE.
5. **Actor loss**: PPO clipped objective using the advantage and `ratio = exp(new_log_prob − old_log_prob)`.

The question is: at steps 4 and 5, **which goal** — the relabeled HER goal or
the original env goal — does the actor input use, and which does the
goal-encoder receive?

---

## 2. The right answer (a priori)

For PPO to be valid, the conditional distribution under which `old_log_prob`
was recorded must equal the one under which `new_log_prob` is computed. Since
the rollout records `log_prob` conditioned on the **real** env goal, the actor
loss must condition `new_log_prob` on the same real goal. Otherwise the ratio
`exp(new − old)` is comparing `pi(a | s, goal_A)` against `pi(a | s, goal_B)`
— two different distributions — and the PPO clip mechanism cannot keep the
update conservative because there is no meaningful "old policy" to stay close
to.

The CRL critic is different: its job is to learn an embedding such that
`phi(s, a) · psi(g)` is high for goals reachable from `(s, a)`. HER relabeling
provides positive examples of "goal that was reached from this `(s, a)`," which
is exactly the contrastive signal the critic needs.

**Therefore the correct routing is:**

| consumer | goal input |
|---|---|
| rollout actor | real env goal |
| critic loss (InfoNCE) | relabeled goal |
| advantage Q and V | real env goal |
| PPO actor loss | real env goal |

---

## 3. What the discrete file does

[`example.py`](example.py) (Mava's `ppo_crl.py` for discrete actions, e.g. RWARE).

After `flatten_crl_fn`, the trajectory has two goal fields:

| field | meaning |
|---|---|
| `transitions.ultimate_goal` | **relabeled** (HER) goal |
| `transitions.extras["real_ultimate_goal"]` | **original env** goal, saved before relabeling |

Usage:

**Rollout** ([example.py:96](example.py#L96)):
```python
actor_input = jnp.concatenate([agents_view, achieved_goal, ultimate_goal], axis=-1)
# ultimate_goal here is the real env goal (relabel hasn't happened yet)
```

**Critic loss** ([example.py:285](example.py#L285)):
```python
goal = ultimate                 # = batch.ultimate_goal = RELABELED
g_repr = g_encoder_apply_fn(critic_params["goal_encoder"], goal)
```

**Advantage** ([example.py:187](example.py#L187)):
```python
real_ultimate_goal = transitions.extras["real_ultimate_goal"]
g_repr = g_encoder_apply_fn(params.goal_encoder, real_ultimate_goal)
actor_input = jnp.concatenate([..., real_ultimate_goal], axis=-1)
```

**Actor loss** ([example.py:233](example.py#L233)):
```python
actor_input = jnp.concatenate([..., batch.extras["real_ultimate_goal"]], axis=-1)
ratio = jnp.exp(new_log_prob - batch.log_prob)
# both probabilities are conditioned on real_ultimate_goal → ratio is meaningful
```

**Verdict:** correct. Real env goal everywhere except the critic's
contrastive loss.

---

## 4. What the continuous file does

[`mava/systems/icrl/anakin/ppo_crl_continuous.py`](mava/systems/icrl/anakin/ppo_crl_continuous.py).

After `_flatten_crl` ([line 260](mava/systems/icrl/anakin/ppo_crl_continuous.py#L260)),
the relabeled dict has:

```python
result = {
    "observation": ...,
    "goal": goal,                                  # RELABELED (HER)
    "action": ..., "x_t": ..., "old_log_prob": ...,
    "reward": ...,
    "current_achieved_goal": ...,
}
```

**The real env goal is not saved here.** It is added later only if `use_gae=True`
([lines 400–412](mava/systems/icrl/anakin/ppo_crl_continuous.py#L400-L412)) under the key `env_goal`.

### 4.1. The `use_gae=False` branch (the pure-CRL setting)

**Advantage** ([lines 432–451](mava/systems/icrl/anakin/ppo_crl_continuous.py#L432-L451)):
```python
goal = batch["goal"]                                          # RELABELED
actor_input = jnp.concatenate([obs, current_ag, goal], ...)   # RELABELED
g_repr = goal_encoder_apply(params.goal_encoder, goal)        # RELABELED
q_taken = energy_fn(..., sa_repr_taken, g_repr)
```

**Actor loss** ([lines 564–583](mava/systems/icrl/anakin/ppo_crl_continuous.py#L564-L583)):
```python
if use_gae:
    actor_goal = batch["env_goal"]   # real
else:
    actor_goal = goal                # RELABELED  ← bug branch
...
actor_input = jnp.concatenate([obs, current_ag, actor_goal], axis=-1)
new_log_prob = ...                   # conditioned on RELABELED
ratio = jnp.exp(new_log_prob - old_log_prob)   # but old_log_prob was logged under REAL
```

This is the bug: `old_log_prob` was recorded during rollout under
`actor_input = [obs, current_ag, env_state.goal]`
([line 175](mava/systems/icrl/anakin/ppo_crl_continuous.py#L175)) — the **real**
env goal — while `new_log_prob` here is computed under the **relabeled** goal.
The PPO ratio is comparing two unrelated conditional distributions.

### 4.2. The `use_gae=True` branch

When GAE is enabled, `actor_goal = batch["env_goal"]` (real), so the actor
ratio is consistent. But this branch uses **env-reward GAE advantages**
([lines 414–419](mava/systems/icrl/anakin/ppo_crl_continuous.py#L414-L419)) instead of CRL
Q–V advantages, and adds a value network trained on env-reward returns
([lines 615–632](mava/systems/icrl/anakin/ppo_crl_continuous.py#L615-L632)). At that point the
algorithm is no longer pure CRL — env reward is back in the loop.

**Verdict on the continuous file:** the only "correct" actor branch
(`use_gae=True`) is correct *because* it stops being pure-CRL. The
pure-CRL branch (`use_gae=False`) has the goal-routing bug.

---

## 5. Side-by-side table

| consumer | discrete file | continuous file (`use_gae=False`) |
|---|---|---|
| rollout actor input | real | real |
| critic InfoNCE goal | relabeled ✓ | relabeled ✓ |
| advantage Q goal | real ✓ | **relabeled ✗** |
| advantage actor input | real ✓ | **relabeled ✗** |
| actor-loss input | real ✓ | **relabeled ✗** |
| PPO ratio is meaningful | yes ✓ | **no ✗** |

---

## 6. Why it matters

PPO's clip mechanism is built on the assumption that `old_log_prob` and
`new_log_prob` are samples from the same conditional distribution
`pi(· | s, g)`, just at different timesteps of the policy parameters. When the
inputs differ (real vs relabeled goal), the ratio carries no information about
how much the *policy* moved — it mostly carries information about how much the
policy responds to the *goal channel*. Empirically:

- **Early training**: the policy barely uses the goal, so
  `pi(a | s, g_real) ≈ pi(a | s, g_HER)`, the ratio is close to 1, and PPO
  appears to work.
- **Late training**: the policy has learned to condition strongly on the goal,
  the two distributions diverge, the ratio explodes or collapses, the clip
  truncates real gradient signal, and learning destabilises.

This is exactly the failure pattern observed for our CPPO port on the harder
JaxGCRL envs (ant, ant_u_maze, humanoid): peak performance ~14–25M steps,
followed by catastrophic collapse to near-zero success rate. The issue is
absent on small-action envs (reacher, simple_u_maze) where the policy never
learns a strongly goal-conditioned mapping in the first place.

---

## 7. The minimal correct fix for the continuous file

Three changes inside `_update_step`:

1. In `_flatten_crl`, also save the original env goal alongside the relabeled
   one (mirroring `extras["real_ultimate_goal"]` from the discrete file).
2. In `_compute_advantages`, replace `goal = batch["goal"]` with the saved real
   env goal for both the actor MC sampling input and the goal encoder.
3. In `_actor_loss_fn`, unconditionally use the real env goal as the actor
   input (regardless of `use_gae`).

The critic loss (`_critic_loss_fn`) keeps `goal = batch["goal"]` unchanged.

---

## 8. Status of our CPPO port

Our [`jaxgcrl/agents/cppo/cppo.py`](jaxgcrl/agents/cppo/cppo.py) was a faithful
port of the continuous file and inherited its bug. We have applied the fix
described in §7 directly:

- [`her_relabel`](jaxgcrl/agents/cppo/cppo.py#L376) already saved the real env
  goal under the key `target_goal`.
- [`actor_loss_fn`](jaxgcrl/agents/cppo/cppo.py#L420) now reads
  `goal = batch["target_goal"]`.
- [`compute_advantage`](jaxgcrl/agents/cppo/cppo.py#L449) now reads
  `goal = batch["target_goal"]`.
- The critic ([line 537](jaxgcrl/agents/cppo/cppo.py#L537)) continues to use
  `batch["relabeled_goal"]`, so the CRL contrastive signal is unchanged.

This makes our CPPO match the **discrete reference's correct routing pattern**
while remaining pure-CRL (no env reward, no GAE, no value head). It is the
only configuration that simultaneously (a) keeps the research promise of
"contrastive critic, no env reward" and (b) gives PPO a mathematically valid
importance ratio.
