# Pure CPPO Cleanup — Prompt for the Refactor Agent

## Your task

Refactor [`mava/systems/icrl/anakin/ppo_crl_continuous.py`](mava/systems/icrl/anakin/ppo_crl_continuous.py) into a **pure PPO + CRL** agent. Strip out everything that isn't textbook PPO or essential CRL machinery, and fix the one algorithmic bug that exists. The result should be a noticeably shorter, much more honest file.

You are working from a known-good outcome. A different team already did this same cleanup on a different fork (`jaxgcrl/agents/cppo/cppo.py`) and it produced **0.598 peak success on JaxGCRL Ant** — a +30% gain over the same team's CRL/SAC baseline. The lessons from that work are baked into this prompt; **follow them, don't re-derive them**.

---

## Algorithm definition (what "pure CPPO" means)

CPPO = **PPO actor** + **CRL contrastive critic** + **HER goal relabeling on the critic only**.

Specifically:

| component | what it does |
|---|---|
| Actor | Gaussian policy `(μ, log σ)`, tanh-squashed to action space. Trained with PPO clipped surrogate plus a **constant** entropy bonus. |
| Critic | InfoNCE between `phi(s, a)` and `psi(g)`. Loss is forward InfoNCE with the `norm` energy function (negative L2 with sqrt). |
| Advantage | `A = Q(s, a_taken, g) − E_{a' ~ π}[Q(s, a', g)]`, where the expectation is approximated with K Monte Carlo samples. **No V network. No GAE. No env reward.** |
| HER | Per-trajectory, sample a future achieved goal as the relabeled goal. Used **only** by the critic's contrastive loss. |
| Goal routing | The actor and the advantage condition on the **real env goal** (the one the rollout actor saw). The critic conditions on the **relabeled goal**. |

There are no SAC bits, no value networks, no env-reward leakage, no entropy controllers, no critic warmup, no temperature scaling.

---

## What to delete

The current file has accumulated five categories of cruft. Remove every line of all five.

### 1. SAC adaptive entropy controller

Search-and-destroy these names — config fields, helper variables, code paths, and any references in `PPOCRLLearnerState`:

- `use_adaptive_entropy`
- `target_entropy`
- `alpha_lr`
- `log_alpha` (in `PPOCRLLearnerState` and everywhere downstream)
- `alpha_opt_state`
- `entropy_floor_coeff`
- `entropy_floor_penalty`
- `init_log_alpha`, `init_alpha_opt_state`
- The "Alpha update (SAC-style adaptive entropy)" block in `_update_minibatch` (~10 lines).
- The `jnp.where(use_adaptive_entropy, ...)` branches that gate `ent_coef`.

The single entropy coefficient that survives is a plain Python float, set once from `config.system.ent_coef`, used as a constant scalar in the actor loss.

### 2. Env-reward leakage paths

This entire family is incompatible with "pure CRL — no env reward":

- `reward_advantage_coeff` and the `_scan_returns` block.
- `use_gae` and **everything** under it:
  - `_gae_step` scan, `gae_advantages`, `gae_targets`, `last_val`, `last_done`.
  - The GAE branch in *Phase 1: Compute advantages*.
  - `relabeled["gae_advantage"]`, `relabeled["gae_target"]`, `relabeled["env_goal"]` plumbing.
  - The `if use_gae: actor_goal = batch["env_goal"] else: actor_goal = goal` switch (collapse to a single, correct path — see "Bug fix" below).
  - Value network entirely: `ICRLValueNet` import, `value_params`, `value_opt_state`, `value_apply`, `value_update_fn`, `_value_loss_fn`, the value update block in `_update_minibatch`, and value-related fields in `PPOCRLLearnerState`.
  - `gae_lambda`, `vf_coef`.
- `use_reinforce` branch (always use the `Q − V_MC` baseline path).

### 3. Mava-specific heuristics not in the JaxGCRL CRL baseline

These were Mava-continuous-only knobs and didn't help in our ablations:

- `contrastive_temperature`. Delete the field, the local variable, and the `logits_scaled = logits / contrastive_temperature` lines in both `_critic_loss_fn` and the warmup variant. Pass `logits` directly into `contrastive_loss_fn`. The JaxGCRL paper baseline uses bare InfoNCE.
- `mc_std_floor` (always 0).
- `num_critic_warmup_epochs` and the `_critic_only_minibatch` / `_critic_warmup_epoch` machinery.
- `ent_schedule_horizon` and `ent_coef_end`. Use a single constant `ent_coef`. Delete the schedule block in `_update_step`.
- `sa_action_scale` (always 1.0). Inline so the SA encoder just receives the raw action.

### 4. Stale config defaults

After (1)–(3), audit every `config.system.get("dead_field", default)` call and delete the ones whose targets no longer exist.

### 5. The unused `value_*` machinery in `learner_setup`

After deleting GAE, the branch that builds `value_torso`, `value_network`, `value_opt`, etc. should be gone. `apply_fns` and `update_fns` should drop the value entries (3-tuples, not 4-tuples).

---

## What to fix — the goal-routing bug

This is the single most important change. It is **not optional**.

### The bug

Look at the rollout in `_env_step` ([line ~175](mava/systems/icrl/anakin/ppo_crl_continuous.py#L175)):

```python
gc_obs = jnp.concatenate([obs, current_ag, env_state.goal], axis=-1)
...
log_prob = ...   # recorded conditioned on env_state.goal (REAL goal)
```

Now look at the actor loss in `_update_minibatch` (with `use_gae=False`):

```python
actor_goal = goal                             # RELABELED HER goal
actor_input = jnp.concatenate([obs, current_ag, actor_goal], axis=-1)
new_log_prob = ...                            # conditioned on RELABELED goal
ratio = jnp.exp(new_log_prob - old_log_prob)  # numerator and denominator condition on different inputs
```

The PPO importance ratio is comparing `π(a | s, real_goal)` (the rollout policy) against `π(a | s, relabeled_goal)` (the training policy on a different input). They're literally different conditional distributions. As the policy starts conditioning strongly on the goal channel, the ratio explodes, the clip kicks in, and the gradient signal dies. Empirically: peak ~0.25 then catastrophic collapse on Ant-class envs.

The **discrete** Mava reference [`mava/systems/icrl/anakin/ppo_crl.py`](mava/systems/icrl/anakin/ppo_crl.py) routes goals correctly and does **not** have this bug. Mirror its pattern.

### The fix

1. In `_flatten_crl`, in addition to writing the relabeled `goal` into the result dict, **also save the original env goal** under a different key (the discrete reference uses `extras["real_ultimate_goal"]` — pick whatever name you like, just be consistent).

2. In `_compute_advantages`:
   - Use the **real env goal** for the actor MC sampling input.
   - Use the **real env goal** in the goal encoder (`g_repr`).

3. In `_actor_loss_fn`:
   - Always use the **real env goal** in the actor input. No `if use_gae` switch — that's gone now anyway.

4. In `_critic_loss_fn`:
   - **Keep** using the relabeled goal. That's the CRL contrastive signal — the whole point of HER.

After this change, the PPO ratio becomes mathematically meaningful (numerator and denominator condition on the same input), and the late-training collapse on hard envs disappears.

---

## What to keep

Untouched, exactly as in the current file:

- The PPO clipped surrogate: `min(ratio * A, clip(ratio) * A)` taken as `-mean(max(...))`.
- Multi-epoch minibatch loop (`num_epochs` epochs over the post-relabel data).
- Adam optimizer, global grad clip.
- Pre-tanh Gaussian log-prob computation (no Jacobian correction — both old and new use the same form, so the missing Jacobian cancels in the ratio). Don't introduce a Jacobian correction; you will break the ratio.
- HER relabeling per-trajectory (`_flatten_crl`).
- InfoNCE forward / backward / sym variants (already in `contrastive_loss_fn`).
- `logsumexp_penalty_coeff` regularizer on the contrastive loss.
- `skip_connections=4`, `use_layer_norm=True`, Swish activation (these are **JaxGCRL CRL paper standards**, not heuristics — verified by reading `jaxgcrl/agents/crl/`).
- `log_std_min=-5.0`, `log_std_max=2.0` (paper standards).
- `repr_size=64` (paper default).
- `use_achieved_goal` flag — keep it, the `[obs, current_ag, goal]` actor input is the paper convention.
- Stochastic eval (sample, don't take the mean) — paper convention.

---

## Final shape of the file

After the cleanup the file should be approximately:

- `make_eval_act_fn(...)` — unchanged, just remove the `use_gae` references if any.
- `PPOCRLLearnerState` — drops `log_alpha`, `alpha_opt_state`, `value_params`, `value_opt_state`. What's left: `params`, `opt_states`, `key`, `env_state`, `last_timestep`, `update_count`.
- `PPOTransition` — drops `value` and `env_goal` fields. (`env_goal` is no longer needed because we're saving real goal in `_flatten_crl` output now, not the rollout buffer.)
- `get_learner_fn`:
  - Top: read PPO + CRL configs only. ~10 lines down from current.
  - `_env_step`: drops the `pre_step_goal`, value prediction, and `flat_env_goal` lines.
  - `_flatten_crl`: writes `goal` (relabeled, same as today) AND a new field for real env goal.
  - `_update_step`: drops the GAE block, the returns scan, the entropy schedule. Just: rollout → relabel → compute advantages → train epochs.
  - `_compute_advantages`: simpler — only the CRL Q-V path, conditioned on the real goal.
  - `_update_minibatch`: drops the alpha update, the value update, the GAE-vs-CRL switch on `actor_goal`. Just: critic loss → critic step → actor loss → actor step.
  - `_critic_loss_fn`: drops `logits_scaled = logits / contrastive_temperature`. Pass logits directly.
  - `_train_epoch`: drops `log_alpha`, `alpha_opt_state`, `vp`, `vo` from the carry tuple.
- `learner_setup`: drops the value network branch entirely. `apply_fns` is `(sa_encoder.apply, goal_encoder.apply, actor_network.apply)`. `update_fns` is `(actor_opt.update, critic_opt.update)`.
- `run_experiment`: minor — drops references to `value_*` if any, otherwise unchanged.

Estimated line count drop: **200–300 lines** out of ~1060.

---

## Acceptance criteria

The cleaned file should satisfy:

1. **No occurrences** of any of these substrings (case-sensitive, in the file):
   - `log_alpha`, `alpha_lr`, `target_entropy`, `entropy_floor`, `adaptive_entropy`
   - `use_gae`, `gae_lambda`, `gae_advantages`, `gae_target`, `gae_step`
   - `value_params`, `value_apply`, `value_update_fn`, `value_opt`, `ICRLValueNet`, `vf_coef`
   - `reward_advantage_coeff`, `use_reinforce`, `mc_std_floor`
   - `num_critic_warmup_epochs`, `_critic_only_minibatch`, `_critic_warmup_epoch`
   - `contrastive_temperature`, `logits_scaled`
   - `ent_coef_end`, `ent_schedule_horizon`, `sa_action_scale`

   (Run `rg` against the file. The list must come back empty.)

2. The PPO ratio in `_actor_loss_fn` is conditioned on the **real env goal**, not the relabeled goal. Verify by reading the actor input concat call.

3. The critic in `_critic_loss_fn` is conditioned on the **relabeled goal**. Verify by reading the goal encoder call.

4. `_critic_loss_fn` passes raw logits to `contrastive_loss_fn` (no temperature division).

5. The file imports do not include `ICRLValueNet`. The `mava.networks` import is `from mava.networks import GoalEncoder, ICRLActor, SAEncoder`.

6. The script still runs end-to-end on a goal-conditioned continuous-action env. JIT compiles. No undefined names.

---

## Reference for "what good looks like"

The companion repo's `jaxgcrl/agents/cppo/cppo.py` (in the parent codebase, not this `cppo-code/` tree) is the production-validated equivalent of this cleanup. **Read it before you start.** Specifically inspect:

- The dataclass fields (no `target_entropy`, `alpha_lr`, etc.).
- The `actor_loss_fn` — uses `batch["target_goal"]`, not `batch["relabeled_goal"]`.
- The `compute_advantage` — uses `batch["target_goal"]`.
- The critic loss — uses `batch["relabeled_goal"]`.

You're producing the Mava-continuous equivalent of that file, not a different design.

---

## Recommended hyperparameters for verification

After cleanup, run the cleaned file against a goal-conditioned continuous env (JaxNav or anything similar Mava supports). The recipe that produced 0.598 peak on Ant in the parent codebase was:

```yaml
system:
  rollout_length: 128
  num_epochs: 8
  batch_size: 256
  num_mc_samples: 64
  gamma: 0.9999
  actor_lr: 3e-4
  q_lr: 3e-4
  clip_eps: 0.15
  max_grad_norm: 1.0
  ent_coef: 0.0001
  contrastive_loss_fn: fwd_infonce
  energy_fn: norm
  logsumexp_penalty_coeff: 0.1
  log_std_min: -5.0
  log_std_max: 2.0
  rep_size: 64
  use_achieved_goal: true

arch:
  num_envs: 1024

network:
  hidden_sizes: [512, 512, 512, 512]
  skip_connections: 4
  use_layer_norm: true

total_env_steps: 100000000
```

Performance benchmark: should match or exceed the equivalent CRL+SAC baseline on the same env. If it underperforms by more than 10%, suspect a bug in the goal routing — re-read the bug fix section.

---

## What NOT to do

- **Don't add a value network "for stability"** — pure CRL means MC V baseline only.
- **Don't add a Jacobian correction to log-prob** — both old and new use pre-tanh Gaussian; the correction would break the ratio.
- **Don't introduce a "soft" alpha controller** as a compromise — that's just adaptive entropy with a different name.
- **Don't keep `contrastive_temperature` as 1.0 default "for flexibility"** — delete it. Flexibility we don't need is dead weight.
- **Don't change `log_std_min`/`log_std_max` defaults** away from the paper's `-5/+2` — those are the JaxGCRL CRL standards, not heuristics.
- **Don't merge the relabeled and real goals into a single field** — they need to coexist in the post-relabel dict because the critic and the actor consume different ones.
- **Don't rename `_flatten_crl`** unless you're also doing wider renaming. Keep the diff scoped.

---

## Output expectation

A single rewritten file at the same path. Tested for syntax (`python -m py_compile`). Acceptance-criteria grep returns clean. Brief change summary at the end of the agent's response, listing how many lines were removed and confirming the goal-routing bug fix landed.
