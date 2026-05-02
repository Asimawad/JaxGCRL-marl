# mava CPPO optimization snapshot v4 — 2026-05-02

Successor to v3. Two important fixes on top of that branch:

1. **The `use_gae` ablation now actually works.** v3 shipped a broken GAE
   variant (`value_diff`, no reward) that drove the actor on noise at init
   and never escaped chance-level win rate. The new default
   (`gae_kind=smooth_adv`) uses the original single-step contrastive
   advantage `Q(s,a) − V(s)` as the per-step signal and applies GAE just
   for variance reduction across time. Working: at handoff time, the GAE
   80M run is ~5-10pp ahead of non-GAE at every mid-training eval.
2. **Production-scale OOM fixed.** First attempt at smooth_adv used 4D
   `(T, E, A, S)` tensors through the SA-encoder, which triggers a
   different XLA memory plan than the equivalent 3D `(N, A, S)` and OOMs
   at ~6.7 GB on 80M-scale runs. Fixed by flattening to `(N, ...)` for the
   network forward (matching what non-GAE already does), reshaping outputs
   back to `(T, E)` only for the small reverse-time GAE scan.

**Read [HANDOFF.md](HANDOFF.md) first** for full context — it has the
project goal, what's done, what's running, and what to do next.

## Headline results so far

| run | wall | peak win-rate | status |
|---|---|---|---|
| `cppo_decoupled` (no GAE) | 26.7 min | 76.51% (matches baseline 76.07%) | Done — recommended for production |
| `cppo_decoupled + use_gae=true (smooth_adv)` | TBD (~30 min) | **66.16% at eval 17/79 and climbing — running** | Currently in progress |

## Recommended config

```bash
# Production (no GAE, matches user's tuned baseline at 76.51%):
python -u -m mava.systems.icrl.anakin.ff_ppo_crl \
    --config-name ppo_crl env=smax env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 system.num_updates=1250 arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 system.rollout_length=128 \
    system.batch_size=256 system.ppo_epochs=2 \
    +system.actor_batch_size=512 \
    system.actor_lr=0.0005 system.q_lr=0.0001 system.ent_coef=0.01 \
    system.max_grad_norm=0.05 system.clip_eps=0.2 \
    system.logsumexp_penalty_coeff=0.91 system.lr_end=1e-07 \
    system.lr_decay_type=linear system.add_agent_id=1

# Same config + GAE smoothing (potentially better sample efficiency):
# add: +system.use_gae=true +system.gae_lambda=0.95 +system.gae_kind=smooth_adv
```
