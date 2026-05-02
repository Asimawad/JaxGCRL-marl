# mava CPPO optimization snapshot v5 — 2026-05-02

Successor to v4 with the **final 80M GAE smooth_adv result** and full set
of production bash scripts.

## Headline result

| | wall | peak win-rate |
|---|---|---|
| User's tuned baseline (`run.sh` Run-9) | ~35 min (est.) | 76.07% |
| `cppo_decoupled` (no GAE) | 26.7 min | 76.51% |
| **`cppo_decoupled` + GAE smooth_adv** | **26.7 min** | **79.88%** ← new best |

GAE smoothing on the contrastive single-step advantage gives **+3.4pp peak
win-rate at the same wall-clock**.

## Read these files first

- [HANDOFF.md](HANDOFF.md) — what this branch is, what we tried, what to do
  next. Read first.
- [CHANGES.md](CHANGES.md) — code-level changelog (delta from v4 is small,
  delta from v3 has the GAE refactor).
- [BENCHMARKS.md](BENCHMARKS.md) — all numbers including the final GAE
  trajectory.

## Files

```
README.md / HANDOFF.md / CHANGES.md / BENCHMARKS.md       # docs
mava/systems/icrl/anakin/ff_ppo_crl.py                    # OPTIMIZED + GAE smooth_adv
mava/systems/icrl/anakin/ff_ppo_crl_baseline.py           # PRISTINE pre-optimization copy
mava/systems/icrl/utils.py                                # micro-cleanup
scripts/                                                  # production run scripts (see scripts/README.md)
scripts/benchmark_systems.py                              # subprocess-isolated SPS benchmark
scripts/summarize_run.py                                  # post-run log analyzer
80M_runs/                                                 # win-rate trajectories from the production runs
```

## Run the new best config

```bash
bash scripts/run_cppo_decoupled_gae.sh
```

That's a 5-seed multi-run with wandb on. Will take ~2.5 hours total
(wall-clock per seed ~27 min, runs sequential).

For a single-seed quick test:

```bash
python -u -m mava.systems.icrl.anakin.ff_ppo_crl \
    --config-name ppo_crl env=smax env.scenario.task_name=smacv2_5_units \
    arch.num_envs=512 system.num_updates=1250 arch.num_evaluation=80 \
    arch.num_eval_episodes=2048 system.rollout_length=128 \
    system.batch_size=256 system.ppo_epochs=2 \
    +system.actor_batch_size=512 \
    +system.use_gae=true +system.gae_lambda=0.95 +system.gae_kind=smooth_adv \
    system.actor_lr=0.0005 system.q_lr=0.0001 system.ent_coef=0.01 \
    system.max_grad_norm=0.05 system.clip_eps=0.2 \
    system.logsumexp_penalty_coeff=0.91 system.lr_end=1e-07 \
    system.lr_decay_type=linear system.add_agent_id=1 \
    logger.loggers.wandb.enabled=false
```

## Prior branches

| branch | what |
|---|---|
| `mava-cppo-optimization-2026-05-02` | v1 — local optimizations only |
| `mava-cppo-optimization-v2-2026-05-02` | v2 — adds decoupled `actor_batch_size` |
| `mava-cppo-optimization-v3-2026-05-02` | v3 — decoupled epochs, critic subsample, broken GAE |
| `mava-cppo-optimization-v4-2026-05-02` | v4 — fixed GAE (smooth_adv), OOM fix, partial 80M GAE results |
| **`mava-cppo-optimization-v5-2026-05-02`** | **(this)** — final GAE 80M result + production bash scripts |
