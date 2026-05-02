# Changelog — v5 (delta from v4)

## What's new

1. **Confirmed final 80M GAE result.** The smooth_adv 80M run completed
   while v4 was being pushed. Final stats: peak 79.88% at eval 74, final
   eval 77.10%, wall-clock 26.7 min, steady SPS 54,933 (statistically
   identical to non-GAE). The GAE smoothing gives +3.4pp peak win-rate
   over non-GAE at the same wall-clock — meaningful and free.

2. **Production bash scripts.** A `scripts/run_*.sh` for each variant we
   tried, modeled on the user's original `run.sh` style. Multi-seed
   (5 seeds), wandb on, all hparams inlined. Includes scripts for:
   - cppo_baseline (reference)
   - cppo_decoupled (recommended optimization)
   - cppo_decoupled_gae (current best)
   - cppo_ippo_actor (ablation)
   - cppo_sub50 / cppo_sub25 (ablations marked "RETUNE_NEEDED" because they
     break learning at the user's tuned baseline hparams)
   - ippo_4layer (matched-depth comparison baseline)
   - icrl (off-policy comparison baseline)
   - benchmark_sps (short SPS benchmark for paper Table-1)

   Plus `scripts/README.md` as a quick reference table.

## No code changes vs v4

`ff_ppo_crl.py`, `ff_ppo_crl_baseline.py`, `utils.py`,
`scripts/benchmark_systems.py`, `scripts/summarize_run.py` are byte-identical
to v4. The improvement came entirely from running v4 to completion at 80M.

## v3 → v4 → v5 in one paragraph

v3 introduced GAE under `system.use_gae` but with a broken default
(`gae_kind=value_diff`, no-reward TD residual that's pure noise at init).
v4 added `gae_kind=smooth_adv` (smooth the existing single-step Q − V over
time — Q and V at the same state cancel network noise so it works at init)
and made it the default; also fixed an OOM caused by 4D tensor shapes
hitting a different XLA memory plan. v5 confirmed the win at 80M scale and
shipped reproducibility scripts.
