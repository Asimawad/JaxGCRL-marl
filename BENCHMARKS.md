# Benchmark numbers — all in one place

All runs at `task_name=smacv2_5_units`, single H100 80 GB, JAX cuda12-pip.

## Reproducing the short benchmarks

```bash
# Short benchmark — 12 updates × 6 evals, used for SPS comparison only
python scripts/benchmark_systems.py \
    --num_envs 256 --num_updates 12 --num_evaluation 6 --rollout_length 128 \
    --num_eval_episodes 16 \
    --systems cppo_baseline cppo_decoupled cppo_ippo_actor icrl ippo \
    --per-system "cppo_decoupled:+system.actor_batch_size=512" \
    --per-system "cppo_ippo_actor:+system.actor_batch_size=81920" \
    --per-system "cppo_ippo_actor:+system.actor_ppo_epochs=4" \
    --per-system "cppo_ippo_actor:+system.critic_ppo_epochs=1" \
    --per-system "ippo:network.actor_network.pre_torso.layer_sizes=[512,512,512,512]" \
    --per-system "ippo:network.critic_network.pre_torso.layer_sizes=[512,512,512,512]"
```

## Speed-only comparison (num_envs=256, ppo_epochs=1)

| variant | actor_batch | critic_batch | extra | steady SPS (n=4) | vs baseline |
|---|---|---|---|---|---|
| `cppo_baseline` | 256 | 256 | — | 82,373 ± 91 | 1.000× |
| `cppo_decoupled` | 512 | 256 | — | 91,642 ± 255 | +11.3% |
| `cppo_ippo_actor` | N/2 ≈ 82k | 256 | actor_ppo_epochs=4 | 94,211 ± 143 | +14.4% |
| `cppo_sub50` | N/2 | 256 | actor_ppo_epochs=4, critic_subsample_fraction=0.5 | 137,180 ± 1,173 | +45.8% |
| `cppo_sub25` | N/2 | 256 | actor_ppo_epochs=4, critic_subsample_fraction=0.25 | 177,548 ± 233 | +88.7% |
| `icrl` (off-policy SAC + replay) | — | 256 | — | 97,887 ± 272 | +18.8% |
| `ippo` (4-layer to match CRL depth) | — | — | num_minibatches=2, ppo_epochs=4 | 273,289 ± 587 | +231.8% |

**Network sizes for the unified comparison:**
- All actors: `[512, 512, 512, 512]` 4-layer MLP
- IPPO critic: `[512, 512, 512, 512]` (overridden up from default `[512, 512]`)
- CPPO/ICRL SA encoder + goal encoder: `[512, 512, 512, 512]` (default)

The IPPO 3.3× speedup over CPPO_decoupled is mostly intrinsic algorithm
cost (no goal encoder, no `[B,B]` matrix, only 8 SGD iters/update vs ~640),
not implementation slack.

## 80M-timestep production runs (num_envs=512, ppo_epochs=2)

Hyperparameters from `run.sh` Run-9 (the user's tuned baseline):
```
actor_lr=0.0005    q_lr=0.0001    ent_coef=0.01
max_grad_norm=0.05    clip_eps=0.2    logsumexp_penalty_coeff=0.91
lr_end=1e-07    lr_decay_type=linear    add_agent_id=1
```

| run | wall | steady SPS | peak eval win-rate | final eval | learning |
|---|---|---|---|---|---|
| `cppo_decoupled` (`actor_batch_size=512`) | **26.7 min** | 54,924 ± ~200 | **76.51%** (eval 66) | 75.59% | ✅ matches baseline (76.07%) |
| `cppo_sub25` (`actor_batch_size=N/2, actor_ppo_epochs=4, critic_subsample_fraction=0.25`) | **9.1 min** | 206,161 ± ~500 | 0.98% | 0.64% | ❌ **broken** — entropy collapsed, PPO clip biting |
| `cppo_decoupled + use_gae=true, gae_lambda=0.95` | (running at handoff) | TBD | TBD | TBD | early signal: 0.05→0.24→0.34% over first 3 evals (vs non-GAE 0.05→14.9→27.6%) — likely too weak a signal at init |

### Win-rate trajectory: cppo_decoupled (the success)

```
  1M:   0.05%   (random init)
  5M:  40.77%
 10M:  51.32%
 20M:  59.42%
 30M:  68.26%
 40M:  69.24%
 50M:  73.83%
 60M:  75.10%
 70M:  75.30%
 80M:  75.59%   ← run end
```

Clean ramp-up through ~50M timesteps, plateau 74-76% from then on.

### Win-rate trajectory: cppo_sub25 (the failure)

```
  1M:  0.05%      80M:  0.64%   <- all 80 evals stuck near zero
```

`Categorical accuracy` (contrastive task) only ever reached 0.072 ≈ 7%, vs
non-broken runs that reach 0.6+. The combination of IPPO-style actor + 25%
critic subsample + the user's tuned-for-baseline hparams pushed the policy
into a deterministic state in the first few updates.

## Win-rate scale reminder

`mava/utils/logger.py:82-83` multiplies by 100. So `Win rate: 76.51` means
**76.51%**, not 0.7651. Use this scale when reading any of the numbers
above.

## Compute breakdown (cppo_decoupled @ num_envs=512, rollout=128, num_agents=5)

Per update:
```
N (transitions per update)        = 512 × 5 × 128 = 327,680
actor_batch_size                  = 512
critic_batch_size                 = 256
actor minibatches per epoch       = 327680 / 512 = 640
critic minibatches per epoch      = 327680 / 256 = 1280
ppo_epochs                        = 2
total actor SGD iters / update    = 1280
total critic SGD iters / update   = 2560
```

## Per-system comparison @ same SGD budget

(For an apples-to-apples "per env step, how many gradient updates does each
system make?")

| system | SGD iters / env step |
|---|---|
| ICRL (off-policy SAC) | 500 / 128 ≈ **3.9** |
| CPPO baseline | 640 / 128 ≈ 5.0 (interleaved, ppo_epochs=1) |
| CPPO decoupled (this work) | 960 / 128 ≈ 7.5 (ppo_epochs=1) |
| CPPO ippo_actor | 648 / 128 ≈ 5.06 (ppo_epochs=1) |
| **CPPO decoupled @ ppo_epochs=2** | 1920 / 128 ≈ 15.0 |
| IPPO (reward) | 8 / 128 ≈ 0.06 (ppo_epochs=4, num_minibatches=2) |
