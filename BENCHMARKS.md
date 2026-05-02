# Benchmark numbers (v4)

Same short-benchmark and 80M-run setup as v3. Only the GAE row(s) changed
between v3 and v4.

## Short SPS comparison (num_envs=256, ppo_epochs=1) — UNCHANGED from v3

| variant | actor_batch | critic_batch | extra | steady SPS (n=4) | vs baseline |
|---|---|---|---|---|---|
| `cppo_baseline` | 256 | 256 | — | 82,373 ± 91 | 1.000× |
| `cppo_decoupled` | 512 | 256 | — | 91,642 ± 255 | +11.3% |
| `cppo_ippo_actor` | N/2 | 256 | actor_ppo_epochs=4 | 94,211 ± 143 | +14.4% |
| `cppo_sub50` | N/2 | 256 | actor_ppo_epochs=4, sub_frac=0.5 | 137,180 ± 1,173 | +45.8% |
| `cppo_sub25` | N/2 | 256 | actor_ppo_epochs=4, sub_frac=0.25 | 177,548 ± 233 | +88.7% |
| `icrl` | — | 256 | — | 97,887 ± 272 | +18.8% |
| `ippo` (4-layer) | — | — | — | 273,289 ± 587 | +231.8% |

## 80M production runs (num_envs=512, ppo_epochs=2)

| run | wall | steady SPS | peak eval win-rate | learning |
|---|---|---|---|---|
| `cppo_decoupled` | 26.7 min | 54,924 ± ~200 | **76.51%** (eval 66) | ✅ matches baseline 76.07% |
| `cppo_sub25` | 9.1 min | 206,161 ± ~500 | 0.98% | ❌ broken (entropy collapsed) |
| `cppo_decoupled + use_gae (value_diff)` v3 | killed early | n/a | 0.34% | ❌ broken (no-reward TD on random V) |
| `cppo_decoupled + use_gae (smooth_adv)` v4 | **in progress at snapshot** | TBD | 67.4% at eval 19/79 and climbing | ✅ tracking ahead of non-GAE by ~5-10pp |

### Win-rate trajectories — head-to-head, evals 0-17

```
eval     no-GAE        smooth_adv
  0     0.05%         0.05%
  1    14.9%         11.4%
  2    27.6%         15.4%
  3    30.2%         24.4%
  4    32.1%         30.4%
  5    40.8%         46.1%   ← smooth_adv pulls ahead
  6    41.1%         48.1%
  7    36.6%         52.7%
  8    47.9%         55.7%
  9    44.5%         55.1%
 10    51.3%         57.3%
 11    51.9%         55.9%
 12    52.8%         54.6%
 13    54.0%         59.0%
 14    55.9%         59.4%
 15    57.2%         62.0%
 16    54.7%         66.0%
 17    58.8%         62.9%   ← latest at snapshot
```

Median lead ≈ +6-7pp through eval 17. Whether this translates to a
higher plateau (>76%) or just faster-to-plateau is the open question;
need the run to finish.

## Reminder: win-rate scale

`Win rate: 76.51` means **76.51%**, not 0.7651. See
`mava/utils/logger.py:82-83`.
