# Handoff for the Next Claude — Read this first

**Read time: 5 minutes. Then `git log --oneline -15` and `ls mava/jaxgcrl/`.**

This file is the authoritative snapshot of where things stand at the end of the
2026-04-28 session. The user (Asimawad / yassir@aims.ac.za) will pull this on a
fresh machine and you'll continue from here.

---

## 1. Who you're working with

- **User**: Asimawad on GitHub, yassir@aims.ac.za, works at AIMS / Instadeep
- **Primary stack**: Mava (multi-agent JAX RL framework), familiar with hydra
  configs and optuna sweeps. Knows the JaxGCRL paper figures from memory.
- **Communication preferences**: terse. No emojis unless asked. State what
  changed and what's next in 1–2 sentences. End-of-turn summary is one or two
  lines, never a wall of text. If you need a decision, ask one focused question;
  don't explain three options when one is clearly correct.
- **Trust profile**: technically strong; doesn't need hand-holding on
  algorithmic content. Will push back if you over-explain. Doesn't paste
  long terminal output for fun — if they paste output, something's wrong.
- **They paste credentials in chat sometimes.** When they do, redact in any
  files you commit, and warn them to rotate. They've been told three times in
  this session; don't lecture, just remind once and move on.

## 2. What the project is

Research goal: **benchmark "CPPO" against the JaxGCRL CRL paper baseline.**

CPPO = **PPO actor + CRL contrastive InfoNCE critic + HER (on critic only).**
No SAC adaptive entropy, no value network, no GAE, no env-reward in the loss.
Pure on-policy PPO actor reading advantages from a contrastive Q-V baseline.

Paper: Bortkiewicz et al. ICLR 2025, *Accelerating Goal-Conditioned RL.* arxiv
2408.11052. Code at github.com/MichalBortkiewicz/JaxGCRL.

**Current best result**: **0.598 peak success_any on JaxGCRL Ant @ 50M steps**,
**0.623 at ~80M.** Compare to paper's CRL (~0.85), our own CRL baseline (~0.43),
SAC+HER (~0.95). For "pure on-policy + contrastive critic" this is a respectable
result. The bigger gap (CPPO 0.60 vs paper CRL 0.85) is the cost of staying
on-policy — paper's CRL uses off-policy SAC actor with replay buffer.

Untested but adapter exists for: **JaxNav** (single-agent variant), via the
`jaxnav-adapter` branch.

## 3. Repo geography

Two parallel implementations co-exist; **don't break either**.

```
/                                 (Asimawad/JaxGCRL-marl, branch cppo-pure-ppo or jaxnav-adapter)
├── jaxgcrl/                      ← ORIGINAL tyro-driven implementation (UNTOUCHED)
│   ├── agents/{cppo,crl,ppo,sac,td3}/
│   ├── envs/                     ← brax single-agent envs (ant, humanoid, …)
│   └── utils/{config.py,env.py,evaluator.py,replay_buffer.py}
├── run.py                        ← original tyro CLI entry
├── scripts/run_cppo_*.sh         ← per-env recipe runners (still work)
│
├── mava/jaxgcrl/                 ← NEW hydra port subpackage
│   ├── agents/{cppo,crl}/        ← copies of jaxgcrl/agents/{cppo,crl}/ —
│   │                                 only difference is import paths
│   │                                 (jaxgcrl.* → mava.jaxgcrl.*)
│   ├── envs/                     ← copy of jaxgcrl/envs/ + jaxnav_adapter.py
│   ├── utils/
│   │   ├── env.py                ← create_env, dropped MetricsRecorder
│   │   ├── config.py             ← RunConfig dataclass, no tyro
│   │   ├── evaluator.py          ← copied verbatim
│   │   ├── replay_buffer.py      ← copied verbatim (CRL only)
│   │   └── logger.py             ← MavaLogger drop-in (Console + Wandb + JSON)
│   └── run.py                    ← @hydra.main entry point
│
├── mava/configs/jaxgcrl/         ← NEW hydra config tree
│   ├── cppo.yaml                 ← top-level, sweep-by-default (mode: MULTIRUN)
│   ├── crl.yaml                  ← same shape for CRL paper baseline
│   ├── cppo_norun.yaml           ← single-run variant (no sweep)
│   ├── system/{cppo,crl}.yaml    ← agent dataclass values
│   ├── env/                      ← {ant,humanoid,reacher,pusher_hard,
│   │                                ant_u_maze,ant_big_maze,jaxnav}.yaml
│   ├── logger/logger.yaml        ← Console + Wandb + JSON (marl-eval format)
│   └── search_space/{cppo,crl}.yaml ← optuna sweep ranges
│
├── mava/                         ← rest of vendored Mava (icrl, ppo_crl_continuous, …)
│                                   intentionally NOT MODIFIED — the user works
│                                   here for other research lines
│
├── pyproject.toml                ← deps + `[project.optional-dependencies] jaxnav = ["jaxmarl"]`
├── Makefile                      ← `make setup` and `make verify` targets
├── scripts/fix_execstack.py      ← idempotent CUDA-wheel patch for hardened kernels
├── goal_routing_report.md        ← analysis of the HER goal-routing bug
├── PURE_CPPO_CLEANUP_PROMPT.md   ← brief for *another* agent to clean up
│                                   mava/systems/icrl/anakin/ppo_crl_continuous.py
└── context.zip                   ← THIS conversation's chat transcript +
                                    plans (gitignored — has exposed creds)
```

## 4. Branches on origin (Asimawad/JaxGCRL-marl)

```
cppo-pure-ppo          base branch with all the work pre-jaxnav
└── jaxnav-adapter     adds JaxNav single-agent support on top
```

**Last commit on `cppo-pure-ppo`**: `02a4899` — "Restore Mava-style sweep
defaults in cppo.yaml and crl.yaml" (or a later metrics-fix commit).

**Last commit on `jaxnav-adapter`**: `a50f6b5` — "JaxNav adapter: add
distance-based goal mode (HER target = 0)". Adds `goal_type` (position |
distance) to `mava/configs/jaxgcrl/env/jaxnav.yaml`; default stays `position`.

When resuming, **first check `git remote -v`** — sandbox refreshes lose the
remote. If it's missing, re-add it (the URL is `JaxGCRL-marl`, NOT `jaxgcrl`):

```
git remote add origin https://github.com/Asimawad/JaxGCRL-marl.git
git fetch origin
```

Then `git log --oneline -10` to see the real tip.

## 5. The recipe that produced 0.598 → 0.623 on Ant

Lives in [mava/configs/jaxgcrl/system/cppo.yaml](mava/configs/jaxgcrl/system/cppo.yaml).
Highlights:

| knob | value | comment |
|---|---|---|
| `actor_lr`, `q_lr` | 3e-4 | standard |
| `clip_eps` | 0.15 | a bit conservative vs vanilla PPO's 0.2 |
| `discounting` | 0.9999 | high; ant is sparse + long-horizon |
| `rollout_length` | 128 | |
| `num_epochs` | 8 | |
| `batch_size` | 256 | |
| `num_mc_samples` | 32 | for the V baseline (Q − E[Q]) |
| `ent_coef` | 1e-4 | constant. 1e-3 was 10× too strong → the actor flailed. |
| `use_adaptive_entropy` | false | the SAC controller — **don't turn this on** |
| `contrastive_loss_fn` | fwd_infonce | |
| `energy_fn` | norm | = neg-L2 with sqrt; matches paper, dot was −5% |
| `contrastive_temperature` | 1.0 | matches paper. 2.5 (Mava) was a regression |
| `log_std_min/max` | -5 / +2 | wider cap was the single biggest knob |
| `h_dim` × `n_hidden` | 512 × 4 | a bit deeper than paper's 256×2 |
| `use_layer_norm`, `skip_connections=4` | true, 4 | paper does the same |
| `terminate_on_success` | true | env wrapper |
| `sa_state_mode` | state_only | SA encoder sees `[state, achieved]` |
| `actor_input_mode` | obs_full_ach | actor sees `[state, achieved, target]` |

**The four changes that took us from 0.498 → 0.598 on Ant**:
1. `log_std_max`: -0.5 → 2.0 (paper default)
2. `contrastive_temperature`: 2.5 → 1.0 (paper bare InfoNCE)
3. `energy_fn`: dot → norm
4. `ent_coef`: 1e-3 → 1e-4 (matches what adaptive controller naturally settled on)

## 6. Bugs we caught and fixed (don't reintroduce these)

### 6a. HER goal-routing bug (the big one)

`mava/systems/icrl/anakin/ppo_crl_continuous.py` (the original Mava
continuous file we ported from) has actor and advantage using the **relabeled
HER goal**, while the rollout's `old_log_prob` is computed under the **real env
goal**. PPO importance ratio is comparing two unrelated conditional
distributions → late-training collapse on hard envs.

Our fix: actor and advantage use the real env goal; only the critic's contrastive
loss uses the relabeled goal. Documented at length in
[goal_routing_report.md](goal_routing_report.md). The discrete Mava reference
([example.py](example.py)) does it correctly; the continuous one doesn't.

If you're tempted to "simplify" by using the same goal everywhere in the actor
path — DON'T. Read the report first.

### 6b. Pre-tanh log_prob without Jacobian correction

The actor samples `x_t = μ + σ·ε`, returns `tanh(x_t)`, but computes `log_prob`
in pre-tanh Gaussian space (no `Σ log(1 − tanh²)` correction). This is **not**
standard PPO and **not** standard SAC. Works because:
- Both `old_log_prob` and `new_log_prob` use the same form → the missing
  Jacobian cancels in the ratio.
- The "entropy" we report is the pre-tanh Gaussian entropy, not the squashed-
  policy entropy. Target_entropy values are interpreted on this scale.

Don't add the Jacobian correction — it'll break the ratio.

### 6c. Gitignore pattern `env/` swallowing the hydra env config dir

The bare `env/` (intended for Python virtualenvs named `env/`) matched
`mava/configs/jaxgcrl/env/` for **weeks** before I caught it on a fresh clone.
Anchor virtualenv ignores: `/env/` (root only), `/.venv/`, `/venv/`. Same lesson
applies to any other "common dir name" pattern.

### 6d. ConsoleLogger duplication

Hydra adds a `StreamHandler` to the root logger when `@hydra.main` runs. Our
`_ConsoleLogger.__init__` was adding another → every record printed twice. Fix:
strip pre-existing `StreamHandler`s on init, preserve `FileHandler`s. See
[mava/jaxgcrl/utils/logger.py](mava/jaxgcrl/utils/logger.py) `_ConsoleLogger`.

### 6e. JsonLogger missing success in metrics.json

`success` was being written to `state.info` but **brax's training wrappers only
aggregate `state.metrics` across episodes**. The JaxGCRL evaluator reads
`episode_metrics["success"]` (from `state.metrics`) — so if you only set info,
the evaluator silently skips the success_any metric. Fix in the JaxNav adapter:
seed `metrics={"success": 0.0, ...}` in reset and merge `success` into
`state.metrics` on every step.

### 6f. Optuna sweeper grammar

Our `search_space/cppo.yaml` originally used optuna's distribution schema
(`{type: float, low: ..., high: ...}`). Hydra-optuna-sweeper expects **CLI
override syntax** (`tag(log, interval(1e-5, 1e-3))`, `choice(256, 512, 1024)`).
Get this wrong and you'll see `OverrideParseException: no viable alternative at
input '{'type''`.

### 6g. Brax wrappers expect specific State pytree

`envs.training.wrap()` adds keys to `state.info` (`steps`, `truncation`,
`first_obs`, `first_pipeline_state`) and `state.metrics` (`reward`). If your env
or wrapper rebuilds State from scratch on `step()`, those keys disappear and
`jax.lax.scan` fails: *"carry input and carry output must have the same pytree
structure."* Always use `state.replace(...)` to preserve wrapper-added keys.

## 7. How to run anything

### Setup (fresh checkout)

```bash
git clone https://github.com/Asimawad/JaxGCRL-marl.git
cd JaxGCRL-marl
git checkout cppo-pure-ppo                  # or jaxnav-adapter for jaxnav
make setup                                  # uv sync + execstack patch
make verify                                 # imports + hydra config resolution
export WANDB_API_KEY=...                    # rotate this; see §10
```

### Sweep (default behavior — Mava-style)

```bash
python -m mava.jaxgcrl.run env=ant_u_maze   # 20-trial optuna sweep
python -m mava.jaxgcrl.run env=ant_u_maze hydra.sweeper.n_trials=1   # single trial w/ random hp
```

### Single run with the recipe (no sweep)

```bash
python -m mava.jaxgcrl.run --config-name=cppo_norun env=ant_u_maze
```
This uses the recipe defaults from `system/cppo.yaml`, output to `outputs/`.

### Multi-seed (no sweep)

```bash
python -m mava.jaxgcrl.run --config-name=cppo_norun -m env=ant run.seed=0,1,2,3,4
```

### JaxNav (requires `--extra jaxnav` install — see §9)

```bash
git checkout jaxnav-adapter
uv sync --extra jaxnav
python -m mava.jaxgcrl.run --config-name=cppo_norun env=jaxnav
```

### Cluster entrypoint (`run.sh`)

The cluster runs `bash run.sh` automatically. Override via env vars: `METHOD`,
`ENV`, `STEPS`, `SEED`, `N_TRIALS`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`,
`WANDB_TAGS`, `EXP_NAME`. See [run.sh](run.sh) header for the full list.

## 8. Outputs

- **Console**: one coloured line per LogEvent (MISC / TRAINER / EVALUATOR /
  ABSOLUTE) per eval. No timestamp prefix. Format matches Mava's real
  ConsoleLogger.
- **Wandb** (when enabled): project default `cppo-brax`. Group format
  `cppo-<env>-<date>`. Tags configurable via cluster env var.
- **`results/json/<env>/metrics.json`**: marl-eval-compatible nested JSON
  ready for rliable / sample-efficiency plots. Schema:
  `env → task → algo → seed_N → {step_M, absolute_metrics}`. Keys inside each
  step block: `step_count`, `elapsed_time`, `success_rate` (0–1 fraction;
  this is the headline metric, NOT 0–100 like SMAX `win_rate`),
  `mean_episode_return`, `steps_per_second`, `mean_episode_length`.
- **`outputs/<date>/<time>/`**: hydra single-run dir (resolved config + log
  file).
- **`multirun/<date>/<time>/<trial>/`**: hydra multi-run / sweep dirs.

## 9. Dependencies and the brax-version conflict

Default `uv sync` installs `brax==0.12.1, jaxlib==0.4.25+cuda12.cudnn89`. Works
for ant, humanoid, reacher, pusher_hard, ant_u_maze, ant_big_maze.

`uv sync --extra jaxnav` adds `jaxmarl`, which **forces brax to 0.10.3** (a
downgrade). The brax envs *still work* on 0.10.3 because the API surface
JaxGCRL touches is stable, but Ant's behavior in particular looks slightly
different (faster termination, `terminate_when_unhealthy` triggers more
aggressively). If precise Ant numbers matter, **use a separate venv for
jaxnav**.

The execstack patch (`scripts/fix_execstack.py`) clears `PT_GNU_STACK.X` on the
CUDA wheels. Idempotent: running on stock kernels is a no-op. Re-run after
every `uv sync` because pip reinstalls clobber the patch.

## 10. Compromised credentials (still valid as of session end)

Three GitHub PATs and two wandb keys leaked into the conversation transcript
(`context.zip` contains all of them verbatim — plus the third PAT was pasted
again on 2026-04-28 when the user wanted to push and there was no auth set
up). **Tell the user once** then move on:

- Revoke any GitHub PATs from this session:
  https://github.com/settings/tokens
- Rotate any wandb keys from this session:
  https://wandb.ai/authorize

**Do not commit `context.zip` to the public fork** — it's gitignored, but
double-check before pushing. If the user wants to push the chat zip
somewhere private (private repo, S3, their laptop), that's their call;
just don't put it in the public history.

## 11. Workflow conventions established this session

- **Always secret-scan staged content** before committing. Use a regex that
  matches the prefixes for wandb keys, classic GitHub PATs, and fine-grained
  GitHub PATs — see the Bash invocations earlier in any commit-and-push step
  in the chat transcript for the exact pattern. The scan is a hard gate: if it
  fails, fix the file and re-stage rather than overriding.
- **Never `git push` automatically.** Pattern: re-add remote with token, push,
  remove remote. The token isn't sitting in `.git/config`. The repo URL is
  `https://github.com/Asimawad/JaxGCRL-marl.git` (capitals — the lowercase
  `jaxgcrl` 404s and ate ~10min last session).
- **Never paste a PAT in chat.** It goes into `context.zip` verbatim. Drop it
  in `/tmp/pat` (chmod 600), or use `gh auth login`. If the user pastes one
  anyway, push with it as a one-shot URL `https://Asimawad:<PAT>@github.com/...`
  (don't `git remote set-url` it — the token persists in `.git/config`), then
  immediately tell them to revoke at https://github.com/settings/tokens.
- **Don't introduce backwards-compat shims** for removed dataclass fields. If
  a field is dead, delete it and let the next config dump look cleaner.
- **One bug fix per commit when possible.** The user reads `git log` to
  reconstruct what happened.
- **Update `goal_routing_report.md` if you find another goal-routing
  inconsistency.** The discrete Mava reference is the trustworthy one.

## 12. Open work / what to look at next

| topic | state | next move |
|---|---|---|
| **Push past 0.65 on Ant** | hit 0.623 at 80M | run 200M with same recipe; or try `h_dim=1024` |
| **Run all 6 base envs with the recipe** | scripts exist; `run_cppo_*.sh` per env | execute on cluster, collect json |
| **JaxNav numbers** | adapter works (position + distance goal modes), untested at scale | full run from `jaxnav-adapter` branch; pick `goal_type` per experiment |
| **Ablate CPPO heuristics** | known load-bearing list in §5; sweep template at `search_space/cppo.yaml` | one knob at a time, n_trials≥50 |
| **Real "absolute metric" eval** | currently reuses last eval; should be separate higher-precision re-eval | track best params during training, re-eval with bigger num_eval_envs at end |
| **Drop dead dataclass fields** | `unroll_length`, `train_step_multiplier` are 0-ref | delete from `CPPO` dataclass |
| **PURE_CPPO_CLEANUP_PROMPT.md** | written, not executed | another agent (or you) cleans up the Mava continuous file |
| **Docker / cluster manifest** | `Dockerfile` and `manifest.yaml` exist but untested with the new hydra path | smoke test on AIchor |

## 13. The user's machines

- **Local sandbox**: `/home/app/jaxgcrl/` (this session's working dir).
- **AIchor cluster**: `experiment-2e66d39a-40c9-worker-0-0` (where most real
  training happens).
- **Other remote**: `167-234-221-155:~/crl-ued-experiment-runner` (a different
  repo entirely — `instadeepai/genrl-mara-autocurricula`, branch
  `exp/cppo-jaxgcrl`). The user cherry-picks files from `Asimawad/JaxGCRL-marl`
  into that tree via `git remote add jaxgcrl-port`.

If they paste output that looks like a different repo (different branch
naming, different files in `mava/`), they're probably on the
`crl-ued-experiment-runner` machine. They'll cherry-pick what they need from
`jaxgcrl-port/<branch>` rather than rebasing.

## 14. What the user is actually optimising for

- A **publishable result**: "pure PPO + CRL critic gets X on JaxGCRL benchmark"
  with honest framing of the gap to off-policy CRL.
- **Reproducibility from a fresh `make setup`** — single command from clone to
  GPU training.
- **Mava workflow consistency** — sweep with hydra, log with MavaLogger,
  metrics in marl-eval JSON. So plotting their CPPO results next to existing
  Mava experiments works without reshaping.

If you find yourself adding code paths that *don't* serve one of these three,
ask before doing it.

## 15. Pitfalls you'll hit

- **`env=jaxnav`** without `--extra jaxnav`: ImportError on `jaxmarl`.
- **`mode: MULTIRUN`** baked into `cppo.yaml`: even single-trial runs go through
  the optuna sweeper. Use `cppo_norun.yaml` for a true single run.
- **Stale checkout**: user might be on an old commit where the env yamls aren't
  tracked (the gitignore bug from §6c). First diagnostic: `git log --oneline -3`
  and `ls mava/configs/jaxgcrl/env/`.
- **Jaxlib reinstalled by pip without execstack patch**: re-run
  `python scripts/fix_execstack.py` after any pip operation that touches
  jaxlib or nvidia-* packages.
- **`jax.devices()` returns `[CpuDevice]`**: CUDA didn't initialise. Probably
  the execstack patch wasn't applied to a freshly-installed jaxlib. Run
  `make setup` again.

## 16. Files to skim on your first 5 minutes back

In order of importance:

1. [mava/jaxgcrl/agents/cppo/cppo.py](mava/jaxgcrl/agents/cppo/cppo.py) — the
   agent. ~700 lines. Has all the algorithm logic + the bugs we fixed.
2. [mava/configs/jaxgcrl/system/cppo.yaml](mava/configs/jaxgcrl/system/cppo.yaml)
   — the recipe.
3. [goal_routing_report.md](goal_routing_report.md) — why the actor doesn't
   use the relabeled goal.
4. [mava/jaxgcrl/run.py](mava/jaxgcrl/run.py) — the hydra entry point.
5. [mava/jaxgcrl/utils/logger.py](mava/jaxgcrl/utils/logger.py) — the
   self-contained MavaLogger.
6. [mava/jaxgcrl/envs/jaxnav_adapter.py](mava/jaxgcrl/envs/jaxnav_adapter.py)
   — how MarlEnv is shimmed into a brax PipelineEnv.

If you can recite the recipe table from §5, you can pick up where we left off.

---

*Generated at end of session 2026-04-28. Update this file in subsequent
sessions when something fundamental changes (recipe, branch structure,
algorithm). Don't let it rot.*
