# Handoff Notes — PPO-CRL Jumanji Single-Agent Ablation

**Date:** 2026-05-02  
**Repo:** Asimawad/JaxGCRL-marl  
**Branch:** mava-jumanji-single-agent-crl-2026-05-02

---

## What Was Done

Wired up three single-agent Jumanji environments (Maze, Sokoban, Sudoku) for CRL training with Mava's `ff_ppo_crl.py` system. Each environment required:
1. A custom CRL wrapper (`mava/wrappers/jumanji_*_crl_wrapper.py`) implementing `reset`, `step`, `observation_spec`, and `action_spec` compatible with `AutoResetWrapper` + `RecordEpisodeMetrics`.
2. Environment config (`mava/configs/env/jumanji_*.yaml`)
3. Experiment config (`mava/configs/default/ppo_crl_jumanji_*.yaml`)
4. System config (`mava/configs/system/icrl/ppo_crl_jumanji_*.yaml`)
5. Registration in `mava/utils/make_env.py` via `make_jumanji_*_crl_env` functions

---

## Key Design Decisions

### Maze
- **Goal:** Normalised 2D agent position `[row/H, col/W]` (2D, goal_dim=2)
- HER naturally gives curriculum: learn to reach nearby cells first
- `agents_view` = flat wall map only (100 dims); positions live in achieved/ultimate goal
- The maze is re-generated randomly each episode → Q-function must generalise across layouts

### Sokoban
- **Puzzle generator:** `_RandomStartSokobanGenerator` — keeps fixed trivial 4-box layout (each box 1 push from target) but randomises agent start over 30 open cells. This provides trajectory diversity for InfoNCE without making the task hard.
- **Goal:** scalar distance score `1 - Σ min_manhattan_dist(box→nearest_target) / 8.0`
  - Initial value ≈ 0.5 (boxes 1 step away, total dist=4, norm=8)
  - Win value = 1.0
  - Continuous change at every box movement → dense Q-function signal
- **Shaped reward:** `solved + 0.5 × Δprop_correct_boxes` — stabilises PPO against sparse win condition
- **Why not discrete/binary goals:** Tried fraction-of-boxes (5 values), binary flags (sparse), 8D box positions (bootstrapping failure). All collapsed to 0% win rate or severe oscillation. Distance-based is the stable option.

### Sudoku
- **Goal:** `correct_fraction = (board == solution).mean()` over all 81 cells
  - At reset: ~0.94 (given cells are already correct)
  - At win: 1.0
  - Each correct agent placement: +1/81 ≈ +0.012
- **Why not fill_fraction:** The original goal rewarded filling cells with ANY valid digit. Since Sudoku's action_mask already prevents constraint violations, the agent could score 100% fill_fraction without actually solving the puzzle. Win rate was stuck at ~50% (random valid moves sometimes solve very-easy puzzles).
- **Solutions pre-computed:** Ran backtracking solver on all 1000 very-easy puzzles, saved to `jumanji/environments/logic/sudoku/data/1000_very_easy_solutions.npy`. The key-split logic in `_SolutionTrackingGenerator.sample_idx` replicates `DatabaseGenerator.__call__` exactly so puzzle and solution are aligned.
- **OOM fix:** Sudoku has 729 action dims. Using 512 envs + 1024-wide network → ~182 GB OOM. Fixed with `num_envs=64`.

---

## Results Summary

### Completed Runs

**Small arch (512×4), short training:**
| Env | Steps | ABSOLUTE |
|---|---|---|
| Maze | 33M | 43.4% |
| Sokoban | 23M | 47.2% |
| Sudoku | ~130K | ~50% (stuck at baseline) |

**Large arch (1024×4), 150M steps:**
| Env | Steps | ABSOLUTE | Trend |
|---|---|---|---|
| Maze | 150M | **64.4%** | Still rising steeply at end |
| Sokoban | 150M | **50.6%** | Oscillates 4–48%, unchanged vs small arch |
| Sudoku | OOMed at 150M | — | Fixed: now uses num_envs=64 |

### In-Progress (300M, large arch, new goals/reward)
- Maze 300M: running (expected ~75–80%+ based on curve slope)
- Sokoban 300M with shaped reward: queued
- Sudoku 300M with correct_fraction: queued

### Comparison to A2C Baseline (from Jumanji paper)
| Env | A2C compute | A2C result | CRL (150M) |
|---|---|---|---|
| Maze | 60M steps | ~100% | 64.4% |
| Sokoban | 1.25B steps | partial solve | 50.6% (trivial puzzle) |
| Sudoku | 1.5B steps | ~90% | stuck at 50% (old goal), new goal pending |

---

## What Didn't Work

1. **Sokoban discrete goals** — tried 4 variants:
   - Scalar fraction-of-boxes (5 distinct values) → InfoNCE categorical accuracy ~0.006, Q-function collapses
   - Binary flags per target cell (4D) → early wins (52% peak) then catastrophic oscillation
   - 8D sorted box positions → bootstrapping failure, 0% win rate from eval 3
   - 6D hybrid (binary×4 + agent position) → 56% peak but unstable

2. **Sudoku fill_fraction goal** → rewarded wrong answers; win rate stuck at 50% (random baseline) for all 150M steps

3. **Maze ent_coef=0.05** → entropy collapsed (0.196→0.073), policy became deterministic too early, win rate plateaued 35–39%

4. **Sudoku num_envs=512 with 1024 network** → XLA OOM (~182 GB allocation). Sudoku has 729-dim action mask vs 4-dim for Maze/Sokoban.

5. **Parallel GPU training** → cuSolver internal error. Must train one environment at a time.

---

## What Should Be Tried Next

### Maze
- **Continue to 500M–1B steps.** The curve was `28% → 50% → 66%` with no plateau at 150M. At A2C's pace it should reach 80%+ by 300M.
- **Bigger eval set** — currently 128 eval envs × 200 steps = 25,600 eval episodes. For cleaner statistics at high win rates, increase to 512 eval envs.

### Sokoban
- **Verify shaped reward stabilises training** — the 300M run with `+0.5×Δprop_correct_boxes` is the first test of this. If it oscillates less, the shaping is working.
- **More diverse puzzle generator** — with only 30 starting positions and a fixed box layout, the Q-function sees highly correlated trajectories. Consider a generator that also varies box positions slightly (e.g., swap box pairs) while keeping the puzzle trivially solvable.
- **Increase DIST_NORM** — currently 8.0 (normalises initial distance=4 to 0.5). Could try 16.0 to spread goal values further from 1.0 and reduce false negatives.

### Sudoku
- **Run 300M with correct_fraction goal** — this is the key experiment. The goal now has 82 distinct levels (0/81...81/81) and each trajectory is unique. Categorical accuracy should improve substantially.
- **Increase entropy** — currently `ent_coef=0.05`. With 729 discrete actions and sparse reward, the agent might need more exploration. Try 0.1–0.2.
- **Harder puzzles** — once the easy setting works, switch to `10000_mixed_puzzles.npy` for a harder curriculum. Note: mixed puzzles have no pre-computed solutions; would need to regenerate the solver.

### General
- **Run vanilla PPO baseline** on all three envs for direct comparison — `ff_ippo.py` with the same num_envs/steps. Currently CRL results can only be compared to A2C from the Jumanji paper.
- **Tune contrastive_loss_fn** — currently `fwd_infonce`. Try `sym_infonce` (symmetric) which treats both positive directions, potentially halving false-negative rate.
- **rep_size** — currently 128. Could try 256 for Sokoban/Sudoku to give the contrastive Q-function more capacity.

---

## Production Hyperparameters (300M large-arch runs)

```yaml
# Shared across Maze + Sokoban
num_updates: 4576          # ~300M steps with num_envs=512, rollout_length=128
rollout_length: 128
batch_size: 512
ppo_epochs: 1
clip_eps: 0.1 (Sokoban) / 0.2 (Maze)
ent_coef: 0.3 (Sokoban) / 0.2 (Maze)
actor_lr: 2.5e-4
q_lr: 2.5e-4
gamma: 0.99
rep_size: 128
logsumexp_penalty_coeff: 0.1
energy_fn: norm
contrastive_loss_fn: fwd_infonce
win_repeat_steps: 5
network: ppo_crl_large  # [1024, 1024, 1024, 1024] × 3 sub-networks
num_envs: 512
num_eval_envs: 128
num_evaluation: 30
eval_steps: 200

# Sudoku specific
num_envs: 64              # OOM with 512 due to 729-dim action space
rollout_length: 128
batch_size: 512
ppo_epochs: 1
ent_coef: 0.05
lr_decay_type: cosine
```

---

## File Map

| File | Purpose |
|---|---|
| `mava/wrappers/jumanji_maze_crl_wrapper.py` | Maze env wrapper |
| `mava/wrappers/jumanji_sokoban_crl_wrapper.py` | Sokoban env wrapper (includes `_RandomStartSokobanGenerator`, shaped reward) |
| `mava/wrappers/jumanji_sudoku_crl_wrapper.py` | Sudoku env wrapper (correct_fraction goal, solution-tracking generator) |
| `mava/utils/make_env.py` | Registration of all 3 envs; `goal_dim=1` for Sokoban/Sudoku, `goal_dim=2` for Maze |
| `mava/configs/network/ppo_crl_large.yaml` | 1024×4 network config |
| `mava/configs/network/ppo_crl.yaml` | Original 512×4 network config (unchanged) |
| `mava/configs/env/jumanji_{maze,sokoban,sudoku}.yaml` | Env configs |
| `mava/configs/default/ppo_crl_jumanji_*.yaml` | Experiment entry configs |
| `mava/configs/system/icrl/ppo_crl_jumanji_*.yaml` | PPO-CRL hyperparameters per env |
| `results/maze_150m_winrates.txt` | Win-rate trajectory from Maze 150M run |
| `results/sokoban_150m_winrates.txt` | Win-rate trajectory from Sokoban 150M run |
