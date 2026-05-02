# PPO-CRL on Single-Agent Jumanji Environments

**Ablation:** Wire up Jumanji Maze, Sokoban, and Sudoku as single-agent CRL tasks inside Mava's PPO-CRL framework.

## Headline Results (150 M steps, 1024×4 network)

| Environment | ABSOLUTE win rate | Notes |
|---|---|---|
| **Maze** (10×10) | **64.4%** (still rising; 300M run in progress) | A2C baseline reaches ~100% at 60M steps |
| **Sokoban** (trivial 4-box) | **50.6%** | Oscillates 4–48%; shaped reward added for next run |
| **Sudoku** (very easy DB) | pending (300M run w/ fixed goal) | Previous fill_fraction goal was broken; now uses correct_fraction |

## Environments

### Maze (`JumanjiMaze`)
- 10×10 randomly generated maze, sparse +1 reward for reaching target
- **Goal:** `achieved_goal = [agent_row/H, agent_col/W]`, `ultimate_goal = [target_row/H, target_col/W]`
- HER relabels future agent positions as goals
- Wrapper: `mava/wrappers/jumanji_maze_crl_wrapper.py`

### Sokoban (`JumanjiSokoban`)
- Fixed trivial 4-box puzzle (SimpleSolveGenerator) + randomised agent start (30 open cells)
- **Goal:** scalar distance score `= 1 - Σ min_manhattan_dist(box→target) / 8.0` ∈ [0, 1]
- Shaped reward: `solved + 0.5 × Δprop_correct_boxes` per step
- Wrapper: `mava/wrappers/jumanji_sokoban_crl_wrapper.py`

### Sudoku (`JumanjiSudoku`)
- 1000 very-easy puzzles from Jumanji DB (~9 empty cells), action space = 729
- **Goal:** `correct_fraction = (board == solution).mean()` ∈ [0.94, 1.0]
- Pre-computed solutions saved to `jumanji/data/1000_very_easy_solutions.npy`
- Wrapper: `mava/wrappers/jumanji_sudoku_crl_wrapper.py`

## How to Run

```bash
# Maze
.venv/bin/python mava/systems/icrl/anakin/ff_ppo_crl.py --config-name=ppo_crl_jumanji_maze

# Sokoban
.venv/bin/python mava/systems/icrl/anakin/ff_ppo_crl.py --config-name=ppo_crl_jumanji_sokoban

# Sudoku
.venv/bin/python mava/systems/icrl/anakin/ff_ppo_crl.py --config-name=ppo_crl_jumanji_sudoku
```

GPU serial constraint: run one at a time (parallel launches cause cuSolver OOM).
