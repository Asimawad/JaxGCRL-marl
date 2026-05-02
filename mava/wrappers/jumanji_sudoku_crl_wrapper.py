# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mava-compatible CRL wrapper for the Jumanji Sudoku environment.

Fill a 9x9 Sudoku grid correctly using discrete actions.
The 3-element Jumanji action (row, col, digit) is flattened to a single integer:
    action_int = row*81 + col*9 + digit  (action space size = 9*9*9 = 729)

Jumanji Sudoku board encoding:
  -1  = empty cell (to be filled)
  0-8 = digit placed (0-indexed, i.e. digit d+1 in standard Sudoku)

Goal (scalar, goal_dim=1):
  achieved_goal: fraction of all 81 cells matching the known solution
                 = (board == solution).mean()  ∈ [given_frac, 1.0]
  ultimate_goal: 1.0  (all cells correct)

HER relabels future achieved_goal values: the curriculum naturally progresses
from nearly-correct (high given_frac baseline) toward fully-correct (1.0).

Compatible with AutoResetWrapper and RecordEpisodeMetrics.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jumanji import specs
from jumanji.environments.logic.sudoku import Sudoku
from jumanji.environments.logic.sudoku.generator import DatabaseGenerator
from jumanji.environments.logic.sudoku.types import State as SudokuNativeState
from jumanji.types import TimeStep, StepType

from mava.types import Observation


def _load_puzzles_and_solutions() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load the very-easy puzzle DB and its pre-computed solutions."""
    import jumanji as _jmj
    data_dir = os.path.join(
        os.path.dirname(_jmj.__file__), "environments/logic/sudoku/data"
    )
    puzzles = jnp.asarray(np.load(os.path.join(data_dir, "1000_very_easy_puzzles.npy")))
    solutions = jnp.asarray(np.load(os.path.join(data_dir, "1000_very_easy_solutions.npy")))
    return puzzles, solutions  # both (1000, 9, 9) int8, values 1-9 (0=empty in puzzles)


_GRID_SIZE = 9
_ACTION_DIM = _GRID_SIZE ** 3  # 729


class _SolutionTrackingGenerator(DatabaseGenerator):
    """Extends DatabaseGenerator to expose the solution for the sampled puzzle."""

    def __init__(self) -> None:
        puzzles, solutions = _load_puzzles_and_solutions()
        super().__init__(database=puzzles)
        # Solutions in 0-indexed form matching Jumanji's board encoding (1-9 → 0-8)
        self._solutions = jnp.asarray(solutions, dtype=jnp.int32) - 1  # (1000, 9, 9)

    def sample_idx(self, key: chex.PRNGKey) -> chex.Array:
        """Reproduce DatabaseGenerator.__call__'s key split to get the puzzle index."""
        _, idx_key = jax.random.split(key)
        return jax.random.randint(idx_key, shape=(), minval=0, maxval=self._boards.shape[0])

    def get_solution(self, idx: chex.Array) -> chex.Array:
        """Return the solution for the puzzle at index idx."""
        return self._solutions[idx]  # (9, 9)


@chex.dataclass
class JumanjiSudokuCRLState:
    """Mava state wrapping Jumanji Sudoku plus HER bookkeeping."""
    board: chex.Array            # (9, 9) int32: -1=empty, 0-8=digit
    solution: chex.Array         # (9, 9) int32: 0-8, fully filled correct grid
    action_mask: chex.Array      # (9, 9, 9) bool – from Jumanji
    step_count: chex.Array       # int32 scalar
    key: chex.PRNGKey            # for AutoResetWrapper
    reset_key: chex.PRNGKey      # for deterministic autoreset
    episode_seed: chex.Array     # int32, for HER trajectory tracking


class JumanjiSudokuCRLWrapper:
    """CRL wrapper for Jumanji Sudoku with correct-fraction goal.

    Observation fields:
        agents_view  : (1, 81)   board values normalised by 8 (range [-0.125, 1])
        action_mask  : (1, 729)  flattened (9,9,9) action mask
        step_count   : (1,)
        achieved_goal: (1, 1)    (board == solution).mean() ∈ [given_frac, 1.0]
        ultimate_goal: (1, 1)    1.0
    """

    def __init__(self, time_limit: int = 100):
        self._time_limit = time_limit
        self._gen = _SolutionTrackingGenerator()
        self._env = Sudoku(generator=self._gen)

        self.num_agents = 1
        self.time_limit = time_limit
        self.action_dim = _ACTION_DIM  # 729
        self._obs_flat_dim = _GRID_SIZE * _GRID_SIZE  # 81

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_jumanji_state(self, state: JumanjiSudokuCRLState) -> SudokuNativeState:
        return SudokuNativeState(
            board=state.board,
            action_mask=state.action_mask,
            key=state.key,
        )

    @staticmethod
    def _correct_fraction(board: chex.Array, solution: chex.Array) -> chex.Array:
        """Fraction of all 81 cells whose current value matches the solution."""
        return jnp.mean((board == solution).astype(jnp.float32))

    def _make_observation(
        self,
        board: chex.Array,
        solution: chex.Array,
        action_mask: chex.Array,
        step_count: chex.Array,
    ) -> Observation:
        agents_view = (board.astype(jnp.float32) / 8.0).reshape(1, -1)  # (1, 81)
        flat_action_mask = action_mask.reshape(1, _ACTION_DIM).astype(jnp.bool_)

        frac = self._correct_fraction(board, solution)
        achieved_goal = jnp.array([[frac]], dtype=jnp.float32)   # (1, 1)
        ultimate_goal = jnp.ones((1, 1), dtype=jnp.float32)      # (1, 1)

        return Observation(
            agents_view=agents_view,
            action_mask=flat_action_mask,
            step_count=jnp.array([step_count], dtype=jnp.int32),
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[JumanjiSudokuCRLState, TimeStep]:
        key, reset_key, next_key = jax.random.split(key, 3)
        jumanji_state, _ = self._env.reset(reset_key)

        # Replicate DatabaseGenerator's key-split to get the same puzzle index
        idx = self._gen.sample_idx(reset_key)
        solution = self._gen.get_solution(idx)  # (9, 9) int32, 0-8

        step_count = jnp.int32(0)
        episode_seed = jnp.int32(0)

        observation = self._make_observation(
            jumanji_state.board,
            solution,
            jumanji_state.action_mask,
            step_count,
        )

        state = JumanjiSudokuCRLState(
            board=jumanji_state.board,
            solution=solution,
            action_mask=jumanji_state.action_mask,
            step_count=step_count,
            key=next_key,
            reset_key=key,
            episode_seed=episode_seed,
        )

        extras: Dict[str, Any] = {
            "seed": episode_seed,
            "truncation": jnp.float32(0.0),
            "env_metrics": {"won_episode": jnp.bool_(False)},
        }
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array([0.0], dtype=jnp.float32),
            discount=jnp.array([1.0], dtype=jnp.float32),
            observation=observation,
            extras=extras,
        )
        return state, timestep

    def step(
        self, state: JumanjiSudokuCRLState, action: chex.Array
    ) -> Tuple[JumanjiSudokuCRLState, TimeStep]:
        action_int = jnp.asarray(action).reshape(()).astype(jnp.int32)
        row = action_int // (_GRID_SIZE * _GRID_SIZE)
        col = (action_int // _GRID_SIZE) % _GRID_SIZE
        digit = action_int % _GRID_SIZE
        jumanji_action = jnp.array([row, col, digit], dtype=jnp.int32)

        _, next_key = jax.random.split(state.key)

        jumanji_state = self._build_jumanji_state(state)
        new_jumanji_state, jumanji_ts = self._env.step(jumanji_state, jumanji_action)

        new_step_count = state.step_count + 1

        won = jumanji_ts.reward > 0.5
        jumanji_done = jumanji_ts.last()
        bad_termination = jumanji_done & ~won
        truncated = ((new_step_count >= self._time_limit) & ~won) | bad_termination
        episode_over = won | truncated

        step_type = jnp.where(episode_over, StepType.LAST, StepType.MID)

        observation = self._make_observation(
            new_jumanji_state.board,
            state.solution,
            new_jumanji_state.action_mask,
            new_step_count,
        )

        reward = won.astype(jnp.float32).reshape(1)
        discount = jnp.where(episode_over, 0.0, 1.0).astype(jnp.float32).reshape(1)

        extras: Dict[str, Any] = {
            "seed": state.episode_seed,
            "truncation": truncated.astype(jnp.float32),
            "env_metrics": {"won_episode": won},
        }
        timestep = TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            extras=extras,
        )

        new_state = JumanjiSudokuCRLState(
            board=new_jumanji_state.board,
            solution=state.solution,
            action_mask=new_jumanji_state.action_mask,
            step_count=new_step_count,
            key=next_key,
            reset_key=state.reset_key,
            episode_seed=state.episode_seed,
        )
        return new_state, timestep

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    @cached_property
    def observation_spec(self) -> specs.Spec:
        agents_view = specs.Array(
            shape=(self.num_agents, self._obs_flat_dim),
            dtype=jnp.float32,
            name="agents_view",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_agents, self.action_dim),
            dtype=jnp.bool_,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        step_count = specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=jnp.int32,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        achieved_goal = specs.BoundedArray(
            shape=(self.num_agents, 1),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="achieved_goal",
        )
        ultimate_goal = specs.BoundedArray(
            shape=(self.num_agents, 1),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="ultimate_goal",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(num_values=self.action_dim, name="action")

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=jnp.float32, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(), dtype=jnp.float32, minimum=0.0, maximum=1.0, name="discount"
        )

    @property
    def unwrapped(self) -> "JumanjiSudokuCRLWrapper":
        return self
