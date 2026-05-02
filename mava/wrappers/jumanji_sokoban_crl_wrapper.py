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

"""Mava-compatible CRL wrapper for the Jumanji Sokoban environment.

Push all boxes onto target squares. Discrete actions (up, right, down, left).

Uses a near-trivial 4-box puzzle (boxes adjacent to targets, each needs
a single upward push) but with a RANDOMISED agent starting position.

Randomising the start position provides trajectory diversity that CRL's
InfoNCE contrastive loss needs to learn a meaningful Q-function, while
keeping the puzzle structure trivially solvable.

Goal (scalar, goal_dim=1):
  achieved_goal: 1 - (sum of min-Manhattan-distances from each box to any target) / MAX_DIST
    = 1.0 when all boxes are on targets, ~0.5 at start, clips to 0 if boxes far away.
  ultimate_goal: 1.0

The distance-based goal changes continuously with every box movement, providing
dense Q-function training signal at every step. This yields stable, high win rates
compared to discrete (fraction, binary) goals that only change on box placements.

Compatible with AutoResetWrapper and RecordEpisodeMetrics.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.routing.sokoban import Sokoban
from jumanji.environments.routing.sokoban.generator import Generator, SimpleSolveGenerator
from jumanji.environments.routing.sokoban.types import State as SokobanNativeState
from jumanji.types import TimeStep, StepType

from mava.types import Observation

_AGENT_VALUE = 3   # variable_grid: agent marker
_BOX_VALUE = 4     # variable_grid: cell with a box
_TARGET_VALUE = 2  # fixed_grid: target cell
_N_BOXES = 4       # SimpleSolveGenerator always has 4 boxes

# Distance normalization: initial total distance = 4 (each of 4 boxes is 1 step away).
# We use 8 so the initial achieved_goal ≈ 0.5, and clip below at 0.
_DIST_NORM = 8.0

# Precomputed: min-Manhattan-distance from every cell (r,c) to the nearest target.
# Targets at row=2, cols=2-5 (verified for SimpleSolveGenerator).
_H, _W = 10, 10
_r = jnp.arange(_H, dtype=jnp.float32)[:, None] * jnp.ones((1, _W))
_c = jnp.arange(_W, dtype=jnp.float32)[None, :] * jnp.ones((_H, 1))
_CELL_TO_TARGET_DIST = jnp.min(jnp.stack([
    jnp.abs(_r - 2) + jnp.abs(_c - 2),
    jnp.abs(_r - 2) + jnp.abs(_c - 3),
    jnp.abs(_r - 2) + jnp.abs(_c - 4),
    jnp.abs(_r - 2) + jnp.abs(_c - 5),
], axis=0), axis=0)  # (H, W) float32

# Interior open cells in the SimpleSolveGenerator puzzle where agent can start.
# The puzzle layout (rows 0-9):
#   row 0,7,8,9: all walls
#   row 1: open cols 1-6
#   row 2: open cols 2-8
#   row 3: cols 2-5 have boxes; cols 6-7 open; col 8 open
#   row 4: open cols 1-5; col 7 open
#   row 5: open cols 1-2; open cols 5-7
#   row 6: open cols 1-8
# We collect all open cells (no wall, no box, no target) as candidate starts.
_OPEN_CELL_ROWS = jnp.array([
    1, 1, 1, 1, 1, 1,        # row 1, cols 1-6
    2, 2, 2, 2, 2,            # row 2, cols 4-8 (cols 2-3 are targets, 4-8 open)
    3, 3, 3,                  # row 3, cols 6-8 (2-5 boxes, 6-8 open)
    4, 4, 4, 4, 4,            # row 4, cols 1-4, 7 (5 and 6 have wall/open issues)
    5, 5, 5, 5,               # row 5, cols 1-2, 5-6
    6, 6, 6, 6, 6, 6, 6, 6,  # row 6, cols 1-8
], dtype=jnp.int32)
_OPEN_CELL_COLS = jnp.array([
    1, 2, 3, 4, 5, 6,         # row 1
    4, 5, 6, 7, 8,             # row 2
    6, 7, 8,                   # row 3
    1, 2, 3, 4, 7,             # row 4
    1, 2, 5, 6,                # row 5
    1, 2, 3, 4, 5, 6, 7, 8,  # row 6
], dtype=jnp.int32)


class _RandomStartSokobanGenerator(Generator):
    """Sokoban generator: fixed SimpleSolveGenerator puzzle + random agent start.

    The box/target layout is always the same (4 boxes, each one step from its
    target). Only the agent's starting cell varies each reset, providing the
    trajectory diversity that CRL's contrastive loss needs.
    """

    def __init__(self) -> None:
        super().__init__()
        # Build the base puzzle once (boxes and targets are always the same)
        _gen = SimpleSolveGenerator()
        import jax as _jax
        base_state = _gen(_jax.random.PRNGKey(0))
        # fixed_grid: walls + targets (never changes)
        self._fixed_grid = base_state.fixed_grid
        # variable_grid template: boxes at fixed positions, agent removed
        var_without_agent = jnp.where(
            base_state.variable_grid == _AGENT_VALUE,
            0,
            base_state.variable_grid,
        )
        self._var_template = var_without_agent  # has boxes but no agent

        self._n_open = len(_OPEN_CELL_ROWS)

    def __call__(self, rng_key: chex.PRNGKey) -> SokobanNativeState:
        rng_key, k1 = jax.random.split(rng_key)
        idx = jax.random.randint(k1, shape=(), minval=0, maxval=self._n_open)
        agent_row = _OPEN_CELL_ROWS[idx]
        agent_col = _OPEN_CELL_COLS[idx]

        variable_grid = self._var_template.at[agent_row, agent_col].set(_AGENT_VALUE)
        agent_location = jnp.array([agent_row, agent_col], dtype=jnp.int32)

        return SokobanNativeState(
            key=rng_key,
            fixed_grid=self._fixed_grid,
            variable_grid=variable_grid,
            agent_location=agent_location,
            step_count=jnp.int32(0),
        )


_SHAPING_COEF = 0.5  # reward bonus per unit improvement in prop_correct_boxes


@chex.dataclass
class JumanjiSokobanCRLState:
    """Mava state wrapping Jumanji Sokoban plus HER bookkeeping."""
    fixed_grid: chex.Array         # (H, W) int32 – immutable per episode
    variable_grid: chex.Array      # (H, W) int32 – changes each step
    agent_row: chex.Array          # int32 scalar
    agent_col: chex.Array          # int32 scalar
    step_count: chex.Array         # int32 scalar
    prop_correct_boxes: chex.Array # float32 scalar – for shaped reward delta
    key: chex.PRNGKey              # for AutoResetWrapper
    reset_key: chex.PRNGKey        # for deterministic autoreset
    episode_seed: chex.Array       # int32, for HER trajectory tracking


class JumanjiSokobanCRLWrapper:
    """CRL wrapper for Jumanji Sokoban with randomised agent start position.

    Observation fields:
        agents_view  : (1, H*W*2 + 2)  variable_grid flat + fixed_grid flat + agent pos
        action_mask  : (1, 4)          all-True (no movement restriction)
        step_count   : (1,)
        achieved_goal: (1, 1)  distance score = 1 - total_min_dist_to_targets / _DIST_NORM
        ultimate_goal: (1, 1)  1.0
    """

    def __init__(self, num_rows: int = 10, num_cols: int = 10, time_limit: int = 120):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._time_limit = time_limit
        self._env = Sokoban(generator=_RandomStartSokobanGenerator())

        self.num_agents = 1
        self.time_limit = time_limit
        self.action_dim = 4

        self._obs_flat_dim = num_rows * num_cols * 2 + 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_jumanji_state(self, state: JumanjiSokobanCRLState) -> SokobanNativeState:
        return SokobanNativeState(
            key=state.key,
            fixed_grid=state.fixed_grid,
            variable_grid=state.variable_grid,
            agent_location=jnp.array([state.agent_row, state.agent_col], dtype=jnp.int32),
            step_count=state.step_count,
        )

    @staticmethod
    def _dist_score(variable_grid: chex.Array) -> chex.Array:
        """Normalised box-to-target distance score: 1.0 = all on targets, ~0.5 at start."""
        box_mask = (variable_grid == _BOX_VALUE)
        total_dist = jnp.sum(jnp.where(box_mask, _CELL_TO_TARGET_DIST, 0.0))
        return jnp.clip(1.0 - total_dist / _DIST_NORM, 0.0, 1.0)

    def _make_observation(
        self,
        fixed_grid: chex.Array,
        variable_grid: chex.Array,
        agent_row: chex.Array,
        agent_col: chex.Array,
        step_count: chex.Array,
    ) -> Observation:
        var_flat = variable_grid.astype(jnp.float32).reshape(1, -1) / 4.0
        fix_flat = fixed_grid.astype(jnp.float32).reshape(1, -1) / 2.0
        agent_pos = jnp.array(
            [agent_row / self._num_rows, agent_col / self._num_cols], dtype=jnp.float32
        ).reshape(1, 2)
        agents_view = jnp.concatenate([var_flat, fix_flat, agent_pos], axis=-1)

        score = self._dist_score(variable_grid)                      # scalar
        achieved_goal = jnp.array([[score]], dtype=jnp.float32)      # (1, 1)
        ultimate_goal = jnp.ones((1, 1), dtype=jnp.float32)         # (1, 1)

        action_mask = jnp.ones((1, self.action_dim), dtype=jnp.bool_)
        sc = jnp.array([step_count], dtype=jnp.int32)

        return Observation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=sc,
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[JumanjiSokobanCRLState, TimeStep]:
        key, reset_key, next_key = jax.random.split(key, 3)
        jumanji_state, _ = self._env.reset(reset_key)

        step_count = jnp.int32(0)
        episode_seed = jnp.int32(0)

        observation = self._make_observation(
            jumanji_state.fixed_grid,
            jumanji_state.variable_grid,
            jumanji_state.agent_location[0],
            jumanji_state.agent_location[1],
            step_count,
        )

        state = JumanjiSokobanCRLState(
            fixed_grid=jumanji_state.fixed_grid,
            variable_grid=jumanji_state.variable_grid,
            agent_row=jumanji_state.agent_location[0],
            agent_col=jumanji_state.agent_location[1],
            step_count=step_count,
            prop_correct_boxes=jnp.float32(0.0),
            key=next_key,
            reset_key=key,
            episode_seed=episode_seed,
        )

        extras: Dict[str, Any] = {
            "seed": episode_seed,
            "truncation": jnp.float32(0.0),
            "env_metrics": {
                "won_episode": jnp.bool_(False),
                "prop_correct_boxes": jnp.float32(0.0),
            },
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
        self, state: JumanjiSokobanCRLState, action: chex.Array
    ) -> Tuple[JumanjiSokobanCRLState, TimeStep]:
        action_scalar = jnp.asarray(action).reshape(()).astype(jnp.int32)
        _, next_key = jax.random.split(state.key)

        jumanji_state = self._build_jumanji_state(state)
        new_jumanji_state, jumanji_ts = self._env.step(jumanji_state, action_scalar)

        new_step_count = new_jumanji_state.step_count
        solved = jumanji_ts.extras["solved"]
        prop_correct = jumanji_ts.extras["prop_correct_boxes"]

        truncated = (new_step_count >= self._time_limit) & ~solved
        episode_over = solved | truncated

        step_type = jnp.where(episode_over, StepType.LAST, StepType.MID)

        observation = self._make_observation(
            new_jumanji_state.fixed_grid,
            new_jumanji_state.variable_grid,
            new_jumanji_state.agent_location[0],
            new_jumanji_state.agent_location[1],
            new_step_count,
        )

        # Shaped reward: sparse solve bonus + delta prop_correct_boxes * coef
        delta_prop = prop_correct - state.prop_correct_boxes
        shaped = solved.astype(jnp.float32) + delta_prop * _SHAPING_COEF
        reward = shaped.reshape(1)
        discount = jnp.where(episode_over, 0.0, 1.0).astype(jnp.float32).reshape(1)

        extras: Dict[str, Any] = {
            "seed": state.episode_seed,
            "truncation": truncated.astype(jnp.float32),
            "env_metrics": {
                "won_episode": solved,
                "prop_correct_boxes": prop_correct,
            },
        }
        timestep = TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            extras=extras,
        )

        new_state = JumanjiSokobanCRLState(
            fixed_grid=new_jumanji_state.fixed_grid,
            variable_grid=new_jumanji_state.variable_grid,
            agent_row=new_jumanji_state.agent_location[0],
            agent_col=new_jumanji_state.agent_location[1],
            step_count=new_step_count,
            prop_correct_boxes=prop_correct,
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
    def unwrapped(self) -> "JumanjiSokobanCRLWrapper":
        return self
