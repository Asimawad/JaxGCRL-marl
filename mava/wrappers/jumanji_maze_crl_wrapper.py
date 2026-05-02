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

"""Mava-compatible CRL wrapper for the Jumanji Maze environment.

Single-agent maze navigation: reach the target cell.

Goal (2D, goal_dim=2):
  achieved_goal: normalised agent position = [agent_row / H, agent_col / W]
  ultimate_goal: normalised target position = [target_row / H, target_col / W]

HER relabels the future agent position as the goal, giving the agent a natural
curriculum: first learn to navigate to nearby cells, then further ones.

Compatible with AutoResetWrapper and RecordEpisodeMetrics.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.environments.routing.maze import Maze
from jumanji.environments.routing.maze.generator import RandomGenerator
from jumanji.environments.routing.maze.types import (
    Position,
    State as JumanjiMazeNativeState,
)
from jumanji.types import TimeStep, StepType

from mava.types import Observation


@chex.dataclass
class JumanjiMazeCRLState:
    """Mava state wrapping the Jumanji Maze native state plus HER bookkeeping."""
    agent_row: chex.Array         # int32 scalar
    agent_col: chex.Array         # int32 scalar
    target_row: chex.Array        # int32 scalar
    target_col: chex.Array        # int32 scalar
    walls: chex.Array             # (H, W) bool
    native_action_mask: chex.Array  # (4,) bool
    step_count: chex.Array        # int32 scalar
    key: chex.PRNGKey             # for AutoResetWrapper
    reset_key: chex.PRNGKey       # for deterministic autoreset
    episode_seed: chex.Array      # int32, for HER trajectory tracking


class JumanjiMazeCRLWrapper:
    """CRL wrapper for Jumanji Maze.

    Observation fields:
        agents_view  : (1, H*W)  wall map only (agent/target positions are in the goals)
        action_mask  : (1, 4)    from Jumanji
        step_count   : (1,)
        achieved_goal: (1, 2)    [agent_row/H, agent_col/W] — current position
        ultimate_goal: (1, 2)    [target_row/H, target_col/W] — where to navigate to
    """

    def __init__(self, num_rows: int = 10, num_cols: int = 10, time_limit: int = 100):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._time_limit = time_limit
        self._env = Maze(generator=RandomGenerator(num_rows=num_rows, num_cols=num_cols))

        self.num_agents = 1
        self.time_limit = time_limit
        self.action_dim = 4  # up, right, down, left

        # agents_view = wall map flat only (H*W); positions come through goals
        self._obs_flat_dim = num_rows * num_cols

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_jumanji_state(self, state: JumanjiMazeCRLState) -> JumanjiMazeNativeState:
        return JumanjiMazeNativeState(
            agent_position=Position(row=state.agent_row, col=state.agent_col),
            target_position=Position(row=state.target_row, col=state.target_col),
            walls=state.walls,
            action_mask=state.native_action_mask,
            step_count=state.step_count,
            key=state.key,
        )

    def _make_observation(
        self,
        walls: chex.Array,
        agent_row: chex.Array,
        agent_col: chex.Array,
        target_row: chex.Array,
        target_col: chex.Array,
        action_mask: chex.Array,
        step_count: chex.Array,
    ) -> Observation:
        agents_view = walls.astype(jnp.float32).reshape(1, -1)  # (1, H*W)

        achieved_goal = jnp.array(
            [[agent_row / self._num_rows, agent_col / self._num_cols]],
            dtype=jnp.float32,
        )  # (1, 2)
        ultimate_goal = jnp.array(
            [[target_row / self._num_rows, target_col / self._num_cols]],
            dtype=jnp.float32,
        )  # (1, 2)

        return Observation(
            agents_view=agents_view,
            action_mask=action_mask.reshape(1, self.action_dim).astype(jnp.bool_),
            step_count=jnp.array([step_count], dtype=jnp.int32),
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[JumanjiMazeCRLState, TimeStep]:
        key, reset_key, next_key = jax.random.split(key, 3)
        jumanji_state, _ = self._env.reset(reset_key)

        step_count = jnp.int32(0)
        episode_seed = jnp.int32(0)

        observation = self._make_observation(
            jumanji_state.walls,
            jumanji_state.agent_position.row,
            jumanji_state.agent_position.col,
            jumanji_state.target_position.row,
            jumanji_state.target_position.col,
            jumanji_state.action_mask,
            step_count,
        )

        state = JumanjiMazeCRLState(
            agent_row=jumanji_state.agent_position.row,
            agent_col=jumanji_state.agent_position.col,
            target_row=jumanji_state.target_position.row,
            target_col=jumanji_state.target_position.col,
            walls=jumanji_state.walls,
            native_action_mask=jumanji_state.action_mask,
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
        self, state: JumanjiMazeCRLState, action: chex.Array
    ) -> Tuple[JumanjiMazeCRLState, TimeStep]:
        action_scalar = jnp.asarray(action).reshape(()).astype(jnp.int32)
        _, next_key = jax.random.split(state.key)

        jumanji_state = self._build_jumanji_state(state)
        new_jumanji_state, _ = self._env.step(jumanji_state, action_scalar)

        new_step_count = new_jumanji_state.step_count

        reached_goal = (
            (new_jumanji_state.agent_position.row == new_jumanji_state.target_position.row)
            & (new_jumanji_state.agent_position.col == new_jumanji_state.target_position.col)
        )
        truncated = (new_step_count >= self._time_limit) & ~reached_goal
        episode_over = reached_goal | truncated

        step_type = jnp.where(episode_over, StepType.LAST, StepType.MID)

        observation = self._make_observation(
            new_jumanji_state.walls,
            new_jumanji_state.agent_position.row,
            new_jumanji_state.agent_position.col,
            new_jumanji_state.target_position.row,
            new_jumanji_state.target_position.col,
            new_jumanji_state.action_mask,
            new_step_count,
        )

        reward = reached_goal.astype(jnp.float32).reshape(1)
        discount = jnp.where(episode_over, 0.0, 1.0).astype(jnp.float32).reshape(1)

        extras: Dict[str, Any] = {
            "seed": state.episode_seed,
            "truncation": truncated.astype(jnp.float32),
            "env_metrics": {"won_episode": reached_goal},
        }
        timestep = TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation,
            extras=extras,
        )

        new_state = JumanjiMazeCRLState(
            agent_row=new_jumanji_state.agent_position.row,
            agent_col=new_jumanji_state.agent_position.col,
            target_row=new_jumanji_state.target_position.row,
            target_col=new_jumanji_state.target_position.col,
            walls=new_jumanji_state.walls,
            native_action_mask=new_jumanji_state.action_mask,
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
            shape=(self.num_agents, 2),
            dtype=jnp.float32,
            minimum=0.0,
            maximum=1.0,
            name="achieved_goal",
        )
        ultimate_goal = specs.BoundedArray(
            shape=(self.num_agents, 2),
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
    def unwrapped(self) -> "JumanjiMazeCRLWrapper":
        return self
