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

"""Wrapper adapting JaxGCRL Brax goal-conditioned envs to the Mava single-agent interface."""

from functools import cached_property
from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from brax.envs.base import PipelineEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart

from mava.types import Observation


@chex.dataclass
class BraxGCRLState:
    """Mava-compatible state wrapping a Brax env state."""

    env_state: Any       # brax.envs.base.State
    goal: chex.Array     # (1, goal_dim) — fixed for the episode
    key: chex.PRNGKey    # used by ParallelDeterministicAutoResetWrapper for reset keys
    step: chex.Array     # scalar int32


class BraxGCRLWrapper:
    """Adapts a JaxGCRL Brax PipelineEnv to the Mava single-agent interface.

    Brax PipelineEnv API: reset(rng) → State, step(state, action) → State
    Mava API:             reset(key) → (State, TimeStep), step(state, action) → (State, TimeStep)

    Goal encoding follows the JaxGCRL convention:
      - goal        = last goal_dim dims of the observation (target position embedded in obs)
      - achieved_goal = obs[env.goal_indices] (current agent position)

    The wrapper exposes num_agents=1, so all arrays have a leading agent dimension of 1.
    """

    def __init__(self, env: PipelineEnv, time_limit: int = 1000):
        self._env = env
        self.time_limit = time_limit
        self.num_agents = 1
        self._goal_indices = env.goal_indices   # indices into obs for achieved_goal
        self._goal_dim = int(len(env.goal_indices))
        self._action_dim = int(env.action_size)
        self._obs_dim = int(env.observation_size)

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    @cached_property
    def action_dim(self) -> int:
        return self._action_dim

    @cached_property
    def goal_dim(self) -> int:
        return self._goal_dim

    @cached_property
    def observation_spec(self) -> specs.Spec:
        agents_view = specs.Array(
            shape=(self.num_agents, self._obs_dim), dtype=jnp.float32, name="agents_view"
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, self._action_dim), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self.num_agents,), jnp.int32, 0, self.time_limit, "step_count"
        )
        return specs.Spec(
            Observation,
            name="observation_spec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Brax actions are in [-1, 1] (tanh-squashed policy output)."""
        return specs.BoundedArray(
            shape=(self.num_agents, self._action_dim),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=jnp.float32, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,), dtype=jnp.float32, minimum=0.0, maximum=1.0, name="discount"
        )

    # ------------------------------------------------------------------
    # Mava API
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[BraxGCRLState, TimeStep[Observation]]:
        key, reset_key = jax.random.split(key)
        brax_state = self._env.reset(reset_key)

        obs = brax_state.obs  # (obs_dim,)
        goal = obs[-self._goal_dim:]               # (goal_dim,)
        achieved_goal = obs[self._goal_indices]    # (goal_dim,)

        # Add agent dimension (num_agents=1)
        obs_a = obs[None, :]               # (1, obs_dim)
        goal_a = goal[None, :]             # (1, goal_dim)
        achieved_a = achieved_goal[None, :]  # (1, goal_dim)

        state = BraxGCRLState(
            env_state=brax_state,
            goal=goal_a,
            key=key,
            step=jnp.array(0, dtype=jnp.int32),
        )

        observation = Observation(
            agents_view=obs_a,
            action_mask=jnp.ones((self.num_agents, self._action_dim), dtype=bool),
            step_count=jnp.zeros((self.num_agents,), dtype=jnp.int32),
        )

        extras = {
            "env_metrics": {
                "achieved_goal": achieved_a,
                "success": jnp.zeros((self.num_agents,), dtype=jnp.float32),
                "goal_reached": jnp.zeros((self.num_agents,), dtype=jnp.float32),
            }
        }

        timestep = restart(observation, shape=(self.num_agents,), extras=extras)
        return state, timestep

    def step(
        self, state: BraxGCRLState, action: chex.Array
    ) -> Tuple[BraxGCRLState, TimeStep[Observation]]:
        # action: (1, action_dim) — squeeze agent dim for brax
        flat_action = action[0]

        brax_state = self._env.step(state.env_state, flat_action)

        obs = brax_state.obs  # (obs_dim,)
        achieved_goal = obs[self._goal_indices]  # (goal_dim,)

        # Add agent dimension
        obs_a = obs[None, :]
        achieved_a = achieved_goal[None, :]

        new_step = state.step + jnp.array(1, dtype=jnp.int32)
        done_health = brax_state.done.astype(bool)
        done_time = new_step >= self.time_limit
        done = done_health | done_time

        step_type = jax.lax.select(done, StepType.LAST, StepType.MID)
        discount = (1.0 - done.astype(jnp.float32))
        success = brax_state.metrics["success"].astype(jnp.float32)

        observation = Observation(
            agents_view=obs_a,
            action_mask=jnp.ones((self.num_agents, self._action_dim), dtype=bool),
            step_count=jnp.full((self.num_agents,), new_step, dtype=jnp.int32),
        )

        extras = {
            "env_metrics": {
                "achieved_goal": achieved_a,
                "success": jnp.broadcast_to(success, (self.num_agents,)),
                "goal_reached": jnp.broadcast_to(success, (self.num_agents,)),
            }
        }

        new_state = BraxGCRLState(
            env_state=brax_state,
            goal=state.goal,  # goal is fixed for the episode
            key=state.key,
            step=new_step,
        )

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.broadcast_to(brax_state.reward.astype(jnp.float32), (self.num_agents,)),
            discount=jnp.broadcast_to(discount, (self.num_agents,)),
            observation=observation,
            extras=extras,
        )

        return new_state, timestep
