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

# Note this is only here until this is merged into jumanji
# PR: https://github.com/instadeepai/jumanji/pull/223

from typing import TYPE_CHECKING, Tuple, Generic, Any

import chex
import jax
from jumanji.env import State, ActionSpec
from jumanji.types import TimeStep
from jumanji.wrappers import Observation, Wrapper
if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass
import jax.numpy as jnp

from mava.types import MarlEnv, ObservableMarlEnv


class AutoResetWrapper(Wrapper):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["real_next_obs"].
    WARNING: do not `jax.vmap` the wrapped environment (e.g. do not use with the `VmapWrapper`),
    which would lead to inefficient computation due to both the `step` and `reset` functions
    being processed each time `step` is called. Please use the `VmapAutoResetWrapper` instead.
    """

    OBS_IN_EXTRAS_KEY = "real_next_obs"

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def _obs_in_extras(self, state: State, timestep: TimeStep[Observation]) -> Tuple[State, TimeStep[Observation]]:
        """Place the observation in timestep.extras[real_next_obs]."""
        extras = timestep.extras
        extras[AutoResetWrapper.OBS_IN_EXTRAS_KEY] = timestep.observation
        return state, timestep.replace(extras=extras)

    def _auto_reset(self, state: State, timestep: TimeStep[Observation]) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Save and increment episode_seed for ICRL buffer relabeling
        old_episode_seed = state.episode_seed if hasattr(state, "episode_seed") else 0

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        state, reset_timestep = self._env.reset(key)

        # Restore episode seed counter (reset() returns 0, but we need to preserve it)
        if hasattr(state, "episode_seed"):
            new_episode_seed = old_episode_seed + 1
            state = state.replace(episode_seed=new_episode_seed)

        # Place original observation in extras.
        state, timestep = self._obs_in_extras(state, timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation)  # type: ignore

        return state, timestep

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        return self._obs_in_extras(*super().reset(key))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state, timestep = self._env.step(state, action)

        # Overwrite the state and timestep appropriately if the episode terminates.
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            self._obs_in_extras,
            state,
            timestep,
        )

        return state, timestep


class DeterministicAutoResetWrapper(Wrapper):
    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: ObservableMarlEnv):
        super().__init__(env)
        self._env: ObservableMarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def step(
        self,
        state: State,
        action: chex.Array,
        state_re: State,
    ) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state_st, timestep_st = self._env.step(state, action)
        timestep_re = self._env.observe(state_re, timestep_st)
        # reset_timestep_raw = timestep_st.replace( # Commenting this out until we implement observe
        #     observation=reset_observation, extras=timestep_st.extras["env_metrics"]
        # )
        # timestep_re = self._env.modify_timestep(reset_timestep_raw)

        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(timestep_st.last(), x, y), state_re, state_st)
        timestep = jax.tree.map(lambda x, y: jax.lax.select(timestep_st.last(), x, y), timestep_re, timestep_st)

        return state, timestep


@dataclass
class DeterministicVmapAutoResetState:
    """State wrapper for DeterministicVmapAutoResetWrapper.
    
    This state stores both the current environment state and the initial
    state/timestep to reset to when episodes terminate.
    
    Attributes:
        env_state: The current environment state.
        initial_env_state: The saved initial state to reset to.
        initial_observation: The saved initial observation to reset to.
        episode_seed: Counter tracking how many episodes have completed per environment.
    """
    env_state: State
    initial_env_state: State
    initial_observation: Observation
    episode_seed: chex.Array  # shape: (num_envs,)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying environment state."""
        return getattr(self.env_state, name)


class DeterministicVmapAutoResetWrapper(
    Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
):
    """Vectorized auto-reset wrapper that resets to a fixed initial state.
    
    Similar to VmapAutoResetWrapper, but instead of generating new random states
    on episode termination, it resets to the saved initial state. This is useful for:
    - Evaluation with consistent starting conditions
    - Curriculum learning with specific initial states
    - Reproducibility testing
    
    The wrapper tracks episode_seed which increments each time an environment resets,
    allowing downstream systems to track episode boundaries.
    
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["next_obs"].
    """

    def __init__(
        self,
        env: MarlEnv,
        next_obs_in_extras: bool = True,
    ):
        """Wrap an environment for deterministic resets to initial state.
        
        Args:
            env: The environment to wrap.
            next_obs_in_extras: Whether to store the terminal observation in extras.
        """
        super().__init__(env)
        self._env: MarlEnv
        self.next_obs_in_extras = next_obs_in_extras

    def reset(self, key: chex.PRNGKey) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Reset all environments and save initial states for future resets.
        
        Args:
            key: Random keys with shape (num_envs, 2) for resetting each environment.
            
        Returns:
            state: DeterministicVmapAutoResetState containing current and initial states.
            timestep: Initial timesteps for all environments.
        """
        jax.debug.print("Called reset()")
        env_state, timestep = jax.vmap(self._env.reset)(key)
        timestep = self._maybe_add_obs_to_extras(timestep)
        
        state = DeterministicVmapAutoResetState(
            env_state=env_state,
            initial_env_state=env_state,
            initial_timestep=timestep,
            episode_seed=jnp.zeros(key.shape[0], dtype=jnp.int32),
        )
        return state, timestep

    def step(
        self, state: DeterministicVmapAutoResetState, action: chex.Array
    ) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Step all environments, resetting done ones to their initial states.
        
        Uses jax.lax.map for heterogeneous computation (conditional resets),
        following the same pattern as VmapAutoResetWrapper.
        
        Args:
            state: Current wrapper state with shape (num_envs, ...).
            action: Actions for all environments with shape (num_envs, ...).
            
        Returns:
            state: Updated wrapper state with reset environments where done.
            timestep: Timesteps with terminal observations stored in extras["next_obs"].
        """
        # Vmap homogeneous computation (step all environments in parallel)
        env_state, timestep = jax.vmap(self._env.step)(state.env_state, action)
        
        # Create intermediate state for _maybe_reset
        intermediate_state = DeterministicVmapAutoResetState(
            env_state=env_state,
            initial_env_state=state.initial_env_state,
            initial_timestep=state.initial_timestep,
            episode_seed=state.episode_seed,
        )
        
        # Map heterogeneous computation (conditional reset per environment)
        new_state, new_timestep = jax.lax.map(
            lambda args: self._maybe_reset(*args),
            (intermediate_state, timestep),
        )
        
        return new_state, new_timestep

    def _deterministic_reset(
        self, state: DeterministicVmapAutoResetState, timestep: TimeStep[Observation]
    ) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Reset to the saved initial state when episode terminates."""
        # Store terminal observation in extras
        timestep = self._maybe_add_obs_to_extras(timestep)
        
        # Replace observation with initial observation (keep reward/discount from terminal step)
        timestep = timestep.replace(observation=state.initial_timestep.observation)
        
        # Reset to initial state and increment episode seed
        new_state = DeterministicVmapAutoResetState(
            env_state=state.initial_env_state,
            initial_env_state=state.initial_env_state,
            initial_timestep=state.initial_timestep,
            episode_seed=state.episode_seed + 1,
        )
        
        return new_state, timestep

    def _maybe_reset(
        self, state: DeterministicVmapAutoResetState, timestep: TimeStep[Observation]
    ) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Conditionally reset if episode has terminated."""
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._deterministic_reset,
            lambda s, t: (s, t),
            state,
            timestep,
        )
        return state, timestep

    def set_initial_state(
        self,
        state: DeterministicVmapAutoResetState,
        new_initial_env_state: State,
        new_initial_timestep: TimeStep[Observation],
        mask: chex.Array = None,
    ) -> DeterministicVmapAutoResetState:
        """Update the initial state that environments reset to.
        
        This is useful for curriculum learning where you want to change the
        reset target dynamically during training.
        
        Args:
            state: Current wrapper state.
            new_initial_env_state: New initial environment state(s) to reset to.
            new_initial_timestep: New initial timestep(s) to reset to.
            mask: Optional boolean array (num_envs,) indicating which environments
                  to update. If None, updates all environments.
                  
        Returns:
            Updated wrapper state with new initial state/timestep.
        """
        if mask is None:
            # Update all environments
            return DeterministicVmapAutoResetState(
                env_state=state.env_state,
                initial_env_state=new_initial_env_state,
                initial_timestep=new_initial_timestep,
                episode_seed=state.episode_seed,
            )
        
        # Selectively update based on mask
        def select_by_mask(old, new):
            # Broadcast mask to match array dimensions
            broadcast_mask = mask.reshape((mask.shape[0],) + (1,) * (old.ndim - 1))
            return jnp.where(broadcast_mask, new, old)
        
        updated_initial_env_state = jax.tree.map(
            select_by_mask,
            state.initial_env_state,
            new_initial_env_state,
        )
        
        updated_initial_timestep = jax.tree.map(
            select_by_mask,
            state.initial_timestep,
            new_initial_timestep,
        )
        
        return DeterministicVmapAutoResetState(
            env_state=state.env_state,
            initial_env_state=updated_initial_env_state,
            initial_timestep=updated_initial_timestep,
            episode_seed=state.episode_seed,
        )

    def render(self, state: DeterministicVmapAutoResetState) -> Any:
        """Render the first environment state of the given batch."""
        state_0 = jax.tree.map(lambda x: x[0], state.env_state)
        return self._env.render(state_0)


class ParallelDeterministicAutoResetWrapper(
    Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
):
    """Fully parallel auto-reset wrapper that resets to a fixed initial state.
    
    Similar to DeterministicVmapAutoResetWrapper, but uses fully parallel operations
    (jnp.where/jax.lax.select) instead of heterogeneous computation (jax.lax.map).
    
    This is more efficient when:
    - Many environments terminate at the same time
    - You want to avoid the sequential overhead of jax.lax.map
    - The reset operation doesn't require different code paths
    
    The trade-off is that both the "done" and "not done" computations are always
    executed, but this can be faster than sequential processing in practice.
    
    The wrapper tracks episode_seed which increments each time an environment resets,
    allowing downstream systems to track episode boundaries.
    
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["next_obs"].
    """

    def __init__(
        self,
        env: MarlEnv,
        num_deterministic: int = 0,
    ):
        """Wrap an environment for parallel hybrid resets.
        
        The first `num_deterministic` environments reset to their saved initial state
        (deterministic), while the remaining environments reset to fresh random states
        (stochastic, like VmapAutoResetWrapper).
        
        Args:
            env: The environment to wrap.
            num_deterministic: Number of environments (starting from index 0) that
                should reset to their saved initial state. The remaining environments
                will reset to fresh random states. When 0, all envs reset randomly.
                When equal to num_envs, all envs reset deterministically.
            next_obs_in_extras: Whether to store the terminal observation in extras.
        """
        super().__init__(env)
        self._env: MarlEnv
        self.num_deterministic = num_deterministic

    def reset(self, key: chex.PRNGKey) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Reset all environments and save initial states for future resets.
        
        Args:
            key: Random keys with shape (num_envs, 2) for resetting each environment.
            
        Returns:
            state: DeterministicVmapAutoResetState containing current and initial states.
            timestep: Initial timesteps for all environments.
        """
        env_state, timestep = jax.vmap(self._env.reset)(key)

        def select_first_n(x):
            return jax.tree.map(lambda x: x[:self.num_deterministic], x)
        
        state = DeterministicVmapAutoResetState(
            env_state=env_state,
            initial_env_state=select_first_n(env_state),
            initial_observation=select_first_n(timestep.observation),
            episode_seed=jnp.zeros(key.shape[0], dtype=jnp.int32),
        )
        return state, timestep

    def step(
        self, state: DeterministicVmapAutoResetState, action: chex.Array
    ) -> Tuple[DeterministicVmapAutoResetState, TimeStep[Observation]]:
        """Step all environments with hybrid reset behavior.
        
        The first `num_deterministic` environments reset to their saved initial state,
        while the remaining environments reset to fresh random states.
        
        Uses fully parallel jnp.where operations instead of heterogeneous jax.lax.map.
        Both the "continue" and "reset" paths are computed, then selected based on done.
        
        Args:
            state: Current wrapper state with shape (num_envs, ...).
            action: Actions for all environments with shape (num_envs, ...).
            
        Returns:
            state: Updated wrapper state with reset environments where done.
            timestep: Timesteps with terminal observations stored in extras["next_obs"].
        """
        # Step all environments in parallel
        env_state, timestep = jax.vmap(self._env.step)(state.env_state, action)
        
        # Get done mask: shape (num_envs,)
        done = timestep.last()
        
        num_generated = done.shape[0] - self.num_deterministic
        if num_generated > 0:
            reset_keys = jax.random.split(state.key[0], num_generated)
            g_env_state, g_timestep = jax.vmap(self._env.reset)(jnp.stack(reset_keys))

            stitched_env_state = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), state.initial_env_state, g_env_state)
            stitched_observation = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), state.initial_observation, g_timestep.observation)
        else:
            stitched_env_state = state.initial_env_state
            stitched_observation = state.initial_observation

        @jax.vmap
        def select_by_done(done, reset_val, live_val):
            return jax.tree.map(lambda x, y: jax.lax.select(done, x, y), reset_val, live_val)

        new_env_state = select_by_done(done, stitched_env_state, env_state)
        new_observation = select_by_done(done, stitched_observation, timestep.observation)
        timestep = timestep.replace(observation=new_observation)
        
        # Increment episode_seed where done
        new_episode_seed = jnp.where(done, state.episode_seed + 1, state.episode_seed)
        
        new_state = DeterministicVmapAutoResetState(
            env_state=new_env_state,
            initial_env_state=state.initial_env_state,
            initial_observation=state.initial_observation,
            episode_seed=new_episode_seed,
        )
        
        return new_state, timestep

    def set_initial_state(
        self,
        state: DeterministicVmapAutoResetState,
        new_initial_env_state: State,
        new_initial_observation: Observation,
        mask: chex.Array = None,
    ) -> DeterministicVmapAutoResetState:
        """Update the initial state that environments reset to.
        
        This is useful for curriculum learning where you want to change the
        reset target dynamically during training.
        
        Args:
            state: Current wrapper state.
            new_initial_env_state: New initial environment state(s) to reset to.
            new_initial_observation: New initial observation(s) to reset to.
            mask: Optional boolean array (num_envs,) indicating which environments
                  to update. If None, updates all environments.
                  
        Returns:
            Updated wrapper state with new initial state/timestep.
        """
        if mask is None:
            # Update all environments
            return DeterministicVmapAutoResetState(
                env_state=state.env_state,
                initial_env_state=new_initial_env_state.env_state,
                initial_observation=new_initial_observation,
                episode_seed=state.episode_seed,
            )
        
        # Selectively update based on mask
        def select_by_mask(old, new):
            broadcast_mask = mask.reshape((mask.shape[0],) + (1,) * (old.ndim - 1))
            return jnp.where(broadcast_mask, new, old)
        
        updated_initial_env_state = jax.tree.map(
            select_by_mask,
            state.initial_env_state,
            new_initial_env_state,
        )
        
        updated_initial_timestep = jax.tree.map(
            select_by_mask,
            state.initial_observation,
            new_initial_timestep,
        )
        
        return DeterministicVmapAutoResetState(
            env_state=state.env_state,
            initial_env_state=updated_initial_env_state,
            initial_timestep=updated_initial_timestep,
            episode_seed=state.episode_seed,
        )

    def render(self, state: DeterministicVmapAutoResetState) -> Any:
        """Render the first environment state of the given batch."""
        state_0 = jax.tree.map(lambda x: x[0], state.env_state)
        return self._env.render(state_0)


@dataclass
class CRLState:
    """State of the `LogWrapper`."""

    env_state: State
    seed: int

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying environment state."""
        return getattr(self.env_state, name)


class VmapAutoResetWrapper(
    Wrapper[State, ActionSpec, Observation], Generic[State, ActionSpec, Observation]
):
    """Efficient combination of VmapWrapper and AutoResetWrapper, to be used as a replacement of
    the combination of both wrappers.
    `env = VmapAutoResetWrapper(env)` is equivalent to `env = VmapWrapper(AutoResetWrapper(env))`
    but is more efficient as it parallelizes homogeneous computation and does not run branches
    of the computational graph that are not needed (heterogeneous computation).
    - Homogeneous computation: call step function on all environments in the batch.
    - Heterogeneous computation: conditional auto-reset (call reset function for some environments
        within the batch because they have terminated).
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["next_obs"].
    """

    def reset(self, key: chex.PRNGKey) -> Tuple[CRLState, TimeStep[Observation]]:
        """Resets a batch of environments to initial states.

        The first dimension of the key will dictate the number of concurrent environments.

        To obtain a key with the right first dimension, you may call `jax.random.split` on key
        with the parameter `num` representing the number of concurrent environments.

        Args:
            key: random keys used to reset the environments where the first dimension is the number
                of desired environments.

        Returns:
            state: `State` object corresponding to the new state of the environments,
            timestep: `TimeStep` object corresponding the first timesteps returned by the
                environments,
        """
        state, timestep = jax.vmap(self._env.reset)(key)
        return CRLState(env_state=state, seed=jnp.zeros(key.shape[0], dtype=int)), timestep

    def step(self, state: CRLState, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of all environments' dynamics. It automatically resets environment(s)
        in which episodes have terminated.

        The first dimension of the state will dictate the number of concurrent environments.

        See `VmapAutoResetWrapper.reset` for more details on how to get a state of concurrent
        environments.

        Args:
            state: `State` object containing the dynamics of the environments.
            action: `Array` containing the actions to take.

        Returns:
            state: `State` object corresponding to the next states of the environments.
            timestep: `TimeStep` object corresponding the timesteps returned by the environments.
        """
        # Vmap homogeneous computation (parallelizable).
        env_state, timestep = jax.vmap(self._env.step)(state, action)
        # Map heterogeneous computation (non-parallelizable).
        state, timestep = jax.lax.map(lambda args: self._maybe_reset(*args), (CRLState(env_state=env_state, seed=state.seed), timestep))
        return state, timestep

    def _auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)
        env_state, reset_timestep = self._env.reset(key)

        # Place original observation in extras.
        timestep = self._maybe_add_obs_to_extras(timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(  # type: ignore
            observation=reset_timestep.observation
        )

        state = CRLState(env_state=env_state, seed=state.seed + 1)

        return state, timestep

    def _maybe_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Overwrite the state and timestep appropriately if the episode terminates."""
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda s, t: (s, self._maybe_add_obs_to_extras(t)),
            state,
            timestep,
        )

        return state, timestep

    def render(self, state: State) -> Any:
        """Render the first environment state of the given batch.
        The remaining elements of the batched state are ignored.

        Args:
            state: State object containing the current dynamics of the environment.
        """
        state_0 = jax.tree.map(lambda x: x[0], state)
        return super().render(state_0)