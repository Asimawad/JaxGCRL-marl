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

from typing import Tuple, Type, TypeAlias

import chex
import jax.numpy as jnp
import gymnasium
import gymnasium as gym
import gymnasium.vector
import gymnasium.wrappers
import jaxmarl
import jumanji
import matrax
from gigastep import ScenarioBuilder
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.environments.routing.cleaner.generator import (
    RandomGenerator as CleanerRandomGenerator,
)
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorRandomGenerator,
    StochasticRandomWalkGenerator as ConnectorStochasticGenerator,
)
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from omegaconf import DictConfig

from mava.types import MarlEnv
from mava.utils.network_utils import is_gnn_based
from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    CleanerWrapper,
    ConnectorWrapper,
    GigastepWrapper,
    GymAgentIDWrapper,
    GymRecordEpisodeMetrics,
    GymToJumanji,
    LbfWrapper,
    MabraxWrapper,
    MatraxWrapper,
    MPEWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
    SmacWrapper,
    SmaxWrapper,
    UoeWrapper,
    VectorConnectorWrapper,
    async_multiagent_worker,
    ICRLVectorConnectorWrapper,
)
from mava.wrappers.icrl_goal import ICRLSmaxWrapper, ICRLMPEWrapper
from mava.wrappers.graph_wrapper import GraphWrapper
from mava.wrappers.jaxmarl import MPEGraphWrapper

from jumanji.environments.routing.connector.reward import (
    DenseRewardFn,
    RewardFn,
    SparseRewardFn,
    SharedDenseRewardFn,
    SharedSparseRewardFn,
)
from jumanji.environments.routing.connector.types import State as ConnectorState


class CompleteSparseRewardFn(RewardFn):
    """Reward all agents equally, but only when ALL agents connect simultaneously."""
    def __call__(self, state: ConnectorState, action: chex.Array, next_state: ConnectorState) -> float:
        all_connected = jnp.all(next_state.agents.connected) & ~jnp.all(state.agents.connected)
        num_agents = state.agents.id.shape[0]
        return all_connected.repeat(num_agents) * 1.0


_reward_fn_registry = {
    "dense": DenseRewardFn,
    "sparse": SparseRewardFn,
    "shared_dense": SharedDenseRewardFn,
    "shared_sparse": SharedSparseRewardFn,
    "complete_sparse": CompleteSparseRewardFn,
}

registry_type: TypeAlias = dict[str, dict[str, Type]]

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry: registry_type = {
    "RobotWarehouse": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "Connector": {"generator": ConnectorRandomGenerator, "wrapper": ConnectorWrapper},
    "VectorConnector": {
        "generator": ConnectorRandomGenerator,
        "wrapper": VectorConnectorWrapper,
        "icrl_wrapper": ICRLVectorConnectorWrapper,
    },
    "Cleaner": {"generator": CleanerRandomGenerator, "wrapper": CleanerWrapper},
}

# Registry mapping environment names directly to the corresponding wrapper classes.
_matrax_registry: registry_type = {"Matrax": {"wrapper": MatraxWrapper}}
_jaxmarl_registry: registry_type = {
    "Smax": {"wrapper": SmaxWrapper, "icrl_wrapper": ICRLSmaxWrapper},
    "MaBrax": {"wrapper": MabraxWrapper},
    "MPE": {"wrapper": MPEWrapper, "graph_wrapper": MPEGraphWrapper, "icrl_wrapper": ICRLMPEWrapper},
}
_gigastep_registry: registry_type = {"Gigastep": {"wrapper": GigastepWrapper}}

_gym_registry: registry_type = {
    "RobotWarehouse": {"wrapper": UoeWrapper},
    "LevelBasedForaging": {"wrapper": UoeWrapper},
    "SMACLite": {"wrapper": SmacWrapper},
}


def add_extra_wrappers(
    train_env: MarlEnv, eval_env: MarlEnv, config: DictConfig, registry: registry_type
) -> Tuple[MarlEnv, MarlEnv]:
    """Wrappers that access and modify observations (like AgentIDWrapper) must come before
    GraphWrapper to avoid special casing observation handling for both regular and graph
    observations. For example, AgentIDWrapper adds agent IDs to observations, which should happen
    before converting observation to GraphObservation."""
    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    # Apply AgentID wrapper BEFORE ICRLGoalWrapper to ensure correct dimension accounting
    # AgentIDWrapper prepends agent IDs to observations, which must happen before
    # ICRLGoalWrapper appends goals (otherwise dimension tracking gets confused)
    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    if is_gnn_based(config):
        # Get the graph wrapper from registry or use default GraphWrapper
        graph_wrapper = registry[config.env.env_name].get("graph_wrapper", GraphWrapper)
        train_env = graph_wrapper(train_env)
        eval_env = graph_wrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)

    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """

    # Config generator and select the wrapper.
    generator_cls = _jumanji_registry[config.env.env_name]["generator"]
    eval_generator = generator_cls(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[config.env.env_name]["wrapper"]

    # Use stochastic generator for training if deterministic_reset is enabled (SFL mode)
    use_stochastic = config.system.get("use_stochastic_generator", False)
    if use_stochastic and generator_cls is ConnectorRandomGenerator:
        train_generator = ConnectorStochasticGenerator(**config.env.scenario.task_config)
    else:
        train_generator = generator_cls(**config.env.scenario.task_config)

    reward_fn = _reward_fn_registry[config.env.reward_fn]()
    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(config.env.scenario.name, generator=train_generator, reward_fn=reward_fn, **env_config)
    eval_env = jumanji.make(config.env.scenario.name, generator=eval_generator, reward_fn=reward_fn, **env_config)

    if config.system.get("use_icrl", False):
        # Get ICRL wrapper and parameters
        icrl_wrapper = _jumanji_registry[config.env.env_name]["icrl_wrapper"]

        # First wrap with standard wrapper to get observation spec
        temp_env = wrapper(train_env, add_global_state=add_global_state)
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # ICRL parameters — goal_dim is determined by goal_type
        goal_type = config.system.icrl.get("goal_type", "distance_per_agent")
        num_agents = config.env.scenario.task_config.num_agents
        _goal_dim_lookup = {
            "total_distance": 1,
            "distance_per_agent": 1,
            "ratio_connected": 1,
            "position_coords": 2,
            "hybrid_ratio": 2,
            "hybrid_team_progress": 2,
            "connected_vector": num_agents,
            "rich": 3,
        }
        goal_dim = _goal_dim_lookup.get(goal_type, config.system.icrl.get("goal_dim", 1))
        # Set config for ICRL — no more goal_start_idx/goal_end_idx
        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

        # Win repeat: only for training (gives contrastive loss more goal=0 examples)
        win_repeat_steps = config.system.get("win_repeat_steps", 0)

        # Wrap with ICRL wrapper (goals stored as separate Observation fields)
        train_env = icrl_wrapper(
            train_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            goal_type=goal_type,
            goal_dim=goal_dim,
            win_repeat_steps=win_repeat_steps,
        )
        eval_env = icrl_wrapper(
            eval_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            goal_type=goal_type,
            goal_dim=goal_dim,
            win_repeat_steps=0,  # Never repeat in eval — clean episode boundaries
        )
    elif (
        config.system.get("deterministic_reset", False)
        and config.env.env_name == "VectorConnector"
    ):
        # SFL mode without full ICRL: use ICRL wrapper only for reset_key support.
        # ConnectorState lacks reset_key, so AutoResetWrapper(deterministic=True)
        # is a no-op with the plain VectorConnectorWrapper.
        icrl_wrapper = _jumanji_registry[config.env.env_name]["icrl_wrapper"]
        train_env = icrl_wrapper(
            train_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            win_repeat_steps=0,
        )
        eval_env = wrapper(eval_env, add_global_state=add_global_state)
    else:
        # Standard wrapper
        train_env = wrapper(train_env, add_global_state=add_global_state)
        eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _jumanji_registry)
    return train_env, eval_env


def make_jaxmarl_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a JAXMARL environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A JAXMARL environment.

    """
    kwargs = dict(config.env.kwargs)
    if "smax" in config.env.env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)
    elif "mpe" in config.env.env_name.lower():
        kwargs.update(config.env.scenario.task_config)

    use_icrl = config.system.get("use_icrl", False)
    env_name = config.env.env_name

    if use_icrl and "icrl_wrapper" in _jaxmarl_registry.get(env_name, {}):
        # ICRL mode: use ICRL wrapper with goal extraction
        icrl_wrapper_cls = _jaxmarl_registry[env_name]["icrl_wrapper"]
        standard_wrapper_cls = _jaxmarl_registry[env_name]["wrapper"]

        # ICRL parameters
        goal_type = config.system.icrl.get("goal_type", "damage_mean")

        # MPE FACMAC-style: adversary_only and max_steps overrides
        adversary_only = config.system.icrl.get("adversary_only", False)
        icrl_max_steps = config.system.icrl.get("max_steps", None)

        # Goal dimension depends on goal_type and env
        if env_name == "MPE":
            # All MPE goal types are scalar (goal_dim=1)
            if goal_type in ("distance_to_prey", "distance_to_prey_raw", "distance_to_landmark", "team_capture"):
                goal_dim = 1
            else:
                goal_dim = config.system.icrl.get("goal_dim", 1)

            # For MPE, use the ICRL wrapper itself to get base_obs_dim
            # (it handles obs padding for heterogeneous envs like Tag)
            temp_env = icrl_wrapper_cls(
                jaxmarl.make(config.env.scenario.name, **kwargs),
                has_global_state=add_global_state,
                goal_dim=goal_dim,
                goal_type=goal_type,
                win_repeat_steps=0,
                adversary_only=adversary_only,
                max_steps=icrl_max_steps,
            )
        else:
            # Get base obs dim from a temp standard wrapper
            temp_env = standard_wrapper_cls(
                jaxmarl.make(config.env.scenario.name, **kwargs), add_global_state
            )

            if goal_type == "damage_per_enemy":
                goal_dim = temp_env._env.num_enemies
            elif goal_type == "damage_stats":
                goal_dim = 4  # [mean_damage, max_damage, fraction_killed, ally_health_mean]
            else:
                goal_dim = config.system.icrl.get("goal_dim", 1)

        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # Account for agent IDs that AgentIDWrapper will prepend
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents

        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

        # Win repeat: only for training (gives contrastive loss more goal=0 examples)
        win_repeat_steps = config.system.get("win_repeat_steps", 0)

        # Extra kwargs only for MPE (adversary_only, max_steps)
        mpe_kwargs = {}
        if env_name == "MPE":
            mpe_kwargs["adversary_only"] = adversary_only
            mpe_kwargs["max_steps"] = icrl_max_steps

        # Create ICRL-wrapped envs
        train_env: MarlEnv = icrl_wrapper_cls(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            has_global_state=add_global_state,
            goal_dim=goal_dim,
            goal_type=goal_type,
            win_repeat_steps=win_repeat_steps,
            **mpe_kwargs,
        )
        eval_env: MarlEnv = icrl_wrapper_cls(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            has_global_state=add_global_state,
            goal_dim=goal_dim,
            goal_type=goal_type,
            win_repeat_steps=0,  # Never repeat in eval — clean episode boundaries
            **mpe_kwargs,
        )
    else:
        # Standard wrapper
        wrapper = _jaxmarl_registry[env_name]["wrapper"]
        train_env: MarlEnv = wrapper(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            add_global_state,
        )
        eval_env: MarlEnv = wrapper(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            add_global_state,
        )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _jaxmarl_registry)

    return train_env, eval_env


def make_matrax_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Creates Matrax environments for training and evaluation.

    Args:
    ----
        env_name: The name of the environment to create.
        config: The configuration of the environment.
        add_global_state: Whether to add the global state to the observation.

    Returns:
    -------
        A tuple containing a train and evaluation Matrax environment.

    """
    # Select the Matrax wrapper.
    wrapper = _matrax_registry[config.env.scenario.name]["wrapper"]

    # Create envs.
    task_name = config["env"]["scenario"]["task_name"]
    train_env = matrax.make(task_name, **config.env.kwargs)
    eval_env = matrax.make(task_name, **config.env.kwargs)
    train_env = wrapper(train_env, add_global_state)
    eval_env = wrapper(eval_env, add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _matrax_registry)
    return train_env, eval_env


def make_gigastep_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a Gigastep environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
    -------
        A tuple of the environments.

    """
    wrapper = _gigastep_registry[config.env.scenario.name]["wrapper"]

    kwargs = config.env.kwargs
    scenario = ScenarioBuilder.from_config(config.env.scenario.task_config)

    train_env: MarlEnv = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)
    eval_env: MarlEnv = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _gigastep_registry)
    return train_env, eval_env


def make_gym_env(
    config: DictConfig,
    num_env: int,
    add_global_state: bool = False,
) -> GymToJumanji:
    """
     Create a gymnasium environment.

    Args:
        config (Dict): The configuration of the environment.
        num_env (int) : The number of parallel envs to create.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
        Async environments.
    """
    wrapper = _gym_registry[config.env.env_name]["wrapper"]
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    def create_gym_env(config: DictConfig, add_global_state: bool = False) -> gymnasium.Env:
        registered_name = f"{config.env.scenario.name}:{config.env.scenario.task_name}"
        env = gym.make(registered_name, disable_env_checker=True, **config.env.kwargs)
        wrapped_env = wrapper(env, config.env.use_shared_rewards, add_global_state)
        if config.system.add_agent_id:
            wrapped_env = GymAgentIDWrapper(wrapped_env)
        wrapped_env = GymRecordEpisodeMetrics(wrapped_env)
        return wrapped_env

    envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: create_gym_env(config, add_global_state) for _ in range(num_env)],
        worker=async_multiagent_worker,
    )

    envs = GymToJumanji(envs)

    return envs


def make_single_train_env(
    config: DictConfig, deterministic_reset: bool = False, add_global_state: bool = False
) -> MarlEnv:
    """Create a single training env with specified autoreset mode.

    For VectorConnector, always uses ICRL wrapper for consistent observation structure
    and reset_key support, regardless of autoreset mode. This enables creating paired
    SFL (deterministic) and random (stochastic) envs with identical pytree structure.
    """
    env_name = config.env.env_name

    if env_name not in _jumanji_registry:
        raise ValueError(f"make_single_train_env only supports Jumanji envs, got {env_name}")

    generator_cls = _jumanji_registry[env_name]["generator"]
    generator = generator_cls(**config.env.scenario.task_config)

    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    raw_env = jumanji.make(config.env.scenario.name, generator=generator, **env_config)

    # Use ICRL wrapper for VectorConnector (consistent obs + reset_key in state)
    if env_name == "VectorConnector" and "icrl_wrapper" in _jumanji_registry[env_name]:
        icrl_wrapper_cls = _jumanji_registry[env_name]["icrl_wrapper"]
        wrapped = icrl_wrapper_cls(
            raw_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            win_repeat_steps=0,
        )
    else:
        wrapper_cls = _jumanji_registry[env_name]["wrapper"]
        wrapped = wrapper_cls(raw_env, add_global_state=add_global_state)

    add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
    if add_agent_id:
        wrapped = AgentIDWrapper(wrapped)

    wrapped = AutoResetWrapper(wrapped, deterministic=deterministic_reset)
    wrapped = RecordEpisodeMetrics(wrapped)

    return wrapped


def make_navix_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """Create Navix environments for training and evaluation.

    Navix environments are single-agent gridworld navigation tasks (Empty, DoorKey, FourRooms).
    The NavixWrapper adapts them to Mava's multi-agent interface with num_agents=1.

    When use_icrl=True, the wrapper populates achieved_goal/ultimate_goal fields
    with normalized agent/goal positions for CRL training.

    Args:
        config: The full experiment configuration.
        add_global_state: Unused for Navix (single-agent), kept for interface consistency.

    Returns:
        A tuple of (train_env, eval_env).
    """
    from mava.wrappers.navix_wrapper import NavixWrapper

    navix_env_name = config.env.scenario.name
    max_steps = config.env.get("time_limit", 100)
    use_icrl = config.system.get("use_icrl", False)
    goal_dim = 2  # (y, x) position

    if use_icrl:
        # Set ICRL config values for the training system
        # Create a temp env to get the obs dim
        temp_env = NavixWrapper(
            env_name=navix_env_name,
            max_steps=max_steps,
            goal_dim=goal_dim,
            use_icrl=True,
        )
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # Account for agent IDs that AgentIDWrapper will prepend
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents

        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

    train_env = NavixWrapper(
        env_name=navix_env_name,
        max_steps=max_steps,
        goal_dim=goal_dim,
        use_icrl=use_icrl,
    )
    eval_env = NavixWrapper(
        env_name=navix_env_name,
        max_steps=max_steps,
        goal_dim=goal_dim,
        use_icrl=use_icrl,
    )

    # Apply standard wrappers (AgentID, AutoReset, RecordEpisodeMetrics)
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)

    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_maze_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """Create JaxUED Maze environments for training and evaluation.

    Single-agent maze navigation with procedurally generated mazes.
    The MazeWrapper produces Mava Observation with achieved_goal (agent position)
    and ultimate_goal (goal position), both normalized by maze dimensions.

    Args:
        config: The full experiment configuration.
        add_global_state: Unused for Maze (single-agent), kept for interface consistency.

    Returns:
        A tuple of (train_env, eval_env).
    """
    from mava.wrappers.mava_maze import MazeWrapper

    height = config.env.get("height", 13)
    width = config.env.get("width", 13)
    n_walls = config.env.get("n_walls", 40)
    max_steps = config.env.get("max_steps", 250)
    agent_view_size = config.env.get("agent_view_size", 5)
    normalize_obs = config.env.get("normalize_obs", True)
    see_agent = config.env.get("see_agent", True)
    check_solvability = config.env.get("check_solvability", True)
    include_goal_in_view = config.env.get("include_goal_in_view", False)
    reward_mode = config.env.get("reward_mode", "time-penalty-sparse")
    step_penalty = config.env.get("step_penalty", 0.01)
    shaping_coef = config.env.get("shaping_coef", 0.1)

    use_icrl = config.system.get("use_icrl", False)
    goal_dim = 2  # (y/H, x/W) normalized position

    maze_kwargs = dict(
        height=height,
        width=width,
        agent_view_size=agent_view_size,
        n_walls=n_walls,
        max_steps=max_steps,
        normalize_obs=normalize_obs,
        see_agent=see_agent,
        check_solvability=check_solvability,
        include_goal_in_view=include_goal_in_view,
        reward_mode=reward_mode,
        step_penalty=step_penalty,
        shaping_coef=shaping_coef,
    )

    if use_icrl:
        # Set ICRL config values for the training system
        temp_env = MazeWrapper(**maze_kwargs)
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # Account for agent IDs that AgentIDWrapper will prepend
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents

        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

    train_env = MazeWrapper(**maze_kwargs)
    eval_env = MazeWrapper(**maze_kwargs)

    # Apply standard wrappers (AgentID, AutoReset, RecordEpisodeMetrics)
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)

    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_maze_crl_env(
    config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """Create Jumanji Maze CRL environments for training and evaluation."""
    from mava.wrappers.jumanji_maze_crl_wrapper import JumanjiMazeCRLWrapper

    num_rows = config.env.get("num_rows", 10)
    num_cols = config.env.get("num_cols", 10)
    time_limit = config.env.get("time_limit", 100)
    goal_dim = 2  # normalised 2D position [agent_row/H, agent_col/W]
    use_icrl = config.system.get("use_icrl", False)

    if use_icrl:
        temp_env = JumanjiMazeCRLWrapper(
            num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
        )
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents
        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

    train_env = JumanjiMazeCRLWrapper(
        num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
    )
    eval_env = JumanjiMazeCRLWrapper(
        num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
    )

    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_sokoban_crl_env(
    config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """Create Jumanji Sokoban CRL environments for training and evaluation."""
    from mava.wrappers.jumanji_sokoban_crl_wrapper import JumanjiSokobanCRLWrapper

    num_rows = config.env.get("num_rows", 10)
    num_cols = config.env.get("num_cols", 10)
    time_limit = config.env.get("time_limit", 120)
    goal_dim = 1  # 1 - normalized sum of min-distances from boxes to targets
    use_icrl = config.system.get("use_icrl", False)

    if use_icrl:
        temp_env = JumanjiSokobanCRLWrapper(
            num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
        )
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents
        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

    train_env = JumanjiSokobanCRLWrapper(
        num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
    )
    eval_env = JumanjiSokobanCRLWrapper(
        num_rows=num_rows, num_cols=num_cols, time_limit=time_limit
    )

    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_sudoku_crl_env(
    config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """Create Jumanji Sudoku CRL environments for training and evaluation."""
    from mava.wrappers.jumanji_sudoku_crl_wrapper import JumanjiSudokuCRLWrapper

    time_limit = config.env.get("time_limit", 100)
    goal_dim = 1  # normalised fill fraction scalar
    use_icrl = config.system.get("use_icrl", False)

    if use_icrl:
        temp_env = JumanjiSudokuCRLWrapper(time_limit=time_limit)
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents
        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

    train_env = JumanjiSudokuCRLWrapper(time_limit=time_limit)
    eval_env = JumanjiSudokuCRLWrapper(time_limit=time_limit)

    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    deterministic_reset = config.system.get("deterministic_reset", False)
    train_env = AutoResetWrapper(train_env, deterministic=deterministic_reset)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = AutoResetWrapper(eval_env, deterministic=False)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create environments for training and evaluation.

    Args:
    ----
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(config, add_global_state)
    elif env_name in _jaxmarl_registry:
        return make_jaxmarl_env(config, add_global_state)
    elif env_name in _matrax_registry:
        return make_matrax_env(config, add_global_state)
    elif env_name in _gigastep_registry:
        return make_gigastep_env(config, add_global_state)
    elif env_name == "Navix":
        return make_navix_env(config, add_global_state)
    elif env_name == "JaxUED-Maze":
        return make_maze_env(config, add_global_state)
    elif env_name == "JumanjiMaze":
        return make_jumanji_maze_crl_env(config, add_global_state)
    elif env_name == "JumanjiSokoban":
        return make_jumanji_sokoban_crl_env(config, add_global_state)
    elif env_name == "JumanjiSudoku":
        return make_jumanji_sudoku_crl_env(config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
