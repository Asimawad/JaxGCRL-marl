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

import copy
import os
import time
from typing import Any, List, Tuple

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_icrl_eval_fn, make_icrl_ff_eval_act_fn
from mava.networks import GoalEncoder, ICRLActor, SAEncoder
from mava.systems.icrl.types import ICRLParams, LearnerState, OptStates
from mava.systems.icrl.types import Transition as ICRLTransition
from mava.types import ExperimentOutput, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: MarlEnv,
    buffer: TrajectoryUniformSamplingQueue,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
) -> Any:
    """Get the learner function."""
    # Unpack apply and update functions
    sa_encoder_apply, goal_encoder_apply, actor_apply = apply_fns
    actor_update_fn, critic_update_fn, alpha_update_fn = update_fns

    # Multi-agent dimensions
    n_agents = env.num_agents
    action_dim = env.action_dim
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    num_envs_agents = config.arch.num_envs * n_agents

    # Observation dimension (base observation from environment)
    obs_dim = env.observation_spec.agents_view.shape[-1]

    # Target entropy for SAC
    target_entropy = -action_dim * config.system.target_entropy_scale

    # Logsumexp penalty coefficient
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff

    def _env_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, ICRLTransition]:
        """Step the environment for rollout_length steps."""
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        def single_step(carry, _):
            """Single environment step."""
            key, env_state, last_timestep, buffer_state = carry

            # RNG and obs
            key, noise_key = jax.random.split(key)
            obs = last_timestep.observation.agents_view  # [N_env, N_agent, obs_dim]
            gc_obs = jnp.concatenate([obs, env_state.goal], axis=-1)
            means, log_stds = actor_apply(params.actor, gc_obs)  # means: [N_env, N_agent, A]
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action_squashed = nn.tanh(x_ts)
            action = action_bias + action_scale * action_squashed

            env_state, timestep = env.step(env_state, action)

            # Flatten for buffer
            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = action.reshape(num_envs_agents, -1)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = jnp.ones((num_envs_agents, env.action_dim))
            flat_achieved_goal = timestep.extras["env_metrics"]["achieved_goal"].reshape(num_envs_agents, -1)

            # Broadcast seed and truncation to all agents
            trunc_per_env = timestep.extras.get("truncation", jnp.zeros(config.arch.num_envs, dtype=jnp.float32))
            seed_per_env = env_state.episode_seed
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            transition = ICRLTransition(
                observation=flat_obs,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras={"state_extras": {"truncation": trunc, "seed": seed}, "achieved_goal": flat_achieved_goal},
            )

            # Collect metrics at each step
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]

            return (key, env_state, timestep, buffer_state), (transition, metrics)

        # Collect transitions and metrics
        (key, env_state, last_timestep, buffer_state), (traj_batch, episode_metrics) = jax.lax.scan(
            single_step, (key, env_state, last_timestep, buffer_state), None, config.system.rollout_length
        )
        # Add trajectory to buffer (time-major format)
        buffer_state = buffer.insert(buffer_state, traj_batch)

        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, episode_metrics

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network (collect rollout + train)."""

        # Collect experience
        learner_state, episode_metrics = _env_step(learner_state, None)

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Sample batch from buffer
        key, sample_key = jax.random.split(key)
        buffer_state, transitions = buffer.sample(buffer_state)

        # Apply hindsight relabeling
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        relabled_observations = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (config.system.gamma, obs_dim), transitions, batch_keys
        )

        # Reshape with Fortran order
        relabled_observations = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            relabled_observations,
        )

        # Randomly permute transitions
        perm_key, sample_key = jax.random.split(sample_key)
        permutation = jax.random.permutation(perm_key, len(relabled_observations.observation))
        relabled_observations = jax.tree_util.tree_map(lambda x: x[permutation], relabled_observations)

        # Truncate to make evenly divisible by batch_size
        num_samples = (len(relabled_observations.observation) // config.system.batch_size) * config.system.batch_size
        relabled_observations = jax.tree_util.tree_map(lambda x: x[:num_samples], relabled_observations)

        # Reshape into batches of batch_size
        relabled_observations = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
            relabled_observations,
        )

        def _update_minibatch(carry, batch_relabled_observations):
            """Update networks on a single minibatch."""
            params, opt_states, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            # Extract obs and action for this batch
            obs = batch_relabled_observations.observation  # [batch_size, obs_dim+goal_dim]
            action = batch_relabled_observations.action  # [batch_size, action_dim]
            goal = batch_relabled_observations.goal  # [batch_size, goal_dim]

            def _critic_loss_fn(critic_params, obs, action, goal):
                """InfoNCE contrastive loss for critic."""

                # Compute representations
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], obs, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)

                # InfoNCE: compute pairwise distances with epsilon for numerical stability
                # logits[i,j] = -distance(sa_repr[i], g_repr[j])
                logits = -jnp.sqrt(
                    jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1) + 1e-8
                )  # [batch, batch]

                # InfoNCE loss: maximize diagonal (positive pairs), minimize off-diagonal
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                # Logsumexp regularization
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

                # Metrics
                logits_pos = jnp.diag(logits).mean()
                logits_neg = (logits.sum() - jnp.diag(logits).sum()) / (logits.size - logits.shape[0])
                categorical_accuracy = (logits.argmax(axis=1) == jnp.arange(logits.shape[0])).mean()

                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": logits_pos,
                    "logits_neg": logits_neg,
                    "categorical_accuracy": categorical_accuracy,
                }
                return critic_loss, loss_info

            def _actor_loss_fn(actor_params, critic_params, obs, goal, alpha, avail_actions, key):
                """Actor loss with Gumbel-Softmax."""

                actor_input = jnp.concatenate([obs, goal], axis=-1)
                means, log_stds = actor_apply(actor_params, actor_input)
                stds = jnp.exp(log_stds)

                key, noise_key = jax.random.split(key)
                noise = jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
                x_ts = means + stds * noise
                action_squashed = nn.tanh(x_ts)
                action = action_bias + action_scale * action_squashed

                # Correct Gaussian log probability (ignoring constants)
                log_prob = -0.5 * noise**2 - log_stds
                # Tanh correction (Jacobian of the transformation)
                log_prob -= jnp.log(1 - jnp.square(action_squashed) + 1e-6)
                log_prob = log_prob.sum(-1)

                # Compute Q-value (negative distance) with epsilon for numerical stability
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], obs, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)
                q_value = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1) + 1e-8)

                actor_loss = (alpha * log_prob - q_value).mean()
                entropy = -log_prob.mean()

                loss_info = {
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return actor_loss, loss_info

            def _alpha_loss_fn(log_alpha, entropy):
                """Temperature loss (matches working implementation exactly)."""
                alpha = jnp.exp(log_alpha)
                # Use stop_gradient to prevent gradient flow back to actor through entropy
                alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy)
                return alpha_loss, {"alpha_loss": alpha_loss, "alpha": alpha}

            # Update critic (both encoders)
            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = critic_grad_fn(critic_params, obs, action, goal)

            # Apply critic updates
            critic_updates, new_critic_opt_state = critic_update_fn(critic_grads, opt_states.critic)
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates["sa_encoder"])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates["goal_encoder"])

            # Update actor
            alpha = jnp.exp(params.log_alpha)
            avail_actions = batch_relabled_observations.avail_actions  # Get avail_actions mask from transitions
            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (actor_loss, actor_info), actor_grads = actor_grad_fn(
                params.actor, critic_params, obs, goal, alpha, avail_actions, actor_key
            )

            # Apply actor updates
            actor_updates, new_actor_opt_state = actor_update_fn(actor_grads, opt_states.actor)
            new_actor = optax.apply_updates(params.actor, actor_updates)

            # Update alpha (temperature)
            if config.system.icrl.learnable_temperature:
                alpha_grad_fn = jax.value_and_grad(_alpha_loss_fn, has_aux=True)
                (alpha_loss, alpha_info), alpha_grads = alpha_grad_fn(params.log_alpha, actor_info["entropy"])

                # Apply alpha updates
                alpha_updates, new_alpha_opt_state = alpha_update_fn(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
            else:
                new_log_alpha = params.log_alpha
                new_alpha_opt_state = opt_states.alpha
                alpha_info = {"alpha": alpha, "alpha_loss": 0.0}

            # Package new params and opt_states
            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=new_actor,
                log_alpha=new_log_alpha,
            )
            new_opt_states = OptStates(
                actor=new_actor_opt_state,
                critic=new_critic_opt_state,
                alpha=new_alpha_opt_state,
            )

            metrics = critic_info | actor_info | alpha_info
            return (new_params, new_opt_states, key), metrics

        # Scan over minibatches
        (params, opt_states, key), train_metrics = jax.lax.scan(
            _update_minibatch, (params, opt_states, key), relabled_observations
        )

        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput:
        """Learner function - performs multiple update steps."""
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, config.system.num_updates_per_eval
        )

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    def prefill_fn(learner_state):
        def filled_enough(ls):
            size = jnp.min(ls.buffer_state.size)
            return size >= config.system.explore_steps

        def cond(ls):
            return jnp.logical_not(filled_enough(ls))

        def body(ls):
            ls, _ = _env_step(ls, None)
            return ls

        learner_state = jax.lax.while_loop(cond, body, learner_state)
        return learner_state

    return learner_fn, prefill_fn


def learner_setup(env: MarlEnv, keys: chex.Array, config: DictConfig) -> Tuple:
    """Initialize learner_fn, networks, optimizers, buffer, and states."""
    # Get number of agents
    config.system.num_agents = env.num_agents

    # PRNG keys
    key, buffer_key, sa_key, goal_key, actor_key = keys

    # Define networks with configurable torsos from config
    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    actor_torso = instantiate(config.network.actor_network.pre_torso)

    # Create networks with configurable torsos
    sa_encoder = SAEncoder(torso=sa_encoder_torso, output_dim=64)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=64)
    actor_network = ICRLActor(torso=actor_torso, action_size=env.action_dim)

    # Initialize network parameters
    n_agents = env.num_agents

    # Create dummy inputs for network initialization
    obs_dim = env.observation_spec.agents_view.shape[-1]
    init_obs = jnp.zeros((1, obs_dim))  # Actor sees full observation
    init_action = jnp.zeros((1, env.action_dim))
    init_goal = jnp.zeros((1, env.goal_dim))
    init_actor_input = jnp.zeros((1, obs_dim + env.goal_dim))

    # SA encoder: takes state and action separately
    sa_encoder_params = sa_encoder.init(sa_key, init_obs, init_action)

    # Goal encoder: takes goal
    goal_encoder_params = goal_encoder.init(goal_key, init_goal)

    # Actor: takes full observation
    actor_params = actor_network.init(actor_key, init_actor_input)

    # Initialize log_alpha
    log_alpha = jnp.log(config.system.get("init_alpha", 0.01))  # Default to 0.01 if not specified

    # Pack parameters
    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
        log_alpha=log_alpha,
    )
    # Make opt states.
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)

    actor_opt = optax.chain(grad_clip, optax.adam(config.system.policy_lr))

    critic_opt = optax.chain(grad_clip, optax.adam(config.system.q_lr))

    alpha_opt = optax.chain(grad_clip, optax.adam(config.system.alpha_lr))

    # Create optimizers (no gradient clipping - matches original ICRL)
    critic_params_struct = {"sa_encoder": sa_encoder_params, "goal_encoder": goal_encoder_params}

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params_struct)
    alpha_opt_state = alpha_opt.init(log_alpha)

    opt_states = OptStates(
        actor=actor_opt_state,
        critic=critic_opt_state,
        alpha=alpha_opt_state,
    )

    # Create replay buffer
    num_envs_agents = config.arch.num_envs * n_agents
    # Dummy transition should match the actual transition structure
    dummy_transition = ICRLTransition(
        observation=jnp.zeros((obs_dim,)),
        action=jnp.zeros((env.action_dim,)),
        reward=0.0,
        discount=0.0,
        avail_actions=jnp.ones((env.action_dim,)),  # Default to all actions available
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}, "achieved_goal": jnp.zeros((env.goal_dim,))},
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.system.buffer_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=config.system.batch_size,
            num_envs=num_envs_agents,
            episode_length=config.env.max_steps,
        )
    )

    # Initialize buffer state
    buffer_state = jax.jit(buffer.init)(buffer_key)

    # Pack apply and update functions
    apply_fns = (sa_encoder.apply, goal_encoder.apply, actor_network.apply)
    update_fns = (actor_opt.update, critic_opt.update, alpha_opt.update)

    # Get learner function and jit it
    learn, prefill = get_learner_fn(env, buffer, apply_fns, update_fns, config)
    learn = jax.jit(learn)
    prefill = jax.jit(prefill)

    # Initialize environment states and timesteps
    key, *env_keys = jax.random.split(key, config.arch.num_envs + 1)
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    # Load model from checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # Initialize learner state (no update_batch_size dimension)
    key, step_key = jax.random.split(key)
    init_learner_state = LearnerState(params, opt_states, buffer_state, step_key, env_states, timesteps)

    return learn, prefill, actor_network, init_learner_state


def render_episode(
    env,
    actor_apply_fn,
    actor_params,
    key: chex.PRNGKey,
) -> List[Any]:
    """Run a single episode and collect rendered frames.

    Args:
        env: The environment (must have a render method).
        actor_apply_fn: The actor network apply function.
        actor_params: The actor network parameters.
        key: PRNG key for environment reset and action sampling.

    Returns:
        List of matplotlib figures for each timestep.
    """
    import matplotlib.pyplot as plt

    # Get action scaling from environment
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    # Reset environment
    key, reset_key = jax.random.split(key)
    env_state, timestep = env.reset(jnp.expand_dims(reset_key, 0))

    frames = []
    done = False
    step_count = 0
    max_steps = env.time_limit

    while not done and step_count < max_steps:
        # Render current state
        fig = env.render(env_state)
        frames.append(fig)

        # Get action from policy (deterministic - use means)
        obs = timestep.observation.agents_view
        goal = env_state.goal
        gc_obs = jnp.concatenate([obs, goal], axis=-1)

        means, _ = actor_apply_fn(actor_params, gc_obs)
        action_squashed = nn.tanh(means)
        action = action_bias + action_scale * action_squashed

        # Step environment
        env_state, timestep = env.step(env_state, action)

        done = timestep.last()
        step_count += 1

    # Render final state
    fig = env.render(env_state)
    frames.append(fig)

    return frames


def save_and_log_gif(
    frames: List[Any],
    output_dir: str,
    eval_step: int,
    logger,
    fps: int = 10,
) -> str:
    """Save frames as GIF locally and upload to Neptune logger.

    Args:
        frames: List of matplotlib figures.
        output_dir: Directory to save the GIF.
        eval_step: Current evaluation step (for naming).
        logger: MavaLogger instance.
        fps: Frames per second for the GIF.

    Returns:
        Path to the saved GIF file.
    """
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    # Convert matplotlib figures to PIL images
    pil_images = []
    for fig in frames:
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)

        # Convert to PIL image
        img = Image.open(buf)
        pil_images.append(img.copy())

        buf.close()
        plt.close(fig)

    # Save as GIF
    gif_path = os.path.join(output_dir, f"episode_render_eval_{eval_step}.gif")

    if pil_images:
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),  # duration in milliseconds
            loop=0,
        )

    # Upload to Neptune if available
    try:
        # Access the underlying Neptune logger if it exists
        if hasattr(logger, "logger") and hasattr(logger.logger, "loggers"):
            for sub_logger in logger.logger.loggers:
                if hasattr(sub_logger, "logger") and hasattr(sub_logger.logger, "__getitem__"):
                    # This is likely the Neptune logger
                    sub_logger.logger[f"eval/episode_render_{eval_step}"].upload(gif_path)
                    break
    except Exception as e:
        # Silently continue if Neptune upload fails
        pass

    return gif_path


def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "ff_icrl"
    config = copy.deepcopy(_config)

    # Create environments for train and eval
    env, eval_env = environments.make(config)

    # PRNG keys
    key, buffer_key, key_e, key_render, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=7
    )

    learn, prefill, actor_network, learner_state = learner_setup(
        env, (key, buffer_key, sa_key, goal_key, actor_key), config
    )

    learner_state = prefill(learner_state)
    jax.block_until_ready(learner_state)
    # Setup evaluator
    eval_act_fn = make_icrl_ff_eval_act_fn(actor_network.apply, eval_env)
    evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert config.system.num_updates >= config.arch.num_evaluation, (
        "Number of updates must be greater than or equal to number of evaluations."
    )

    # Calculate number of updates per evaluation
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        config.system.num_updates_per_eval * config.system.rollout_length * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    max_episode_return = -jnp.inf
    best_params = None

    # Create fixed evaluation key (same key used for every evaluation)
    fixed_eval_key = key_e[jnp.newaxis, :]  # Shape: [1, key_dim]

    # Rendering configuration
    render_every_n_evals = config.system.get("render_every_n_evals", 0)
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except ValueError:
        # HydraConfig not set (e.g., running outside of Hydra context)
        output_dir = os.getcwd()

    for eval_step in range(config.arch.num_evaluation):
        # Train
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of training
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        # Extract episode metrics, filtering for completed episodes
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Log timesteps and metrics
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation (use fixed key for reproducibility)
        trained_params = learner_output.learner_state.params.actor

        eval_metrics = evaluator(trained_params, fixed_eval_key, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # Render episode if enabled and at the right frequency
        if (render_every_n_evals > 0) and (eval_step % render_every_n_evals == 0):
            key_render, render_key = jax.random.split(key_render)
            frames = render_episode(
                eval_env,
                actor_network.apply,
                trained_params,
                render_key,
            )
            gif_path = save_and_log_gif(frames, output_dir, eval_step, logger)
            print(f"Saved episode render to: {gif_path}")

        if save_checkpoint:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_output.learner_state,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update learner state
        learner_state = learner_output.learner_state

    # Record final performance
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

        eval_metrics = abs_metric_evaluator(best_params, fixed_eval_key, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_icrl.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes
    OmegaConf.set_struct(cfg, False)

    # Run experiment
    eval_performance = run_experiment(cfg)
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
