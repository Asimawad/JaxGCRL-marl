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
import time
from datetime import datetime
from typing import Any, Callable, Dict, Protocol, Tuple, Union
from flax.core.frozen_dict import FrozenDict
import pickle
import numpy as np
import os
import chex
import flax
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from jax import tree
from omegaconf import DictConfig, OmegaConf
import traceback
from mava.evaluator import get_icrl_eval_fn
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import GoalEncoder
from mava.networks import SAEncoder
from mava.networks import FeedForwardActor as Actor
from mava.systems.icrl.utils import compute_contrastive_metrics, compute_logits, contrastive_loss_fn, energy_fn, flatten_crl_fn
from mava.systems.icrl.types import ICRLParams, OptStates, PQNLearnerState as LearnerState
from mava.systems.icrl.types import Transition as ICRLTransition
from mava.types import ExperimentOutput, MarlEnv, Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: MarlEnv,
    actor_apply_fn,
    sa_encoder_apply_fn,
    g_encoder_apply_fn,
    actor_update_fn,
    critic_update_fn,
    config: DictConfig,
    ) -> Callable:
    """Learner function for PPO-CRL."""

    # Multi-agent dimensions
    n_agents = env.num_agents
    action_dim = env.action_dim
    num_envs_agents = config.arch.num_envs * n_agents
    clip_eps = config.system.clip_eps
    ent_coef = config.system.ent_coef
    # Logsumexp penalty coefficient (for contrastive loss)
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    energy_fn_name = config.system.get("energy_fn", "norm")
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")

    def _env_step(learner_state: LearnerState) -> Tuple[LearnerState, Tuple[dict, ICRLTransition]]:
        """Step the environment for rollout_length steps and return trajectory."""
        params, opt_states, key, env_state, last_timestep, t = learner_state

        def single_step(carry, _):
            """Single environment step."""
            key, env_state, last_timestep, t = carry

            # RNG
            key, action_key = jax.random.split(key)

            # Get observations
            observation = last_timestep.observation
            done = last_timestep.last()
            agents_view = observation.agents_view       # [N_env, N_agent, base_obs_dim]
            achieved_goal = observation.achieved_goal   # [N_env, N_agent, goal_dim]
            ultimate_goal = observation.ultimate_goal     # [N_env, N_agent, goal_dim]
            action_mask = observation.action_mask       # [N_env, N_agent, A]
            # Build actor input: concat everything the policy needs
            actor_input = jnp.concatenate([agents_view, achieved_goal, ultimate_goal], axis=-1)
            actor_obs = Observation(
                    agents_view=actor_input,
                    action_mask=action_mask,
                    )

            # Get action from policy
            pi = actor_apply_fn(params.actor, actor_obs)
            actions = pi.sample(seed=action_key)
            log_prob = pi.log_prob(actions)
 
            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, actions)

            # Create one-hot actions
            actions_onehot = jax.nn.one_hot(actions, action_dim)

            # Flatten for processing — separate obs, achieved_goal, ultimate_goal
            flat_obs = agents_view.reshape(num_envs_agents, -1)
            flat_achieved = achieved_goal.reshape(num_envs_agents, -1)
            flat_desired = ultimate_goal.reshape(num_envs_agents, -1)
            flat_action = actions_onehot.reshape(num_envs_agents, -1)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = action_mask.reshape(num_envs_agents, -1)
            flat_log_prob = log_prob.reshape(num_envs_agents)
            flat_done = jnp.repeat(done, n_agents)
            # Broadcast seed and truncation to all agents
            trunc_per_env = timestep.extras["truncation"]
            seed_per_env = timestep.extras["seed"]
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            transition = ICRLTransition(
                observation=flat_obs,
                achieved_goal=flat_achieved,
                ultimate_goal=flat_desired,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras={"state_extras": {"truncation": trunc, "seed": seed}, "done": flat_done},
                log_prob=flat_log_prob
            )

            # Increment timestep counter
            return (key, env_state, timestep, t + config.arch.num_envs), (metrics, transition)

        # Collect transitions
        (key, env_state, last_timestep, t), (metrics, traj_batch) = jax.lax.scan(
            single_step, (key, env_state, last_timestep, t), None, config.system.rollout_length
        )
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, t)
        return learner_state, (metrics, traj_batch)

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update: collect rollout and train on it directly (no buffer)."""

        # Collect experience - returns trajectory directly
        learner_state, (episode_metrics, traj_batch) = _env_step(learner_state)

        params, opt_states, key, env_state, last_timestep, t = learner_state

        key, sample_key = jax.random.split(key)

        # traj_batch has shape (rollout_length, num_envs_agents, ...)
        # Transpose to (num_envs_agents, rollout_length, ...) for hindsight relabeling
        transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # Apply hindsight relabeling to collected trajectory
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_crl_fn, in_axes=(None, 0, 0))(
            config.system.gamma, transitions, batch_keys
        )
        # Output shape: (num_envs_agents, rollout_length-1, ...)

        # Reshape with Fortran order to flatten env-agents and time
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Truncate to make evenly divisible by batch_size
        num_samples = (len(transitions.observation) // config.system.batch_size) * config.system.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)

        def compute_crl_advantages(transitions, params, sa_encoder_apply_fn, g_encoder_apply_fn):
            # Compute Q-values by vmapping over actions (per-action SA encoder)
            def compute_q_values(obs, achieved, g_repr):
                obs_achieved = jnp.concatenate([obs, achieved], axis=-1)
                def compute_q_for_action(action_idx):
                    taken_action = jnp.full(obs.shape[:-1], action_idx)
                    a_onehot = jax.nn.one_hot(taken_action, action_dim)
                    sa_repr = sa_encoder_apply_fn(params.sa_encoder, obs_achieved, a_onehot)
                    q = energy_fn(energy_fn_name, sa_repr, g_repr)
                    return q
                q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_dim))
                return q_values


            real_ultimate_goal = transitions.extras["real_ultimate_goal"]
            g_repr = g_encoder_apply_fn(params.goal_encoder, real_ultimate_goal)
            # get q values and v values
            q_values = compute_q_values(transitions.observation, transitions.achieved_goal, g_repr)
            q_values = jnp.moveaxis(q_values, 0, -1)
            # Get policy probs
            actor_input = jnp.concatenate([transitions.observation, transitions.achieved_goal, real_ultimate_goal], axis=-1)
            actor_obs = Observation(agents_view=actor_input, action_mask=transitions.avail_actions)
            pi = actor_apply_fn(params.actor, actor_obs)
            action_probs = jax.nn.softmax(pi.distribution.logits)
            v_values = jnp.sum(action_probs * q_values, axis=-1) # [N]
            # V = expected Q under policy
            # Get Q for the action that was actually taken
            taken_action_idx = jnp.argmax(transitions.action, axis=-1) # one-hot → integer [N]
            q_taken = q_values[jnp.arange(q_values.shape[0]), taken_action_idx] # [N]

            # Advantages
            advantages = q_taken - v_values # [N]


            # Normalize
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)       # Note: transitions stay flat here; shuffling and batching happen inside each epoch
            return jax.lax.stop_gradient(advantages)
        advantages = compute_crl_advantages(transitions, params, sa_encoder_apply_fn, g_encoder_apply_fn)

        transitions = transitions._replace(extras={**transitions.extras, "advantages": advantages})
        def _update_minibatch(carry, batch_transitions):
            """Update critic on a single minibatch."""
            params, opt_states, key = carry
            key, actor_key, critic_key = jax.random.split(key, 3)

            # Extract fields for this batch
            obs = batch_transitions.observation        # [batch_size, base_obs_dim]
            achieved = batch_transitions.achieved_goal # [batch_size, goal_dim]
            ultimate_goal = batch_transitions.ultimate_goal   # [batch_size, goal_dim]
            action_onehot = batch_transitions.action   # [batch_size, action_dim]

            def _actor_loss_fn(actor_params, batch, key) -> Tuple:
                """Calculate the actor loss."""
                # Rerun network
                actor_input = jnp.concatenate([batch.observation, batch.achieved_goal, batch.extras["real_ultimate_goal"]], axis=-1)
                actor_obs = Observation(agents_view=actor_input, action_mask=batch.avail_actions)
                pi = actor_apply_fn(actor_params, actor_obs)

                # Get log prob of the TAKEN action (need integer index, not one-hot)
                taken_action_idx = jnp.argmax(batch.action, axis=-1)  # [256]
                new_log_prob = pi.distribution.log_prob(taken_action_idx)  # [256]
                # Calculate actor loss
                # ratio = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
                ratio = jnp.exp(new_log_prob - batch.log_prob)  # [256]
                advantages = batch.extras["advantages"]  # [256] — FIXED from before epochs

                # Two terms
                actor_loss1 = ratio * advantages
                actor_loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

                # Take the MINIMUM (pessimistic — PPO's conservative update)
                actor_loss = -jnp.mean(jnp.minimum(actor_loss1, actor_loss2))
                
                # The seed will be used in the TanhTransformedDistribution:
                # entropy = pi.entropy(seed=key).mean()  
                entropy = pi.distribution.entropy().mean()  # scalar
                total_actor_loss = actor_loss - ent_coef * entropy
                return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}

            def _critic_loss_fn(critic_params, obs, achieved, ultimate, action_onehot, critic_key):
                """Pure contrastive loss."""

                # SA encoder input = concat(obs, achieved_goal)
                sa_input = jnp.concatenate([obs, achieved], axis=-1)
                goal = ultimate

                # Get representation for the taken action directly
                sa_repr = sa_encoder_apply_fn(critic_params["sa_encoder"], sa_input, action_onehot)  # [batch, rep_size]

                # Get goal representation: [batch, rep_size]
                g_repr = g_encoder_apply_fn(critic_params["goal_encoder"], goal)

                # Compute pairwise logits for InfoNCE
                logits = compute_logits(energy_fn_name, sa_repr, g_repr)  # [batch, batch]

                # Compute contrastive loss
                critic_loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)

                # Metrics for monitoring
                metrics = compute_contrastive_metrics(logits)

                # Q-value for the taken action
                q_taken = energy_fn(energy_fn_name, sa_repr, g_repr)

                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": metrics["logits_pos"],
                    "logits_neg": metrics["logits_neg"],
                    "categorical_accuracy": metrics["categorical_accuracy"],
                    "logsumexp": metrics["logsumexp"],
                    "q_mean": q_taken.mean(),
                }
                return critic_loss, loss_info

            # Pack params for gradient computation
            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            actor_params = params.actor
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (actor_loss, actor_info), actor_grads = actor_grad_fn(actor_params, batch_transitions, actor_key)

            # Average gradients across devices
            actor_grads, actor_info = lax.pmean((actor_grads, actor_info), axis_name="device")

            # Apply actor updates
            actor_updates, new_actor_opt_state = actor_update_fn(actor_grads, opt_states.actor)
            new_actor = optax.apply_updates(actor_params, actor_updates)

            # Update critic
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = critic_grad_fn(critic_params, obs, achieved, ultimate_goal, action_onehot, critic_key)

            # Average gradients across devices
            critic_grads, critic_info = lax.pmean((critic_grads, critic_info), axis_name="device")

            # Apply critic updates
            critic_updates, new_critic_opt_state = critic_update_fn(critic_grads, opt_states.critic)
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates["sa_encoder"])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates["goal_encoder"])
            # === Pack results ===
            new_params = ICRLParams(
                actor=new_actor,
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
            )
            new_opt_states = OptStates(critic=new_critic_opt_state, actor=new_actor_opt_state)
            
            metrics = critic_info | actor_info
            return (new_params, new_opt_states, key), metrics

        def _update_epoch(carry, _):
            """Train for one epoch over all data (with reshuffling)."""
            params, opt_states, key, flat_transitions = carry

            # Reshuffle data for this epoch
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_transitions.observation.shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_transitions)

            # Reshape into minibatches
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                shuffled,
            )

            # Train on all minibatches
            (params, opt_states, key), epoch_metrics = jax.lax.scan(
                _update_minibatch, (params, opt_states, key), batched
            )

            return (params, opt_states, key, flat_transitions), epoch_metrics

        # Train for 1 epoch (each epoch reshuffles and trains on all data)
        (params, opt_states, key, _), train_metrics = jax.lax.scan(
            _update_epoch, (params, opt_states, key, transitions), None, config.system.ppo_epochs
        )

        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, t)
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

    return learner_fn


def learner_setup(env: MarlEnv, key: chex.PRNGKey, config: DictConfig) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states (no buffer)."""
    n_devices = len(jax.devices())
    config.system.num_agents = env.num_agents

    # PRNG keys
    key, sa_key, g_key, policy_key = jax.random.split(key, 4)

    # Define networks with configurable torsos
    actor_torso = instantiate(config.network.actor_network.pre_torso)
    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    # Create encoders (per-action SA encoder: takes (state, action_onehot))
    sa_encoder = SAEncoder(torso=sa_encoder_torso, output_dim=config.system.rep_size)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=config.system.rep_size)
    actor = Actor(torso=actor_torso, action_head=actor_action_head)

    # Initialize network parameters
    base_obs_dim = config.system.icrl.base_obs_dim
    goal_dim = config.system.icrl.goal_dim
    sa_input_dim = base_obs_dim + goal_dim  # SA encoder sees obs + achieved_goal

    # Create dummy inputs for initialization
    init_state = jnp.zeros((1, sa_input_dim))
    init_action = jnp.zeros((1, env.action_dim))
    init_goal = jnp.zeros((1, goal_dim))
    actor_input_dim = base_obs_dim + goal_dim + goal_dim # obs + achieved + goal
    init_actor_obs = Observation(
        agents_view=jnp.zeros((1, actor_input_dim)),
        action_mask=jnp.ones((1, env.action_dim)), # all actions valid for init
        )

    # Initialize encoders
    sa_encoder_params = sa_encoder.init(sa_key, init_state, init_action)
    goal_encoder_params = goal_encoder.init(g_key, init_goal)
    actor_params = actor.init(policy_key, init_actor_obs)    # Pack parameters
    
    params = ICRLParams(
        actor=actor_params,
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
    )

    # After flatten_crl_fn the last timestep is dropped, so actual samples per update:
    samples_per_update = config.arch.num_envs * env.num_agents * (config.system.rollout_length - 1)
    minibatches_per_update = samples_per_update // config.system.batch_size
    total_grad_steps = config.system.num_updates * config.system.ppo_epochs * minibatches_per_update
    config.system.total_grad_steps = total_grad_steps
    config.system.num_minibatches = minibatches_per_update
    crl_lr = make_learning_rate(config.system.q_lr, config)
    actor_lr = make_learning_rate(config.system.actor_lr, config)
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)

    # Optimizer - single optimizer for both encoders
    actor_optim = optax.chain(grad_clip,optax.adam(actor_lr, eps=1e-5))
    critic_optim = optax.chain(grad_clip, optax.adam(crl_lr))
    
    # optimizer states
    actor_optim_state = actor_optim.init(actor_params)
    critic_optim_state = critic_optim.init({"sa_encoder": sa_encoder_params, "goal_encoder": goal_encoder_params})

    opt_states = OptStates(critic=critic_optim_state, actor=actor_optim_state)

    # Get learner function and pmap it
    learn = get_learner_fn(env, actor.apply, sa_encoder.apply, goal_encoder.apply, actor_optim.update, critic_optim.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialize environment states
    key, *env_keys = jax.random.split(key, n_devices * config.arch.num_envs + 1)
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(jnp.stack(env_keys))

    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.num_envs) + x.shape[1:]
    )
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Replicate learner state across devices and batches
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices)

    # Initialize timestep counter for epsilon decay
    t0 = jnp.zeros((n_devices), dtype=jnp.int32)

    # Replicate params, opt_states
    replicate_items = (params, opt_states)
    replicate_items = flax.jax_utils.replicate(replicate_items, devices=jax.devices())

    params, opt_states = replicate_items
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, t0)

    return learn, actor.apply, init_learner_state




def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "ppo_crl"
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())
    
    # Create environments
    env, eval_env = environments.make(config)

    # PRNG keys
    key, key_e = jax.random.split(jax.random.PRNGKey(config.system.seed), num=2)
    
    learn, actor_apply_fn, learner_state = learner_setup(env, key, config)
    jax.block_until_ready(learner_state)

    # Setup evaluator
    eval_keys = jax.random.split(key_e, n_devices)
    def crl_eval_act_fn(
        params: FrozenDict, timestep, key, actor_state
    ) -> Tuple:
        obs = timestep.observation
        actor_input = jnp.concatenate(
            [obs.agents_view, obs.achieved_goal, obs.ultimate_goal], axis=-1
        )
        actor_obs = Observation(agents_view=actor_input, action_mask=obs.action_mask)
        pi = actor_apply_fn(params, actor_obs)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    evaluator = get_eval_fn(eval_env, crl_eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates must be greater than number of evaluations."

    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation 
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.num_envs
        )
    config.system.num_steps_per_evaluation_m = config.system.total_timesteps // config.arch.num_evaluation / 1e6
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))



    max_episode_return = -jnp.inf
    best_params = None
    max_win_rate = 0.0
    #  'save_args': {'save_interval_steps': 1, 'max_to_keep': 1, 'keep_period': None, 'checkpoint_uid': None, 'rel_dir': 'checkpoints'},
    checkpoint_folder = config.logger.checkpointing.save_args.rel_dir
    checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
    checkpoint_uid = config.logger.checkpointing.save_args.checkpoint_uid
    checkpoint_uid = checkpoint_uid if checkpoint_uid else datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_folder, checkpoint_uid)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hard environment rendering setup
    render_hard = config.system.get("render_hard_envs", False)
    render_dir = None
    if render_hard and checkpoint_dir:
        render_dir = os.path.join(checkpoint_dir, "hard_env_renders")
        os.makedirs(render_dir, exist_ok=True)

    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluation
        trained_params = learner_state.params.actor
          
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if config.logger.checkpointing.save_model and checkpoint_dir:
            # Get win rate for this evaluation
            win_rate = eval_metrics.get("win_rate", 0.0)
            
            # Save best model based on win rate
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                
                # Extract params (convert to numpy for pickle)
                current_params = {
                       "policy": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_state.params.actor
                    )
                }
                
                # Save best checkpoint
                best_path = os.path.join(checkpoint_dir, "best_model.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump({
                        "params": current_params,
                        "timestep": t,
                        "win_rate": float(win_rate),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, f)
                print(f"  Saved best model (win_rate={win_rate:.2f}%) to {best_path}")
            
            # Also save periodic checkpoint every 10 evals
            if eval_step % 10 == 0:
                current_params = {
                    "policy": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_state.params.actor
                    )
                }
                periodic_path = os.path.join(checkpoint_dir, f"checkpoint_step{t}.pkl")
                with open(periodic_path, "wb") as f:
                    pickle.dump({
                        "params": current_params,
                        "timestep": t,
                        "win_rate": float(win_rate),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, f)


        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        learner_state = learner_output.learner_state

    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, crl_eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key_e, n_devices)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ppo_crl.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    try:
        # Run experiment
        eval_performance = run_experiment(cfg)
        return eval_performance
    except Exception as e:
        # Log the error but don't crash the sweep
        try:
            override_info = cfg.hydra.job.override_dirname
        except:
            override_info = "unknown"
        print(f"Error executing job with overrides: {override_info}")
        print(f"Exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return -inf for maximization (worst possible value)
        # Optuna will treat this as a very poor result and continue with the next trial
        return float('-inf')

if __name__ == "__main__":
    hydra_entry_point()
