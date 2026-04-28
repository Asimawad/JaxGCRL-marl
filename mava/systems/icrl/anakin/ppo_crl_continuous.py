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

"""PPO-CRL for continuous action spaces (e.g., JaxNav).

Key idea: PPO actor update + CRL contrastive critic, with Monte Carlo
V(s,g) estimation for advantage computation. The SA encoder takes
raw continuous actions (not one-hot).
"""

import copy
import time
from typing import Any, Dict, Tuple

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

from mava.evaluator import get_icrl_eval_fn
from mava.networks import GoalEncoder, ICRLActor, ICRLValueNet, SAEncoder
from mava.systems.icrl.losses import compute_contrastive_metrics, compute_logits, contrastive_loss_fn, energy_fn
from mava.systems.icrl.types import ICRLParams, OptStates
from mava.systems.icrl.types import Transition as ICRLTransition
from mava.types import ExperimentOutput, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics


def make_eval_act_fn(actor_apply_fn, env, deterministic=False, use_achieved_goal=True):
    """Make eval act function. Stochastic by default (matches training distribution)."""
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    def eval_act_fn(params, timestep, key, goal):
        obs_view = timestep.observation.agents_view
        if use_achieved_goal:
            current_ag = timestep.extras["env_metrics"]["achieved_goal"]
            while current_ag.ndim < obs_view.ndim:
                current_ag = current_ag[..., None]
            gc_obs = jnp.concatenate([obs_view, current_ag, goal], axis=-1)
        else:
            gc_obs = jnp.concatenate([obs_view, goal], axis=-1)
        means, log_stds = actor_apply_fn(params, gc_obs)
        if deterministic:
            action = action_bias + action_scale * nn.tanh(means)
        else:
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action = action_bias + action_scale * nn.tanh(x_ts)
        return action, {}

    return eval_act_fn


class PPOCRLLearnerState(NamedTuple):
    """Learner state for PPO-CRL continuous."""

    params: ICRLParams
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: Any
    last_timestep: Any
    update_count: chex.Array  # scalar int for entropy scheduling
    log_alpha: chex.Array  # learnable log entropy coefficient
    alpha_opt_state: Any  # optimizer state for log_alpha
    value_params: Any  # value network params (None if use_gae=False)
    value_opt_state: Any  # value optimizer state (None if use_gae=False)


class PPOTransition(NamedTuple):
    """Transition for PPO-CRL with log probs for importance sampling."""

    observation: chex.Array  # agent obs (without goal appended)
    action: chex.Array  # continuous action (scaled for env & SA encoder)
    x_t: chex.Array  # pre-tanh value (for stable log_prob recomputation)
    log_prob: chex.Array  # Gaussian log probability in pre-tanh space
    reward: chex.Array
    discount: chex.Array
    avail_actions: chex.Array
    value: chex.Array  # V(s,g) prediction for GAE (0 if use_gae=False)
    env_goal: chex.Array  # env goal at this step (for GAE actor training)
    extras: Dict[str, Any]


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
) -> Any:
    """Get the learner function for PPO-CRL continuous."""

    _sa_encoder_apply, goal_encoder_apply, actor_apply, value_apply = apply_fns

    def sa_encoder_apply(params, obs, action):
        return _sa_encoder_apply(params, obs, action * sa_action_scale)
    actor_update_fn, critic_update_fn, value_update_fn = update_fns

    n_agents = env.num_agents
    action_dim = env.action_dim
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    num_envs_agents = config.arch.num_envs * n_agents

    obs_dim = env.observation_spec.agents_view.shape[-1]

    # CRL config
    gamma = config.system.gamma
    energy_fn_name = config.system.get("energy_fn", "norm")
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "sym_infonce")
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    sa_action_scale = config.system.get("sa_action_scale", 1.0)
    contrastive_temperature = config.system.get("contrastive_temperature", 1.0)

    # PPO config
    clip_eps = config.system.clip_eps
    ent_coef_start = config.system.ent_coef
    ent_coef_end = config.system.get("ent_coef_end", config.system.ent_coef)
    ent_schedule_horizon = config.system.get("ent_schedule_horizon", None) or config.system.num_updates
    num_updates_total = config.system.num_updates
    num_mc_samples = config.system.num_mc_samples
    num_epochs = config.system.get("num_epochs", 4)
    target_entropy = config.system.get("target_entropy", 0.0)
    entropy_floor_coeff = config.system.get("entropy_floor_coeff", 0.0)
    use_adaptive_entropy = config.system.get("use_adaptive_entropy", False)
    alpha_lr = config.system.get("alpha_lr", 3e-4)
    use_reinforce = config.system.get("use_reinforce", False)  # Use Q directly as advantage (no V baseline)
    mc_std_floor = config.system.get("mc_std_floor", 0.0)  # Min std for MC V estimation (prevents vanishing advantages)
    num_critic_warmup_epochs = config.system.get("num_critic_warmup_epochs", 0)
    use_achieved_goal = config.system.get("use_achieved_goal", True)
    reward_advantage_coeff = config.system.get("reward_advantage_coeff", 0.0)
    use_gae = config.system.get("use_gae", False)
    gae_lambda = config.system.get("gae_lambda", 0.95)
    vf_coef = config.system.get("vf_coef", 0.5)

    def _env_step(
        learner_state: PPOCRLLearnerState, _: Any
    ) -> Tuple[PPOCRLLearnerState, Tuple[dict, PPOTransition]]:
        """Step the environment for rollout_length steps."""
        params, opt_states, key, env_state, last_timestep, update_count, log_alpha, alpha_opt_state, value_params, value_opt_state = learner_state

        def single_step(carry, _):
            key, env_state, last_timestep = carry
            key, noise_key = jax.random.split(key)

            obs = last_timestep.observation.agents_view  # [N_env, N_agent, obs_dim]
            if use_achieved_goal:
                current_ag = last_timestep.extras["env_metrics"]["achieved_goal"]
                while current_ag.ndim < obs.ndim:
                    current_ag = current_ag[..., None]
                gc_obs = jnp.concatenate([obs, current_ag, env_state.goal], axis=-1)
            else:
                current_ag = None
                gc_obs = jnp.concatenate([obs, env_state.goal], axis=-1)

            means, log_stds = actor_apply(params.actor, gc_obs)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action_squashed = nn.tanh(x_ts)
            action = action_bias + action_scale * action_squashed

            # Compute Gaussian log probability in pre-tanh space (stable for PPO)
            log_prob = -0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = log_prob.sum(-1)  # [N_env, N_agent]

            # Save env goal before stepping (matches gc_obs used for actor/value)
            pre_step_goal = env_state.goal

            env_state, timestep = env.step(env_state, action)

            # Flatten for processing
            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = action.reshape(num_envs_agents, -1)
            flat_x_t = x_ts.reshape(num_envs_agents, -1)
            flat_log_prob = log_prob.reshape(num_envs_agents)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = jnp.ones((num_envs_agents, action_dim))
            flat_achieved_goal = timestep.extras["env_metrics"]["achieved_goal"].reshape(
                num_envs_agents, -1
            )

            # Value prediction for GAE (uses same input as actor: obs + [ag] + goal)
            if use_gae:
                flat_value = value_apply(value_params, gc_obs.reshape(num_envs_agents, -1))
            else:
                flat_value = jnp.zeros(num_envs_agents)

            # Store env goal for actor training with GAE (pre-step goal matches gc_obs)
            flat_env_goal = pre_step_goal.reshape(num_envs_agents, -1)

            # Broadcast seed and truncation to all agents
            trunc_per_env = timestep.extras.get(
                "truncation", jnp.zeros(config.arch.num_envs, dtype=jnp.float32)
            )
            seed_per_env = env_state.episode_seed
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            extras = {
                "state_extras": {"truncation": trunc, "seed": seed},
                "achieved_goal": flat_achieved_goal,
            }
            if use_achieved_goal:
                flat_current_ag = current_ag.reshape(num_envs_agents, -1)
                extras["current_achieved_goal"] = flat_current_ag

            transition = PPOTransition(
                observation=flat_obs,
                action=flat_action,
                x_t=flat_x_t,
                log_prob=flat_log_prob,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                value=flat_value,
                env_goal=flat_env_goal,
                extras=extras,
            )

            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]

            return (key, env_state, timestep), (transition, metrics)

        (key, env_state, last_timestep), (traj_batch, episode_metrics) = jax.lax.scan(
            single_step,
            (key, env_state, last_timestep),
            None,
            config.system.rollout_length,
        )

        learner_state = PPOCRLLearnerState(params, opt_states, key, env_state, last_timestep, update_count, log_alpha, alpha_opt_state, value_params, value_opt_state)
        return learner_state, (episode_metrics, traj_batch)

    def _flatten_crl(buffer_config, transition, sample_key):
        """Hindsight relabeling for on-policy CRL data.

        Takes a trajectory of transitions and relabels goals using future achieved goals.
        Returns flat arrays ready for minibatch training.
        """
        gamma, base_obs_dim = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(
            arrangement[:, None] < arrangement[None], dtype=jnp.float32
        )
        discount = gamma ** jnp.array(
            arrangement[None] - arrangement[:, None], dtype=jnp.float32
        )
        probs = is_future_mask * discount

        # Mask by episode boundaries (same seed = same episode)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )
        seed_mask = jnp.equal(single_trajectories, single_trajectories.T)
        probs = probs * seed_mask + jnp.eye(seq_len) * 1e-5

        # Sample future goals
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))
        goal = jnp.take(
            transition.extras["achieved_goal"], goal_index[:-1], axis=0
        )

        result = {
            "observation": transition.observation[:-1],
            "goal": goal,
            "action": transition.action[:-1],
            "x_t": transition.x_t[:-1],
            "old_log_prob": transition.log_prob[:-1],
            "reward": transition.reward[:-1],
        }
        if use_achieved_goal:
            result["current_achieved_goal"] = transition.extras["current_achieved_goal"][:-1]
        return result

    def _update_step(
        learner_state: PPOCRLLearnerState, _: Any
    ) -> Tuple[PPOCRLLearnerState, Tuple]:
        """Collect rollout and train."""

        learner_state, (episode_metrics, traj_batch) = _env_step(learner_state, None)

        params, opt_states, key, env_state, last_timestep, update_count, log_alpha, alpha_opt_state, value_params, value_opt_state = learner_state

        # Entropy coefficient from learnable log_alpha (if adaptive) or from schedule
        if use_adaptive_entropy:
            ent_coef = jnp.exp(log_alpha)
        else:
            frac = jnp.clip(update_count / jnp.maximum(ent_schedule_horizon, 1), 0.0, 1.0)
            ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * frac

        key, sample_key = jax.random.split(key)

        # Compute discounted returns from env rewards (for reward-based advantages)
        if reward_advantage_coeff > 0:
            def _scan_returns(next_return, step_data):
                reward, discount = step_data
                current_return = reward + gamma * discount * next_return
                return current_return, current_return

            _, returns_raw = jax.lax.scan(
                _scan_returns,
                jnp.zeros(num_envs_agents),
                (traj_batch.reward[::-1], traj_batch.discount[::-1]),
            )
            returns_raw = returns_raw[::-1]  # (rollout_length, num_envs_agents)

        # ---- GAE computation (if use_gae=True) ----
        # Uses env rewards and value predictions from rollout, independent of relabeling.
        if use_gae:
            # Compute last value for bootstrap
            last_obs = last_timestep.observation.agents_view
            if use_achieved_goal:
                last_ag = last_timestep.extras["env_metrics"]["achieved_goal"]
                while last_ag.ndim < last_obs.ndim:
                    last_ag = last_ag[..., None]
                last_gc_obs = jnp.concatenate([last_obs, last_ag, env_state.goal], axis=-1)
            else:
                last_gc_obs = jnp.concatenate([last_obs, env_state.goal], axis=-1)
            last_val = value_apply(value_params, last_gc_obs.reshape(num_envs_agents, -1))
            last_done = 1.0 - last_timestep.discount.reshape(num_envs_agents)

            # GAE scan (reverse through trajectory)
            def _gae_step(carry, t_data):
                gae, next_value, next_done = carry
                reward, value, done = t_data
                delta = reward + gamma * next_value * (1 - next_done) - value
                gae = delta + gamma * gae_lambda * (1 - next_done) * gae
                return (gae, value, done), gae

            done_flags = 1.0 - traj_batch.discount  # (T, N)
            _, gae_advantages = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_val), last_val, last_done),
                (traj_batch.reward, traj_batch.value, done_flags),
                reverse=True,
            )
            gae_targets = gae_advantages + traj_batch.value  # (T, N)
            # gae_advantages shape: (rollout_length, num_envs_agents)

        # traj_batch: (rollout_length, num_envs_agents, ...)
        # Transpose to (num_envs_agents, rollout_length, ...) for hindsight relabeling
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj_batch
        )

        # Apply hindsight relabeling
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        relabeled = jax.vmap(_flatten_crl, in_axes=(None, 0, 0))(
            (config.system.gamma, obs_dim), transitions, batch_keys
        )
        # relabeled is dict with shape (num_envs_agents, rollout_length-1, ...)

        # Flatten with Fortran order
        relabeled = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), relabeled
        )

        # Truncate to make divisible by batch_size
        num_samples = (
            relabeled["observation"].shape[0] // config.system.batch_size
        ) * config.system.batch_size
        relabeled = jax.tree_util.tree_map(lambda x: x[:num_samples], relabeled)

        # Add returns to relabeled data (if using reward-based advantages)
        if reward_advantage_coeff > 0:
            returns_per_agent = jnp.swapaxes(returns_raw, 0, 1)[:, :-1]  # (N, T-1)
            returns_flat = jnp.reshape(returns_per_agent, (-1,), order="F")
            relabeled["env_returns"] = returns_flat[:num_samples]

        # Add GAE advantages, targets, and env_goal to relabeled data
        if use_gae:
            # GAE shape: (T, N) → swap to (N, T) → take [:-1] to match relabeled
            gae_adv_per_agent = jnp.swapaxes(gae_advantages, 0, 1)[:, :-1]  # (N, T-1)
            gae_tgt_per_agent = jnp.swapaxes(gae_targets, 0, 1)[:, :-1]  # (N, T-1)
            env_goal_per_agent = jnp.swapaxes(traj_batch.env_goal, 0, 1)[:, :-1]  # (N, T-1, goal_dim)

            gae_adv_flat = jnp.reshape(gae_adv_per_agent, (-1,), order="F")
            gae_tgt_flat = jnp.reshape(gae_tgt_per_agent, (-1,), order="F")
            env_goal_flat = jnp.reshape(env_goal_per_agent, (-1,) + env_goal_per_agent.shape[2:], order="F")

            relabeled["gae_advantage"] = gae_adv_flat[:num_samples]
            relabeled["gae_target"] = gae_tgt_flat[:num_samples]
            relabeled["env_goal"] = env_goal_flat[:num_samples]

        # ---- Phase 1: Compute advantages ----
        if use_gae:
            # Use GAE advantages (already computed from env rewards + value function)
            advantages = relabeled["gae_advantage"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            relabeled["advantage"] = advantages
            # Dummy values for metrics
            q_taken_all = jnp.zeros(1)
            v_mc_all = jnp.zeros(1)
        else:
            # CRL-based advantages (Q-V gap)
            key, adv_key = jax.random.split(key)

            def _compute_advantages(carry, batch):
                """Compute Q and V for advantage estimation (no gradient)."""
                key = carry
                key, mc_key = jax.random.split(key)

                obs = batch["observation"]
                goal = batch["goal"]
                action = batch["action"]
                x_t_stored = batch["x_t"]
                if use_achieved_goal:
                    current_ag = batch["current_achieved_goal"]
                    sa_state = jnp.concatenate([obs, current_ag], axis=-1)
                    actor_input = jnp.concatenate([obs, current_ag, goal], axis=-1)
                else:
                    sa_state = obs
                    actor_input = jnp.concatenate([obs, goal], axis=-1)

                # Actor forward pass (for MC sampling)
                means, log_stds = actor_apply(params.actor, actor_input)
                stds = jnp.exp(log_stds)

                # Q(s, a_taken, g)
                sa_repr_taken = sa_encoder_apply(params.sa_encoder, sa_state, action)
                g_repr = goal_encoder_apply(params.goal_encoder, goal)
                q_taken = energy_fn(energy_fn_name, sa_repr_taken, g_repr)

                # MC V(s,g) estimation (use wider sampling if mc_std_floor > 0)
                mc_stds = jnp.maximum(stds, mc_std_floor) if mc_std_floor > 0 else stds
                mc_noise = jax.random.normal(
                    mc_key, shape=(num_mc_samples,) + means.shape, dtype=means.dtype
                )
                mc_x_ts = means[None, :, :] + mc_stds[None, :, :] * mc_noise
                mc_actions_squashed = nn.tanh(mc_x_ts)
                mc_actions = action_bias + action_scale * mc_actions_squashed

                def compute_q_for_sample(mc_action):
                    sa_repr = sa_encoder_apply(params.sa_encoder, sa_state, mc_action)
                    return energy_fn(energy_fn_name, sa_repr, g_repr)

                mc_q_values = jax.vmap(compute_q_for_sample)(mc_actions)
                v_mc = jnp.mean(mc_q_values, axis=0)

                if use_reinforce:
                    advantage = q_taken  # REINFORCE: use Q directly (higher variance, but no V cancellation)
                else:
                    advantage = q_taken - v_mc  # Approach A: MC baseline
                return key, {"advantage": advantage, "q_taken": q_taken, "v_mc": v_mc}

            # Batch relabeled data for advantage computation
            adv_batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                relabeled,
            )
            _, adv_results = jax.lax.scan(_compute_advantages, adv_key, adv_batched)
            # Flatten advantages back
            advantages = adv_results["advantage"].reshape(-1)
            q_taken_all = adv_results["q_taken"].reshape(-1)
            v_mc_all = adv_results["v_mc"].reshape(-1)

            # Optionally blend or replace CRL advantages with env return-based advantages
            if reward_advantage_coeff > 0:
                env_returns = relabeled["env_returns"].reshape(-1)
                norm_ret = (env_returns - env_returns.mean()) / (env_returns.std() + 1e-8)
                if reward_advantage_coeff >= 10.0:
                    # Pure return-based advantages (ignore CRL Q-V)
                    advantages = norm_ret
                else:
                    norm_crl = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    advantages = norm_crl + reward_advantage_coeff * norm_ret

            # Normalize advantages globally
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Add advantages to relabeled data
            relabeled["advantage"] = advantages

        # ---- Phase 2: PPO actor epochs + contrastive critic epochs ----
        def _update_minibatch(carry, batch):
            """Update critic and actor on a single minibatch."""
            params, opt_states, key, ent_coef, log_alpha, alpha_opt_state, vp, vo = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            obs = batch["observation"]
            goal = batch["goal"]  # relabeled goal (for CRL)
            action = batch["action"]
            x_t_stored = batch["x_t"]
            old_log_prob = batch["old_log_prob"]
            advantage = batch["advantage"]
            if use_achieved_goal:
                current_ag = batch["current_achieved_goal"]
                sa_state = jnp.concatenate([obs, current_ag], axis=-1)
            else:
                sa_state = obs

            # ---- Critic update (InfoNCE contrastive loss) ----
            def _critic_loss_fn(critic_params, sa_state, action, goal):
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], sa_state, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)

                logits = compute_logits(energy_fn_name, sa_repr, g_repr)
                logits_scaled = logits / contrastive_temperature
                critic_loss = contrastive_loss_fn(
                    contrastive_loss_name, logits_scaled, logsumexp_penalty_coeff
                )

                metrics = compute_contrastive_metrics(logits_scaled)
                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": metrics["logits_pos"],
                    "logits_neg": metrics["logits_neg"],
                    "categorical_accuracy": metrics["categorical_accuracy"],
                    "logsumexp": metrics["logsumexp"],
                }
                return critic_loss, loss_info

            critic_params = {
                "sa_encoder": params.sa_encoder,
                "goal_encoder": params.goal_encoder,
            }
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (_, critic_info), critic_grads = critic_grad_fn(
                critic_params, sa_state, action, goal
            )

            critic_updates, new_critic_opt_state = critic_update_fn(
                critic_grads, opt_states.critic
            )
            new_sa_encoder = optax.apply_updates(
                params.sa_encoder, critic_updates["sa_encoder"]
            )
            new_goal_encoder = optax.apply_updates(
                params.goal_encoder, critic_updates["goal_encoder"]
            )

            # ---- Actor update (PPO clipped loss with pre-computed advantages) ----
            # When use_gae: actor uses env_goal (matches GAE advantages from env rewards)
            # When not: actor uses relabeled goal (matches CRL Q-V advantages)
            if use_gae:
                actor_goal = batch["env_goal"]
            else:
                actor_goal = goal

            def _actor_loss_fn(actor_params, obs, actor_goal, x_t_stored, old_log_prob, advantage):
                if use_achieved_goal:
                    actor_input = jnp.concatenate([obs, batch["current_achieved_goal"], actor_goal], axis=-1)
                else:
                    actor_input = jnp.concatenate([obs, actor_goal], axis=-1)
                means, log_stds = actor_apply(actor_params, actor_input)
                stds = jnp.exp(log_stds)

                # Recompute Gaussian log prob from stored pre-tanh value
                noise = (x_t_stored - means) / (stds + 1e-8)
                log_prob = -0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)
                log_prob = log_prob.sum(-1)  # [B]

                # PPO clipped loss with pre-computed advantages
                ratio = jnp.exp(log_prob - old_log_prob)
                ratio_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                loss_unclipped = -advantage * ratio
                loss_clipped = -advantage * ratio_clipped
                ppo_loss = jnp.mean(jnp.maximum(loss_unclipped, loss_clipped))

                # Entropy bonus (analytical Gaussian entropy)
                entropy = (log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()

                # Entropy floor penalty: extra cost when entropy drops below target
                entropy_floor_penalty = jnp.maximum(target_entropy - entropy, 0.0) * entropy_floor_coeff

                actor_loss = ppo_loss - ent_coef * entropy + entropy_floor_penalty

                loss_info = {
                    "actor_loss": actor_loss,
                    "ppo_loss": ppo_loss,
                    "entropy": entropy,
                    "ratio_mean": ratio.mean(),
                }
                return actor_loss, loss_info

            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (_, actor_info), actor_grads = actor_grad_fn(
                params.actor, obs, actor_goal, x_t_stored, old_log_prob, advantage
            )

            actor_updates, new_actor_opt_state = actor_update_fn(
                actor_grads, opt_states.actor
            )
            new_actor = optax.apply_updates(params.actor, actor_updates)

            # ---- Value update (MSE loss on GAE targets, only when use_gae) ----
            new_vp, new_vo = vp, vo
            value_loss_val = jnp.zeros(())
            if use_gae:
                gae_target = batch["gae_target"]

                def _value_loss_fn(value_params, obs, actor_goal):
                    if use_achieved_goal:
                        v_input = jnp.concatenate([obs, batch["current_achieved_goal"], actor_goal], axis=-1)
                    else:
                        v_input = jnp.concatenate([obs, actor_goal], axis=-1)
                    v_pred = value_apply(value_params, v_input)
                    return vf_coef * jnp.mean((v_pred - gae_target) ** 2), v_pred.mean()

                value_grad_fn = jax.value_and_grad(_value_loss_fn, has_aux=True)
                (value_loss_val, _), value_grads = value_grad_fn(vp, obs, actor_goal)
                value_updates, new_vo = value_update_fn(value_grads, vo)
                new_vp = optax.apply_updates(vp, value_updates)

            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=new_actor,
            )
            new_opt_states = OptStates(
                actor=new_actor_opt_state,
                critic=new_critic_opt_state,
            )

            # ---- Alpha update (SAC-style adaptive entropy) ----
            entropy_for_alpha = actor_info["entropy"]
            # grad = (entropy - target): when entropy > target, decrease alpha; when < target, increase
            alpha_grad = entropy_for_alpha - target_entropy
            new_log_alpha = log_alpha - alpha_lr * alpha_grad  # simple SGD
            # Clip log_alpha to prevent extreme values: ent_coef in [0.003, 0.05]
            new_log_alpha = jnp.clip(new_log_alpha, jnp.log(0.003), jnp.log(0.05))
            new_alpha_opt_state = alpha_opt_state  # no optimizer needed for SGD
            # Update ent_coef for next minibatch if adaptive
            new_ent_coef = jnp.where(use_adaptive_entropy, jnp.exp(new_log_alpha), ent_coef)

            metrics = critic_info | actor_info
            metrics["advantage_mean"] = advantage.mean()
            metrics["q_taken_mean"] = q_taken_all.mean()
            metrics["v_mc_mean"] = v_mc_all.mean()
            metrics["ent_coef"] = ent_coef
            metrics["value_loss"] = value_loss_val
            return (new_params, new_opt_states, key, new_ent_coef, new_log_alpha, new_alpha_opt_state, new_vp, new_vo), metrics

        # ---- Critic-only warmup epochs (improve representations before actor update) ----
        def _critic_only_minibatch(carry, batch):
            """Update only the critic (SA + goal encoders) on a minibatch."""
            params, opt_states, key = carry
            key, _ = jax.random.split(key)

            obs = batch["observation"]
            goal = batch["goal"]
            action = batch["action"]
            if use_achieved_goal:
                sa_state = jnp.concatenate([obs, batch["current_achieved_goal"]], axis=-1)
            else:
                sa_state = obs

            def _critic_loss_fn(critic_params, sa_state, action, goal):
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], sa_state, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)
                logits = compute_logits(energy_fn_name, sa_repr, g_repr)
                logits_scaled = logits / contrastive_temperature
                critic_loss = contrastive_loss_fn(
                    contrastive_loss_name, logits_scaled, logsumexp_penalty_coeff
                )
                return critic_loss, {}

            critic_params = {
                "sa_encoder": params.sa_encoder,
                "goal_encoder": params.goal_encoder,
            }
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (_, _), critic_grads = critic_grad_fn(critic_params, sa_state, action, goal)

            critic_updates, new_critic_opt_state = critic_update_fn(
                critic_grads, opt_states.critic
            )
            new_sa_encoder = optax.apply_updates(
                params.sa_encoder, critic_updates["sa_encoder"]
            )
            new_goal_encoder = optax.apply_updates(
                params.goal_encoder, critic_updates["goal_encoder"]
            )

            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=params.actor,
            )
            new_opt_states = OptStates(
                actor=opt_states.actor,
                critic=new_critic_opt_state,
            )
            return (new_params, new_opt_states, key), {}

        def _critic_warmup_epoch(carry, _):
            """One critic-only epoch over all data."""
            params, opt_states, key, flat_data = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_data["observation"].shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                shuffled,
            )
            (params, opt_states, key), _ = jax.lax.scan(
                _critic_only_minibatch, (params, opt_states, key), batched
            )
            return (params, opt_states, key, flat_data), {}

        # Run critic warmup epochs (before advantages are computed with updated critic)
        if num_critic_warmup_epochs > 0:
            (params, opt_states, key, _), _ = jax.lax.scan(
                _critic_warmup_epoch, (params, opt_states, key, relabeled),
                None, num_critic_warmup_epochs
            )

        def _train_epoch(carry, _):
            """Train for one epoch over all data (with reshuffling)."""
            params, opt_states, key, flat_data, ent_coef, log_alpha, alpha_opt_state, vp, vo = carry

            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_data["observation"].shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)

            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                shuffled,
            )

            (params, opt_states, key, ent_coef, log_alpha, alpha_opt_state, vp, vo), epoch_metrics = jax.lax.scan(
                _update_minibatch, (params, opt_states, key, ent_coef, log_alpha, alpha_opt_state, vp, vo), batched
            )

            return (params, opt_states, key, flat_data, ent_coef, log_alpha, alpha_opt_state, vp, vo), epoch_metrics

        # Train for num_epochs
        (params, opt_states, key, _, _, log_alpha, alpha_opt_state, value_params, value_opt_state), train_metrics = jax.lax.scan(
            _train_epoch, (params, opt_states, key, relabeled, ent_coef, log_alpha, alpha_opt_state, value_params, value_opt_state), None, num_epochs
        )

        learner_state = PPOCRLLearnerState(
            params, opt_states, key, env_state, last_timestep, update_count + 1, log_alpha, alpha_opt_state, value_params, value_opt_state
        )
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: PPOCRLLearnerState) -> ExperimentOutput:
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


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states."""
    config.system.num_agents = env.num_agents

    key, sa_key, goal_key, actor_key, value_key = jax.random.split(keys[0], 5)

    # Build networks from config
    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    actor_torso = instantiate(config.network.actor_network.pre_torso)

    rep_size = config.system.get("rep_size", 64)
    action_embed_dim = config.system.get("action_embed_dim", 0)
    use_film = config.system.get("use_film", False)
    sa_encoder = SAEncoder(torso=sa_encoder_torso, output_dim=rep_size, action_embed_dim=action_embed_dim, use_film=use_film)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=rep_size)
    log_std_min = config.system.get("log_std_min", -5.0)
    log_std_max = config.system.get("log_std_max", 2.0)
    actor_network = ICRLActor(
        torso=actor_torso, action_size=env.action_dim,
        LOG_STD_MIN=log_std_min, LOG_STD_MAX=log_std_max,
    )

    # Dummy inputs for init
    obs_dim = env.observation_spec.agents_view.shape[-1]
    goal_dim = env.goal_dim
    use_achieved_goal = config.system.get("use_achieved_goal", True)
    if use_achieved_goal:
        init_sa_state = jnp.zeros((1, obs_dim + goal_dim))  # concat(obs, achieved_goal)
        init_actor_input = jnp.zeros((1, obs_dim + goal_dim + goal_dim))  # concat(obs, ag, goal)
    else:
        init_sa_state = jnp.zeros((1, obs_dim))  # obs only
        init_actor_input = jnp.zeros((1, obs_dim + goal_dim))  # concat(obs, goal)
    init_action = jnp.zeros((1, env.action_dim))
    init_goal = jnp.zeros((1, goal_dim))

    sa_encoder_params = sa_encoder.init(sa_key, init_sa_state, init_action)
    goal_encoder_params = goal_encoder.init(goal_key, init_goal)
    actor_params = actor_network.init(actor_key, init_actor_input)

    # Value network (for GAE, same input as actor)
    use_gae = config.system.get("use_gae", False)
    if use_gae:
        value_torso = instantiate(config.network.actor_network.pre_torso)  # same architecture as actor
        value_network = ICRLValueNet(torso=value_torso)
        value_params = value_network.init(value_key, init_actor_input)
    else:
        value_network = None
        value_params = None

    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
    )

    # Optimizers
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)

    # LR scheduling
    lr_linear_decay = config.system.get("lr_linear_decay", False)
    if lr_linear_decay:
        num_epochs = config.system.get("num_epochs", 4)
        samples_per_update = (
            config.arch.num_envs * env.num_agents * (config.system.rollout_length - 1)
        )
        minibatches_per_update = samples_per_update // config.system.batch_size
        total_grad_steps = config.system.num_updates * num_epochs * minibatches_per_update

        actor_lr = optax.linear_schedule(
            init_value=config.system.actor_lr,
            end_value=config.system.get("lr_end", 1e-7),
            transition_steps=total_grad_steps,
        )
        critic_lr = optax.linear_schedule(
            init_value=config.system.q_lr,
            end_value=config.system.get("lr_end", 1e-7),
            transition_steps=total_grad_steps,
        )
    else:
        actor_lr = config.system.actor_lr
        critic_lr = config.system.q_lr

    actor_opt = optax.chain(grad_clip, optax.adam(actor_lr))
    critic_opt = optax.chain(grad_clip, optax.adam(critic_lr))

    critic_params_struct = {
        "sa_encoder": sa_encoder_params,
        "goal_encoder": goal_encoder_params,
    }
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params_struct)

    # Value optimizer (for GAE)
    if use_gae:
        value_opt = optax.chain(grad_clip, optax.adam(actor_lr))  # same LR schedule as actor
        value_opt_state = value_opt.init(value_params)
        value_apply_fn = value_network.apply
        value_update_fn = value_opt.update
    else:
        value_opt_state = None
        value_apply_fn = lambda *a, **k: None  # dummy
        value_update_fn = lambda *a, **k: (None, None)  # dummy

    opt_states = OptStates(actor=actor_opt_state, critic=critic_opt_state)

    # Apply and update functions
    apply_fns = (sa_encoder.apply, goal_encoder.apply, actor_network.apply, value_apply_fn)
    update_fns = (actor_opt.update, critic_opt.update, value_update_fn)

    # Get learner function
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.jit(learn)

    # Initialize environment states
    key, *env_keys = jax.random.split(key, config.arch.num_envs + 1)
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    # Load checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # Initialize learnable entropy coefficient (log_alpha)
    init_log_alpha = jnp.log(jnp.array(config.system.ent_coef, dtype=jnp.float32))
    init_alpha_opt_state = jnp.array(0.0)  # Placeholder (using SGD, no state needed)

    key, step_key = jax.random.split(key)
    init_learner_state = PPOCRLLearnerState(
        params, opt_states, step_key, env_states, timesteps,
        update_count=jnp.array(0, dtype=jnp.int32),
        log_alpha=init_log_alpha,
        alpha_opt_state=init_alpha_opt_state,
        value_params=value_params,
        value_opt_state=value_opt_state,
    )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Run PPO-CRL continuous experiment."""
    _config.logger.system_name = "ppo_crl_continuous"
    config = copy.deepcopy(_config)

    # Create environments
    env, eval_env = environments.make(config)

    # PRNG keys
    key, key_e, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )

    learn, actor_network, learner_state = learner_setup(
        env, (key, sa_key, goal_key, actor_key), config
    )

    jax.block_until_ready(learner_state)

    # Setup evaluator (stochastic: sample from policy, matches training distribution)
    use_ag = config.system.get("use_achieved_goal", True)
    eval_act_fn = make_eval_act_fn(actor_network.apply, eval_env, deterministic=False, use_achieved_goal=use_ag)
    evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert config.system.num_updates >= config.arch.num_evaluation, (
        "Number of updates must be >= number of evaluations."
    )

    config.system.num_updates_per_eval = (
        config.system.num_updates // config.arch.num_evaluation
    )
    steps_per_rollout = (
        config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.num_envs
    )

    # Logger
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    max_episode_return = -jnp.inf
    best_params = None

    fixed_eval_key = key_e[jnp.newaxis, :]

    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(
            learner_output.episode_metrics
        )
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluation
        trained_params = learner_output.learner_state.params.actor

        eval_metrics = evaluator(trained_params, fixed_eval_key, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_output.learner_state,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        learner_state = learner_output.learner_state

    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    if config.arch.absolute_metric:
        abs_metric_evaluator = get_icrl_eval_fn(
            eval_env, eval_act_fn, config, absolute_metric=True
        )
        eval_metrics = abs_metric_evaluator(best_params, fixed_eval_key, {})
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ppo_crl_continuous.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    try:
        eval_performance = run_experiment(cfg)
        return eval_performance
    except Exception as e:
        try:
            override_info = cfg.hydra.job.override_dirname
        except Exception:
            override_info = "unknown"
        print(f"Error executing job with overrides: {override_info}")
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback

        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
