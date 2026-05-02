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
    # Decoupled minibatch sizes AND epoch counts. The InfoNCE critic loss
    # builds a [B, B] logits matrix and is memory-bound at B > ~1024; the PPO
    # actor loss has no such constraint and benefits from many fewer SGD
    # steps over many big batches (IPPO-style: a couple of huge minibatches,
    # several epochs of clipped-ratio updates over the same data). Both pairs
    # default to the legacy single-knob values for backward compatibility.
    critic_batch_size = config.system.batch_size
    actor_batch_size = config.system.get("actor_batch_size", critic_batch_size)
    actor_ppo_epochs = int(config.system.get("actor_ppo_epochs", config.system.ppo_epochs))
    critic_ppo_epochs = int(config.system.get("critic_ppo_epochs", config.system.ppo_epochs))
    # Per-epoch fraction of on-policy transitions the critic actually trains
    # on. 1.0 = train on every collected transition (default, sample-efficient
    # but maximum SGD-iter count). 0.5 / 0.25 = randomly subsample, fewer
    # critic minibatches, faster wall-clock at the cost of throwing away
    # fresh on-policy data each epoch. Each epoch reshuffles + resamples.
    critic_subsample_fraction = float(config.system.get("critic_subsample_fraction", 1.0))
    if not (0.0 < critic_subsample_fraction <= 1.0):
        raise ValueError(
            f"critic_subsample_fraction must be in (0, 1], got {critic_subsample_fraction}"
        )
    scan_unroll = int(config.system.get("scan_unroll", 8))
    # When use_gae is True, advantages are computed by smoothing the
    # contrastive single-step advantage Q(s,a,g) - V(s,g) across time with
    # GAE-style discounting. gae_kind selects the per-step signal:
    #   "smooth_adv" (default, recommended): δ_t = Q_taken_t - V_t.
    #       Smooths the original single-step contrastive advantage along the
    #       trajectory. λ=0 recovers the no-GAE baseline; λ→1 fully credits
    #       each action with all later single-step advantages on the
    #       trajectory. Doesn't need a reward, doesn't need V to satisfy a
    #       Bellman equation — just temporal smoothing of an already-good
    #       per-step signal.
    #   "value_diff" (the originally-tried, broken variant): δ_t =
    #       γ V_{t+1} - V_t. Standard PPO TD residual without reward. Fails
    #       at init because V is random and the bootstrap residual has no
    #       grounding signal.
    #   "q_as_reward": δ_t = Q_taken_t + γ V_{t+1} - V_t. Treats Q as a
    #       per-step "reward". Closer to the standard PPO formula, but with
    #       a Bellman bootstrap that the contrastive V doesn't satisfy.
    use_gae = bool(config.system.get("use_gae", False))
    gae_lambda = float(config.system.get("gae_lambda", 0.95))
    gae_kind = str(config.system.get("gae_kind", "smooth_adv"))
    gamma_for_gae = float(config.system.get("gamma", 0.99))

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

        # traj_batch has shape (T, E, ...). vmap HER over axis 1 directly to skip
        # an explicit swapaxes copy. C-order reshape on (T-1, E, ...) interleaves
        # consecutive flat indices by env (same as the previous (E, T-1) Fortran
        # reshape) — downstream PPO shuffles anyway, so order is preserved.
        batch_keys = jax.random.split(sample_key, traj_batch.observation.shape[1])
        transitions = jax.vmap(flatten_crl_fn, in_axes=(None, 1, 0), out_axes=1)(
            config.system.gamma, traj_batch, batch_keys
        )
        # Output shape: (T-1, E, ...)

        transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            transitions,
        )

        # Truncate so the flat batch divides cleanly by *both* actor and
        # critic batch sizes. With power-of-two batch sizes max() is the LCM.
        align = max(actor_batch_size, critic_batch_size)
        num_samples = (len(transitions.observation) // align) * align
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)

        def compute_crl_advantages(transitions, params, sa_encoder_apply_fn, g_encoder_apply_fn):
            real_ultimate_goal = transitions.extras["real_ultimate_goal"]
            g_repr = g_encoder_apply_fn(params.goal_encoder, real_ultimate_goal)  # [N, R]

            # Batched per-action Q: stack all actions on a new axis and call the
            # SA encoder ONCE on [N, A, ...] instead of action_dim times on [N, ...].
            # Mathematically identical to the previous vmap-over-actions; gives XLA
            # a single big matmul to fuse and removes per-action launch overhead.
            obs_achieved = jnp.concatenate(
                [transitions.observation, transitions.achieved_goal], axis=-1
            )  # [N, S]
            n = obs_achieved.shape[0]
            obs_achieved_tiled = jnp.broadcast_to(
                obs_achieved[:, None, :], (n, action_dim, obs_achieved.shape[-1])
            )  # [N, A, S]
            actions_tiled = jnp.broadcast_to(
                jnp.eye(action_dim, dtype=obs_achieved.dtype)[None, :, :],
                (n, action_dim, action_dim),
            )  # [N, A, A]
            sa_repr = sa_encoder_apply_fn(
                params.sa_encoder, obs_achieved_tiled, actions_tiled
            )  # [N, A, R]
            q_values = energy_fn(energy_fn_name, sa_repr, g_repr[:, None, :])  # [N, A]

            # Policy distribution for V = E_pi[Q]
            actor_input = jnp.concatenate(
                [transitions.observation, transitions.achieved_goal, real_ultimate_goal], axis=-1
            )
            actor_obs = Observation(agents_view=actor_input, action_mask=transitions.avail_actions)
            pi = actor_apply_fn(params.actor, actor_obs)
            action_probs = jax.nn.softmax(pi.distribution.logits)

            v_values = jnp.sum(action_probs * q_values, axis=-1)  # [N]
            taken_action_idx = jnp.argmax(transitions.action, axis=-1)  # [N]
            q_taken = jnp.take_along_axis(q_values, taken_action_idx[:, None], axis=-1).squeeze(-1)

            advantages = q_taken - v_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return jax.lax.stop_gradient(advantages)

        def compute_crl_gae_advantages(traj_batch, params):
            """GAE-style advantages using the contrastive critic.

            See `gae_kind` config (smooth_adv | value_diff | q_as_reward) for
            the per-step signal δ_t. The smoothing recurrence is the standard
            GAE one, run as a reverse-time scan:

                GAE_t = δ_t + γ * λ * (1 - done_t) * GAE_{t+1}

            Operates on (T, E, ...) traj_batch BEFORE HER relabeling so V is
            conditioned on the env's real ultimate_goal (the same goal the
            actor sees in its loss). Returns advantages of shape (T, E).
            """
            # Flatten (T, E, ...) to (N, ...) for the network forward passes —
            # same shape and memory plan as the non-GAE compute_crl_advantages
            # path. We reshape back to (T, E, ...) only for the small reverse
            # scan at the end. This matters for memory: the 4D
            # (T, E, A, S) -> Dense layout caused OOM at 80M scale, while the
            # equivalent 3D (N, A, S) -> Dense plan is fine.
            T_ = traj_batch.observation.shape[0]
            E_ = traj_batch.observation.shape[1]
            obs_achieved_flat = jnp.concatenate(
                [traj_batch.observation, traj_batch.achieved_goal], axis=-1
            ).reshape(T_ * E_, -1)  # [N, S]
            ultimate_flat = traj_batch.ultimate_goal.reshape(T_ * E_, -1)
            avail_flat = traj_batch.avail_actions.reshape(T_ * E_, -1)
            action_flat = traj_batch.action.reshape(T_ * E_, -1)

            n = obs_achieved_flat.shape[0]
            obs_achieved_tiled = jnp.broadcast_to(
                obs_achieved_flat[:, None, :], (n, action_dim, obs_achieved_flat.shape[-1])
            )  # [N, A, S]
            actions_tiled = jnp.broadcast_to(
                jnp.eye(action_dim, dtype=obs_achieved_flat.dtype)[None, :, :],
                (n, action_dim, action_dim),
            )  # [N, A, A]
            sa_repr = sa_encoder_apply_fn(
                params.sa_encoder, obs_achieved_tiled, actions_tiled
            )  # [N, A, R]
            g_repr = g_encoder_apply_fn(params.goal_encoder, ultimate_flat)  # [N, R]
            q_values_flat = energy_fn(
                energy_fn_name, sa_repr, g_repr[:, None, :]
            )  # [N, A]

            actor_input = jnp.concatenate(
                [
                    obs_achieved_flat[:, : traj_batch.observation.shape[-1]],
                    obs_achieved_flat[:, traj_batch.observation.shape[-1]:],
                    ultimate_flat,
                ],
                axis=-1,
            )
            actor_obs = Observation(agents_view=actor_input, action_mask=avail_flat)
            pi = actor_apply_fn(params.actor, actor_obs)
            action_probs = jax.nn.softmax(pi.distribution.logits)  # [N, A]
            values_flat = jnp.sum(action_probs * q_values_flat, axis=-1)  # [N]

            taken_action_idx_flat = jnp.argmax(action_flat, axis=-1)  # [N]
            q_taken_flat = jnp.take_along_axis(
                q_values_flat, taken_action_idx_flat[:, None], axis=-1
            ).squeeze(-1)  # [N]

            # Reshape back to (T, E) for the scan.
            values = values_flat.reshape(T_, E_)
            q_taken = q_taken_flat.reshape(T_, E_)
            dones = traj_batch.extras["done"].astype(values.dtype)  # [T, E]

            # Build the per-step delta_t according to gae_kind.
            if gae_kind == "smooth_adv":
                # δ_t = Q_taken_t - V_t (the original single-step adv).
                # Each timestep is independently grounded; GAE just smooths
                # along the trajectory. No bootstrap term, no reward.
                deltas = q_taken - values  # [T, E]

                def _gae_step(carry, transition):
                    gae, next_done = carry
                    delta, done = transition
                    gae = delta + gamma_for_gae * gae_lambda * (1.0 - next_done) * gae
                    return (gae, done), gae

                _, advantages_te = jax.lax.scan(
                    _gae_step,
                    (jnp.zeros_like(values[-1]), dones[-1]),
                    (deltas, dones),
                    reverse=True,
                )
            elif gae_kind == "q_as_reward":
                # δ_t = Q_taken_t + γ V_{t+1} (1 - done_t) - V_t.
                # Treats Q as a per-step reward and bootstraps off V.
                # Note: V isn't trained by Bellman regression in CRL, so
                # this isn't strictly principled — included for ablation.
                def _gae_step(carry, transition):
                    gae, next_value, next_done = carry
                    value, q, done = transition
                    delta = q + gamma_for_gae * next_value * (1.0 - next_done) - value
                    gae = delta + gamma_for_gae * gae_lambda * (1.0 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages_te = jax.lax.scan(
                    _gae_step,
                    (jnp.zeros_like(values[-1]), values[-1], dones[-1]),
                    (values, q_taken, dones),
                    reverse=True,
                )
            elif gae_kind == "value_diff":
                # δ_t = γ V_{t+1} (1 - done_t) - V_t. Pure value-difference
                # TD without reward. Empirically broken on this task at
                # init — kept here only for completeness/ablation.
                def _gae_step(carry, transition):
                    gae, next_value, next_done = carry
                    value, done = transition
                    delta = gamma_for_gae * next_value * (1.0 - next_done) - value
                    gae = delta + gamma_for_gae * gae_lambda * (1.0 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages_te = jax.lax.scan(
                    _gae_step,
                    (jnp.zeros_like(values[-1]), values[-1], dones[-1]),
                    (values, dones),
                    reverse=True,
                )
            else:
                raise ValueError(f"Unknown gae_kind: {gae_kind!r}")

            advantages_te = (advantages_te - advantages_te.mean()) / (advantages_te.std() + 1e-8)
            return jax.lax.stop_gradient(advantages_te)

        if use_gae:
            # Compute on the pre-HER (T, E, ...) traj batch and reshape
            # C-order to (T*E,). The transitions tree is also reshaped C-order
            # from (T, E, ...) so the index ordering matches and we can pair
            # advantages with transitions directly.
            advantages_te = compute_crl_gae_advantages(traj_batch, params)
            advantages_full = advantages_te.reshape(-1)  # [T*E]
            advantages = advantages_full[:num_samples]
        else:
            advantages = compute_crl_advantages(
                transitions, params, sa_encoder_apply_fn, g_encoder_apply_fn
            )

        # Strip extras to only the fields the PPO/critic losses read. Drops
        # state, future_state, future_action, done, state_extras (truncation/seed)
        # and policy_extras — none of them are referenced past this point but they
        # would otherwise be carried + shuffled in every epoch and minibatch.
        transitions = transitions._replace(extras={
            "real_ultimate_goal": transitions.extras["real_ultimate_goal"],
            "advantages": advantages,
        })
        def _actor_loss_fn(actor_params, batch, key) -> Tuple:
            """PPO clipped surrogate + entropy bonus, advantages frozen."""
            actor_input = jnp.concatenate(
                [batch.observation, batch.achieved_goal, batch.extras["real_ultimate_goal"]],
                axis=-1,
            )
            actor_obs = Observation(agents_view=actor_input, action_mask=batch.avail_actions)
            pi = actor_apply_fn(actor_params, actor_obs)
            taken_action_idx = jnp.argmax(batch.action, axis=-1)
            new_log_prob = pi.distribution.log_prob(taken_action_idx)
            ratio = jnp.exp(new_log_prob - batch.log_prob)
            advantages = batch.extras["advantages"]
            actor_loss1 = ratio * advantages
            actor_loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -jnp.mean(jnp.minimum(actor_loss1, actor_loss2))
            entropy = pi.distribution.entropy().mean()
            total_actor_loss = actor_loss - ent_coef * entropy
            return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}

        def _critic_loss_fn(critic_params, obs, achieved, ultimate, action_onehot, critic_key):
            """InfoNCE contrastive loss on a [B, B] logits matrix."""
            sa_input = jnp.concatenate([obs, achieved], axis=-1)
            sa_repr = sa_encoder_apply_fn(critic_params["sa_encoder"], sa_input, action_onehot)
            g_repr = g_encoder_apply_fn(critic_params["goal_encoder"], ultimate)
            logits = compute_logits(energy_fn_name, sa_repr, g_repr)
            critic_loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)
            metrics = compute_contrastive_metrics(logits)
            q_taken = energy_fn(energy_fn_name, sa_repr, g_repr)
            return critic_loss, {
                "critic_loss": critic_loss,
                "logits_pos": metrics["logits_pos"],
                "logits_neg": metrics["logits_neg"],
                "categorical_accuracy": metrics["categorical_accuracy"],
                "logsumexp": metrics["logsumexp"],
                "q_mean": q_taken.mean(),
            }

        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)

        def _actor_update_minibatch(carry, batch):
            actor_params, actor_opt_state, key = carry
            key, sub = jax.random.split(key)
            (_, info), grads = actor_grad_fn(actor_params, batch, sub)
            grads, info = lax.pmean((grads, info), axis_name="device")
            updates, new_opt_state = actor_update_fn(grads, actor_opt_state)
            new_params = optax.apply_updates(actor_params, updates)
            return (new_params, new_opt_state, key), info

        def _critic_update_minibatch(carry, batch):
            critic_params, critic_opt_state, key = carry
            key, sub = jax.random.split(key)
            (_, info), grads = critic_grad_fn(
                critic_params,
                batch.observation,
                batch.achieved_goal,
                batch.ultimate_goal,
                batch.action,
                sub,
            )
            grads, info = lax.pmean((grads, info), axis_name="device")
            updates, new_opt_state = critic_update_fn(grads, critic_opt_state)
            new_sa = optax.apply_updates(critic_params["sa_encoder"], updates["sa_encoder"])
            new_g = optax.apply_updates(critic_params["goal_encoder"], updates["goal_encoder"])
            return ({"sa_encoder": new_sa, "goal_encoder": new_g}, new_opt_state, key), info

        def _actor_epoch(carry, _):
            """One actor epoch: shuffle, batch into actor_batch_size, scan minibatches."""
            actor_params, actor_opt_state, key, flat_transitions = carry
            n = flat_transitions.observation.shape[0]
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_transitions)
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, actor_batch_size) + x.shape[1:]),
                shuffled,
            )
            (new_params, new_opt_state, key), metrics = jax.lax.scan(
                _actor_update_minibatch,
                (actor_params, actor_opt_state, key),
                batched,
                unroll=scan_unroll,
            )
            return (new_params, new_opt_state, key, flat_transitions), jax.tree_util.tree_map(jnp.mean, metrics)

        def _critic_epoch(carry, _):
            """One critic epoch: shuffle, optionally subsample, batch, scan minibatches."""
            critic_params, critic_opt_state, key, flat_transitions = carry
            n = flat_transitions.observation.shape[0]
            # n_critic is computed at trace time from static shapes + a Python
            # float, so it stays a compile-time constant.
            n_critic = int(n * critic_subsample_fraction)
            n_critic = (n_critic // critic_batch_size) * critic_batch_size
            key, perm_key = jax.random.split(key)
            # Each epoch picks a fresh random subset of size n_critic.
            perm = jax.random.permutation(perm_key, n)[:n_critic]
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_transitions)
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, critic_batch_size) + x.shape[1:]),
                shuffled,
            )
            (new_params, new_opt_state, key), metrics = jax.lax.scan(
                _critic_update_minibatch,
                (critic_params, critic_opt_state, key),
                batched,
                unroll=scan_unroll,
            )
            return (new_params, new_opt_state, key, flat_transitions), jax.tree_util.tree_map(jnp.mean, metrics)

        # Actor and critic epoch loops are fully decoupled. The actor side
        # can do IPPO-style "few epochs over a couple of huge minibatches";
        # the critic side stays at small minibatches because of the InfoNCE
        # [B, B] matrix. Mathematically equivalent to the interleaved loop
        # because actor and critic params are disjoint and the advantages
        # (which the actor reads) are frozen before this whole block.
        actor_carry_in = (params.actor, opt_states.actor, key, transitions)
        (new_actor, new_actor_opt, key, _), actor_train_metrics = jax.lax.scan(
            _actor_epoch, actor_carry_in, None, actor_ppo_epochs,
        )

        critic_init = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
        critic_carry_in = (critic_init, opt_states.critic, key, transitions)
        (new_critic, new_critic_opt, key, _), critic_train_metrics = jax.lax.scan(
            _critic_epoch, critic_carry_in, None, critic_ppo_epochs,
        )

        new_params = ICRLParams(
            actor=new_actor,
            sa_encoder=new_critic["sa_encoder"],
            goal_encoder=new_critic["goal_encoder"],
        )
        new_opt_states = OptStates(actor=new_actor_opt, critic=new_critic_opt)
        # Reduce per-epoch series to scalars so actor and critic with
        # different epoch counts produce the same shape for the outer scan.
        train_metrics = {
            **jax.tree_util.tree_map(jnp.mean, actor_train_metrics),
            **jax.tree_util.tree_map(jnp.mean, critic_train_metrics),
        }

        learner_state = LearnerState(new_params, new_opt_states, key, env_state, last_timestep, t)
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
    critic_batch_size = config.system.batch_size
    actor_batch_size = config.system.get("actor_batch_size", critic_batch_size)
    actor_ppo_epochs = int(config.system.get("actor_ppo_epochs", config.system.ppo_epochs))
    critic_ppo_epochs = int(config.system.get("critic_ppo_epochs", config.system.ppo_epochs))
    critic_subsample_fraction = float(config.system.get("critic_subsample_fraction", 1.0))
    critic_samples_per_epoch = int(samples_per_update * critic_subsample_fraction)
    critic_samples_per_epoch = (critic_samples_per_epoch // critic_batch_size) * critic_batch_size
    actor_minibatches = samples_per_update // actor_batch_size
    critic_minibatches = critic_samples_per_epoch // critic_batch_size
    actor_total_grad_steps = config.system.num_updates * actor_ppo_epochs * actor_minibatches
    critic_total_grad_steps = config.system.num_updates * critic_ppo_epochs * critic_minibatches

    # Build the LR schedules with per-side total_grad_steps. make_learning_rate
    # reads config.system.total_grad_steps and config.system.num_minibatches,
    # so we set them, build the schedule, then move on.
    config.system.total_grad_steps = actor_total_grad_steps
    config.system.num_minibatches = actor_minibatches
    actor_lr = make_learning_rate(config.system.actor_lr, config)
    config.system.total_grad_steps = critic_total_grad_steps
    config.system.num_minibatches = critic_minibatches
    crl_lr = make_learning_rate(config.system.q_lr, config)
    # Leave config.system.* set to whatever value is more useful for downstream
    # code (we publish the total of both for visibility in logs/checkpoints).
    config.system.actor_total_grad_steps = actor_total_grad_steps
    config.system.critic_total_grad_steps = critic_total_grad_steps
    config.system.total_grad_steps = actor_total_grad_steps + critic_total_grad_steps
    config.system.num_minibatches = max(actor_minibatches, critic_minibatches)
    config.system.actor_batch_size = actor_batch_size
    config.system.critic_batch_size = critic_batch_size
    config.system.actor_ppo_epochs = actor_ppo_epochs
    config.system.critic_ppo_epochs = critic_ppo_epochs
    config.system.critic_subsample_fraction = critic_subsample_fraction
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

    compile_seconds = 0.0
    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        # SPS = env steps run by learn() / wall-clock for that learn() call.
        # First eval includes JIT compilation, so its SPS is not representative
        # of steady-state throughput. We additionally log compile_seconds (only
        # on iter 0) and wall_clock_seconds (every iter) so a benchmark can
        # isolate compile cost from steady-state SPS.
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        episode_metrics["wall_clock_seconds"] = elapsed_time
        if eval_step == 0:
            compile_seconds = elapsed_time
            episode_metrics["compile_seconds"] = compile_seconds

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
