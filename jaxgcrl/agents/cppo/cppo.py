"""CPPO — PPO actor + CRL contrastive critic for continuous control.

JaxGCRL-style port. Pure CRL (no env-reward dependency anywhere).

Core algorithm:
  1. Collect rollout of length `rollout_length` with current actor.
  2. Hindsight-relabel goals via future achieved goals (per-env, masked by traj_id).
  3. Critic: InfoNCE contrastive between phi(s,a) and psi(g_relabeled).
  4. Advantage: Monte-Carlo V(s,g) baseline.
       Q(s,a,g) = energy(phi(s,a), psi(g))
       V(s,g)   = mean over K actor samples of Q(s,a',g)
       A        = Q(s,a_taken,g) - V(s,g)
  5. Actor: PPO clipped objective + entropy bonus (optionally adaptive).

No GAE, no reward advantage blend, no critic warmup, no REINFORCE branch.
Tunable knobs only (network size, lr, entropy schedule, MC samples, etc.).
"""

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass as flax_dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TerminateOnSuccessWrapper, TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator

from .losses import (
    compute_contrastive_metrics,
    compute_logits,
    contrastive_loss_fn as contrastive_loss_fn_,
    energy_fn as energy_fn_,
)
from .networks import Actor, GoalEncoder, SAEncoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


@flax_dataclass
class TrainingState:
    env_steps: jnp.ndarray
    update_count: jnp.ndarray
    actor_state: TrainState
    sa_encoder_state: TrainState
    g_encoder_state: TrainState
    log_alpha: jnp.ndarray  # adaptive entropy coefficient (used when use_adaptive_entropy)


class Transition(NamedTuple):
    observation: jnp.ndarray  # full obs [state | target_goal] from env
    action: jnp.ndarray  # post-tanh action in [-1, 1]
    x_t: jnp.ndarray  # pre-tanh value (for stable log_prob recomputation)
    log_prob: jnp.ndarray  # Gaussian log_prob in pre-tanh space (sum over action dims)
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Dict[str, Any]


@dataclass
class CPPO:
    """PPO actor + CRL contrastive critic for continuous control (JaxGCRL port)."""

    # Optimization
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    max_grad_norm: float = 0.5
    lr_linear_decay: bool = False
    lr_end: float = 1e-7

    # Rollout / batching
    rollout_length: int = 256
    num_epochs: int = 4
    batch_size: int = 256

    # PPO
    discounting: float = 0.99
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    ent_coef_end: float = 0.01
    ent_schedule_horizon: Optional[int] = None  # default: total updates

    # MC V(s,g) estimate
    num_mc_samples: int = 16
    mc_std_floor: float = 0.0

    # CRL
    logsumexp_penalty_coeff: float = 0.1
    repr_dim: int = 64
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"
    contrastive_temperature: float = 1.0

    # Network
    h_dim: int = 256
    n_hidden: int = 2
    use_layer_norm: bool = False
    use_relu: bool = False
    skip_connections: int = 0  # CRL paper uses 4 (every 4 layers add residual). 0 = off.
    # Per-actor overrides. None = inherit from the shared flags above.
    # Use these to keep the encoders complex (swish + LN + skip) while giving the
    # actor a simple ReLU MLP — for "more textbook PPO" ablations.
    actor_use_relu: Optional[bool] = None
    actor_use_layer_norm: Optional[bool] = None
    actor_skip_connections: Optional[int] = None
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # Goal handling
    use_achieved_goal: bool = True
    # If True, wraps env to set done=1 when goal reached → episode terminates on success.
    # JaxGCRL default is done=0 always (full 1001-step episodes).
    terminate_on_success: bool = False
    # SA encoder input mode:
    #   "obs_full"   = [obs (incl. target_goal), achieved_goal]   (Mava convention)
    #   "state_only" = [state, achieved_goal]                     (CRL paper convention)
    # Note: when use_achieved_goal=False, "state_only" → just state; "obs_full" → just obs.
    sa_state_mode: Literal["obs_full", "state_only"] = "obs_full"
    # Actor input mode (target_goal embedded in obs_full is always stripped):
    #   "obs_full_ach" = [state, achieved_goal, goal]
    #   "state_goal"   = [state, goal]
    actor_input_mode: Literal["obs_full_ach", "state_goal"] = "obs_full_ach"

    # Adaptive entropy (SAC-style log_alpha SGD)
    use_adaptive_entropy: bool = False
    target_entropy: float = 2.0
    alpha_lr: float = 3e-4
    log_alpha_clip_min: float = 0.003
    log_alpha_clip_max: float = 0.05

    # Compatibility with run.py's utd_ratio computation (unused here).
    train_step_multiplier: int = 1
    unroll_length: int = 256  # alias for rollout_length used by run.py utd_ratio calc

    def check_config(self, config):
        assert config.num_envs * self.rollout_length % self.batch_size == 0, (
            "num_envs * rollout_length must be divisible by batch_size"
        )

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[Callable] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        # ---------------- Environment wrappers ----------------
        unwrapped_env = train_env
        if self.terminate_on_success:
            train_env = TerminateOnSuccessWrapper(train_env)
            eval_env = TerminateOnSuccessWrapper(eval_env)
        train_env_w = TrajectoryIdWrapper(train_env)
        train_env_w = envs.training.wrap(
            train_env_w,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env_w = TrajectoryIdWrapper(eval_env)
        eval_env_w = envs.training.wrap(
            eval_env_w,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        action_size = int(unwrapped_env.action_size)
        state_size = int(unwrapped_env.state_dim)
        goal_indices = tuple(int(i) for i in unwrapped_env.goal_indices)
        goal_size = len(goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == int(unwrapped_env.observation_size), (
            f"obs_size {obs_size} != env.observation_size {unwrapped_env.observation_size}"
        )

        # SA state size depends on sa_state_mode:
        #   obs_full   → [obs_full, achieved_goal]   = obs_size + goal_size  (Mava)
        #   state_only → [state, achieved_goal]      = state_size + goal_size  (CRL)
        sa_base_size = obs_size if self.sa_state_mode == "obs_full" else state_size
        if self.use_achieved_goal:
            sa_state_size = sa_base_size + goal_size
        else:
            sa_state_size = sa_base_size

        # Actor input size depends on actor_input_mode.
        # target_goal embedded in obs_full is always stripped (redundant).
        if self.actor_input_mode == "state_goal":
            # [state, goal]
            actor_input_size = state_size + goal_size
        else:
            # [state, achieved_goal, goal] or [state, goal] if !use_achieved_goal
            if self.use_achieved_goal:
                actor_input_size = state_size + goal_size + goal_size
            else:
                actor_input_size = state_size + goal_size

        # ---------------- Schedules ----------------
        env_steps_per_update = config.num_envs * self.rollout_length
        num_evals_after_init = max(config.num_evals - 1, 1)
        num_updates_total = int(np.ceil(config.total_env_steps / env_steps_per_update))
        num_updates_per_epoch = int(np.ceil(num_updates_total / num_evals_after_init))
        ent_horizon = self.ent_schedule_horizon or num_updates_total

        if self.lr_linear_decay:
            samples_per_update = config.num_envs * self.rollout_length
            mb_per_update = samples_per_update // self.batch_size
            grad_steps = num_updates_total * self.num_epochs * mb_per_update
            actor_lr_sched = optax.linear_schedule(self.actor_lr, self.lr_end, grad_steps)
            critic_lr_sched = optax.linear_schedule(self.q_lr, self.lr_end, grad_steps)
        else:
            actor_lr_sched = self.actor_lr
            critic_lr_sched = self.q_lr

        logging.info(
            "CPPO: total_updates=%d  per_eval=%d  env_steps_per_update=%d",
            num_updates_total,
            num_updates_per_epoch,
            env_steps_per_update,
        )

        # ---------------- Networks ----------------
        hidden_sizes = tuple([self.h_dim] * self.n_hidden)
        # Resolve actor overrides (None → inherit shared flag).
        a_relu = self.use_relu if self.actor_use_relu is None else self.actor_use_relu
        a_ln = self.use_layer_norm if self.actor_use_layer_norm is None else self.actor_use_layer_norm
        a_skip = self.skip_connections if self.actor_skip_connections is None else self.actor_skip_connections
        actor = Actor(
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            use_layer_norm=a_ln,
            use_relu=a_relu,
            skip_connections=a_skip,
        )
        sa_encoder = SAEncoder(
            hidden_sizes=hidden_sizes,
            output_dim=self.repr_dim,
            use_layer_norm=self.use_layer_norm,
            use_relu=self.use_relu,
            skip_connections=self.skip_connections,
        )
        g_encoder = GoalEncoder(
            hidden_sizes=hidden_sizes,
            output_dim=self.repr_dim,
            use_layer_norm=self.use_layer_norm,
            use_relu=self.use_relu,
            skip_connections=self.skip_connections,
        )

        key = jax.random.PRNGKey(config.seed)
        key, env_key, eval_key, actor_key, sa_key, g_key, train_key = jax.random.split(key, 7)

        actor_params = actor.init(actor_key, jnp.zeros((1, actor_input_size)))
        sa_params = sa_encoder.init(sa_key, jnp.zeros((1, sa_state_size)), jnp.zeros((1, action_size)))
        g_params = g_encoder.init(g_key, jnp.zeros((1, goal_size)))

        actor_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(actor_lr_sched))
        sa_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(critic_lr_sched))
        g_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(critic_lr_sched))

        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_opt)
        sa_state = TrainState.create(apply_fn=sa_encoder.apply, params=sa_params, tx=sa_opt)
        g_state = TrainState.create(apply_fn=g_encoder.apply, params=g_params, tx=g_opt)

        training_state = TrainingState(
            env_steps=jnp.zeros((), dtype=jnp.int64),
            update_count=jnp.zeros((), dtype=jnp.int32),
            actor_state=actor_state,
            sa_encoder_state=sa_state,
            g_encoder_state=g_state,
            log_alpha=jnp.log(jnp.array(self.ent_coef, dtype=jnp.float32)),
        )

        # ---------------- Helpers ----------------
        loss_name = self.contrastive_loss_fn
        ene_name = self.energy_fn
        temp = self.contrastive_temperature
        lse_pen = self.logsumexp_penalty_coeff
        ach_g = self.use_achieved_goal
        clip_eps = self.clip_eps
        gamma = self.discounting
        K = self.num_mc_samples
        mc_floor = self.mc_std_floor
        ent_start = self.ent_coef
        ent_end = self.ent_coef_end
        use_adapt_ent = self.use_adaptive_entropy
        target_ent = self.target_entropy
        alpha_lr_ = self.alpha_lr
        log_alpha_lo = jnp.log(self.log_alpha_clip_min)
        log_alpha_hi = jnp.log(self.log_alpha_clip_max)

        def split_obs(obs):
            """obs: (..., obs_size).  Returns (obs_full, target_goal, achieved_goal).

            Mava convention: obs_full kept as-is (already contains target_goal embedded).
            target_goal = obs[..., -goal_size:]
            achieved_goal = obs[..., goal_indices]
            """
            target_goal = obs[..., -goal_size:]
            achieved_goal = obs[..., jnp.array(goal_indices)]
            return obs, target_goal, achieved_goal

        def build_actor_input(obs_full, achieved_goal, target_goal):
            """Actor input. target_goal embedded in obs_full is always stripped.

            actor_input_mode == "state_goal":  [state, goal]
            actor_input_mode == "obs_full_ach":[state, achieved_goal, goal]
                                               or [state, goal] if !use_achieved_goal
            """
            state = obs_full[..., :state_size]
            if self.actor_input_mode == "state_goal":
                return jnp.concatenate([state, target_goal], axis=-1)
            if ach_g:
                return jnp.concatenate([state, achieved_goal, target_goal], axis=-1)
            return jnp.concatenate([state, target_goal], axis=-1)

        def build_sa_state(obs_full, achieved_goal):
            """SA encoder input.

            sa_state_mode == "obs_full":  [obs_full, achieved_goal]    (Mava)
            sa_state_mode == "state_only":[state,    achieved_goal]    (CRL paper)
            """
            base = obs_full if self.sa_state_mode == "obs_full" else obs_full[..., :state_size]
            if ach_g:
                return jnp.concatenate([base, achieved_goal], axis=-1)
            return base

        def actor_sample(actor_params, obs, key):
            """Sample action from policy. Returns (action_tanh, x_t, log_prob)."""
            state, target, ach = split_obs(obs)
            inp = build_actor_input(state, ach, target)
            means, log_stds = actor.apply(actor_params, inp)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            x_t = means + stds * noise
            action = nn.tanh(x_t)
            log_prob = (-0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
            return action, x_t, log_prob

        # ---------------- Rollout ----------------
        def actor_step_for_rollout(carry, _):
            env_state, key, actor_params = carry
            key, sample_key = jax.random.split(key)
            action, x_t, log_prob = actor_sample(actor_params, env_state.obs, sample_key)
            n_state = train_env_w.step(env_state, action)
            extras = {
                "state_extras": {
                    "truncation": n_state.info["truncation"],
                    "traj_id": n_state.info["traj_id"],
                }
            }
            transition = Transition(
                observation=env_state.obs,
                action=action,
                x_t=x_t,
                log_prob=log_prob,
                reward=n_state.reward,
                discount=1.0 - n_state.done,
                extras=extras,
            )
            return (n_state, key, actor_params), transition

        def collect_rollout(env_state, key, actor_params):
            (env_state, key, _), traj = jax.lax.scan(
                actor_step_for_rollout,
                (env_state, key, actor_params),
                None,
                length=self.rollout_length,
            )
            return env_state, key, traj

        # ---------------- Hindsight relabeling ----------------
        # Operates per-env trajectory of length T = rollout_length.
        def her_relabel(transition, sample_key):
            """
            transition fields shape: (T, ...).  Returns dict with shape (T-1, ...).
            Goal sampled from future achieved goal in same trajectory.
            """
            T = transition.observation.shape[0]
            arr = jnp.arange(T)
            future_mask = jnp.array(arr[:, None] < arr[None], dtype=jnp.float32)
            disc = gamma ** jnp.array(arr[None] - arr[:, None], dtype=jnp.float32)
            probs = future_mask * disc
            traj_id = transition.extras["state_extras"]["traj_id"]
            same_traj = jnp.equal(traj_id[:, None], traj_id[None, :]).astype(jnp.float32)
            probs = probs * same_traj + jnp.eye(T) * 1e-5
            goal_idx = jax.random.categorical(sample_key, jnp.log(probs))
            future_obs = jnp.take(transition.observation, goal_idx[:-1], axis=0)
            relabeled_goal = future_obs[:, jnp.array(goal_indices)]  # (T-1, goal_size)

            obs = transition.observation[:-1]  # (T-1, obs_size)
            achieved = obs[:, jnp.array(goal_indices)]

            return {
                "obs": obs,
                "achieved_goal": achieved,
                "target_goal": obs[:, -goal_size:],  # original env target — used by actor/advantage
                "relabeled_goal": relabeled_goal,  # HER future-achieved — used by critic only
                "action": transition.action[:-1],
                "x_t": transition.x_t[:-1],
                "old_log_prob": transition.log_prob[:-1],
                "reward": transition.reward[:-1],
                "discount": transition.discount[:-1],
            }

        # ---------------- Critic loss ----------------
        def critic_loss_fn(sa_params, g_params, sa_state_arr, action, goal):
            sa_repr = sa_encoder.apply(sa_params, sa_state_arr, action)
            g_repr = g_encoder.apply(g_params, goal)
            logits = compute_logits(ene_name, sa_repr, g_repr) / temp
            loss = contrastive_loss_fn_(loss_name, logits, lse_pen)
            metrics = compute_contrastive_metrics(logits)
            return loss, {"critic_loss": loss, **metrics}

        # ---------------- Actor loss ----------------
        def actor_loss_fn(actor_params, batch, advantage, ent_coef):
            obs_full = batch["obs"]
            ach = batch["achieved_goal"]
            # Actor uses the REAL env goal — matches the rollout actor input,
            # so the PPO ratio compares like-with-like. (Mava convention.)
            goal = batch["target_goal"]
            x_t = batch["x_t"]
            old_log_prob = batch["old_log_prob"]

            inp = build_actor_input(obs_full, ach, goal)
            means, log_stds = actor.apply(actor_params, inp)
            stds = jnp.exp(log_stds)
            noise = (x_t - means) / (stds + 1e-8)
            log_prob = (-0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)

            ratio = jnp.exp(log_prob - old_log_prob)
            ratio_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            ppo_loss = jnp.mean(jnp.maximum(-advantage * ratio, -advantage * ratio_clipped))

            entropy = (log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
            actor_loss = ppo_loss - ent_coef * entropy

            return actor_loss, {
                "actor_loss": actor_loss,
                "ppo_loss": ppo_loss,
                "entropy": entropy,
                "ratio_mean": ratio.mean(),
                "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)),
            }

        # ---------------- Advantage (no-grad) ----------------
        def compute_advantage(actor_params, sa_params, g_params, batch, key):
            obs_full = batch["obs"]
            ach = batch["achieved_goal"]
            # Advantage uses REAL env goal — Q(s,a,g_real) vs V(s,g_real) under
            # actor(·|s, g_real). Matches rollout/eval conditioning. (Mava convention.)
            goal = batch["target_goal"]
            action = batch["action"]
            sa_in = build_sa_state(obs_full, ach)

            actor_in = build_actor_input(obs_full, ach, goal)
            means, log_stds = actor.apply(actor_params, actor_in)
            stds = jnp.exp(log_stds)
            mc_stds = jnp.maximum(stds, mc_floor) if mc_floor > 0 else stds

            sa_repr_taken = sa_encoder.apply(sa_params, sa_in, action)
            g_repr = g_encoder.apply(g_params, goal)
            q_taken = energy_fn_(ene_name, sa_repr_taken, g_repr)

            mc_noise = jax.random.normal(key, shape=(K,) + means.shape, dtype=means.dtype)
            mc_xt = means[None] + mc_stds[None] * mc_noise
            mc_actions = nn.tanh(mc_xt)

            def q_for_sample(a_):
                sa_r = sa_encoder.apply(sa_params, sa_in, a_)
                return energy_fn_(ene_name, sa_r, g_repr)

            mc_q = jax.vmap(q_for_sample)(mc_actions)  # (K, B)
            v_mc = mc_q.mean(axis=0)
            adv = q_taken - v_mc
            return adv, q_taken, v_mc

        # ---------------- Single update step ----------------
        @jax.jit
        def update_step(training_state, env_state, key):
            key, rollout_key, her_key, adv_key, train_key, perm_key = jax.random.split(key, 6)

            actor_p = training_state.actor_state.params
            sa_p = training_state.sa_encoder_state.params
            g_p = training_state.g_encoder_state.params

            # Rollout
            env_state, _, traj = collect_rollout(env_state, rollout_key, actor_p)

            # Trajectory shape: (T, num_envs, ...). Swap to (num_envs, T, ...) for per-env HER.
            traj_per_env = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj)

            her_keys = jax.random.split(her_key, traj_per_env.observation.shape[0])
            relabeled = jax.vmap(her_relabel)(traj_per_env, her_keys)
            # shape: (num_envs, T-1, ...)

            # Flatten env+time into batch dimension (Fortran order to interleave envs).
            relabeled = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), relabeled
            )

            B_total = relabeled["obs"].shape[0]
            B_use = (B_total // self.batch_size) * self.batch_size
            relabeled = jax.tree_util.tree_map(lambda x: x[:B_use], relabeled)

            # Compute advantages over full data in batch_size chunks.
            adv_batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), relabeled
            )

            def adv_scan(carry_key, batch):
                k, sub_k = jax.random.split(carry_key)
                adv, q, v = compute_advantage(actor_p, sa_p, g_p, batch, sub_k)
                return k, {"advantage": adv, "q_taken": q, "v_mc": v}

            _, adv_results = jax.lax.scan(adv_scan, adv_key, adv_batched)
            advantages = adv_results["advantage"].reshape(-1)
            q_taken_all = adv_results["q_taken"].reshape(-1)
            v_mc_all = adv_results["v_mc"].reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            relabeled["advantage"] = advantages

            # Entropy schedule (linear; only when not adaptive)
            frac = jnp.clip(training_state.update_count / jnp.maximum(ent_horizon, 1), 0.0, 1.0)
            sched_ent_coef = ent_start + (ent_end - ent_start) * frac
            init_ent_coef = jnp.where(use_adapt_ent, jnp.exp(training_state.log_alpha), sched_ent_coef)

            # Train epochs
            def minibatch_step(carry, batch):
                a_state, sa_state_, g_state_, log_alpha_, ent_coef = carry

                # Critic
                def c_loss_fn(c_params, batch):
                    sa_in = build_sa_state(batch["obs"], batch["achieved_goal"])
                    return critic_loss_fn(c_params["sa"], c_params["g"], sa_in, batch["action"], batch["relabeled_goal"])

                c_params = {"sa": sa_state_.params, "g": g_state_.params}
                (_, c_info), c_grads = jax.value_and_grad(c_loss_fn, has_aux=True)(c_params, batch)
                sa_state_ = sa_state_.apply_gradients(grads=c_grads["sa"])
                g_state_ = g_state_.apply_gradients(grads=c_grads["g"])

                # Actor
                (_, a_info), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                    a_state.params, batch, batch["advantage"], ent_coef
                )
                a_state = a_state.apply_gradients(grads=a_grads)

                # Adaptive entropy: SGD on log_alpha to push entropy toward target.
                # grad = (entropy - target_entropy); when entropy > target, decrease alpha.
                entropy_now = a_info["entropy"]
                alpha_grad = entropy_now - target_ent
                new_log_alpha = log_alpha_ - alpha_lr_ * alpha_grad
                new_log_alpha = jnp.clip(new_log_alpha, log_alpha_lo, log_alpha_hi)
                new_log_alpha = jnp.where(use_adapt_ent, new_log_alpha, log_alpha_)
                new_ent_coef = jnp.where(use_adapt_ent, jnp.exp(new_log_alpha), ent_coef)

                metrics = {**c_info, **a_info, "ent_coef": ent_coef, "log_alpha": new_log_alpha}
                return (a_state, sa_state_, g_state_, new_log_alpha, new_ent_coef), metrics

            def train_epoch(carry, _):
                a_state, sa_state_, g_state_, log_alpha_, ent_coef, key, flat_data = carry
                key, perm_k = jax.random.split(key)
                perm = jax.random.permutation(perm_k, flat_data["obs"].shape[0])
                shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)
                batched = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), shuffled
                )
                (a_state, sa_state_, g_state_, log_alpha_, ent_coef), m = jax.lax.scan(
                    minibatch_step, (a_state, sa_state_, g_state_, log_alpha_, ent_coef), batched
                )
                return (a_state, sa_state_, g_state_, log_alpha_, ent_coef, key, flat_data), m

            (a_state, sa_state_, g_state_, final_log_alpha, _, _, _), epoch_metrics = jax.lax.scan(
                train_epoch,
                (training_state.actor_state, training_state.sa_encoder_state, training_state.g_encoder_state,
                 training_state.log_alpha, init_ent_coef, train_key, relabeled),
                None,
                self.num_epochs,
            )

            new_state = TrainingState(
                env_steps=training_state.env_steps + env_steps_per_update,
                update_count=training_state.update_count + 1,
                actor_state=a_state,
                sa_encoder_state=sa_state_,
                g_encoder_state=g_state_,
                log_alpha=final_log_alpha,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
            metrics["advantage_mean"] = advantages.mean()
            metrics["advantage_std"] = advantages.std()
            metrics["q_taken_mean"] = q_taken_all.mean()
            metrics["v_mc_mean"] = v_mc_all.mean()
            metrics["rollout_reward_mean"] = traj.reward.mean()
            return new_state, env_state, metrics

        # ---------------- Eval ----------------
        def eval_actor_step(ts, env, env_state, extra_fields):
            means, log_stds = actor.apply(ts.actor_state.params, _build_actor_input_full(env_state.obs))
            stds = jnp.exp(log_stds)
            # Stochastic eval matches paper.
            # Use a key derived from env_state to keep determinism per call.
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=means.shape, dtype=means.dtype)
            x_t = means + stds * noise
            action = nn.tanh(x_t)
            n_state = env.step(env_state, action)
            return n_state, Transition(
                observation=env_state.obs,
                action=action,
                x_t=x_t,
                log_prob=jnp.zeros(action.shape[:-1]),
                reward=n_state.reward,
                discount=1.0 - n_state.done,
                extras={"state_extras": {x: n_state.info[x] for x in extra_fields}},
            )

        def _build_actor_input_full(obs):
            state, target, ach = split_obs(obs)
            return build_actor_input(state, ach, target)

        evaluator = ActorEvaluator(
            actor_step=eval_actor_step,
            eval_env=eval_env_w,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_key,
        )

        # ---------------- Main training loop ----------------
        env_state = jax.jit(train_env_w.reset)(jax.random.split(env_key, config.num_envs))

        def make_policy(params):
            def policy(obs, rng):
                state, target, ach = split_obs(obs)
                inp = build_actor_input(state, ach, target)
                means, log_stds = actor.apply(params, inp)
                stds = jnp.exp(log_stds)
                noise = jax.random.normal(rng, shape=means.shape, dtype=means.dtype)
                return nn.tanh(means + stds * noise)
            return policy

        # Initial eval
        eval_metrics = evaluator.run_evaluation(training_state, {})
        progress_fn(
            int(training_state.env_steps),
            eval_metrics,
            make_policy,
            training_state.actor_state.params,
            unwrapped_env,
            do_render=False,
        )

        loop_key = train_key
        for eval_idx in range(num_evals_after_init):
            t0 = time.time()
            for _ in range(num_updates_per_epoch):
                loop_key, sub_key = jax.random.split(loop_key)
                training_state, env_state, train_metrics = update_step(training_state, env_state, sub_key)
            jax.block_until_ready(training_state.env_steps)
            sps = (num_updates_per_epoch * env_steps_per_update) / (time.time() - t0)

            train_metrics = jax.tree_util.tree_map(lambda x: float(np.asarray(x)), train_metrics)
            train_metrics = {f"training/{k}": v for k, v in train_metrics.items()}
            train_metrics["training/sps"] = sps

            eval_metrics = evaluator.run_evaluation(training_state, train_metrics)
            progress_fn(
            int(training_state.env_steps),
            eval_metrics,
            make_policy,
            training_state.actor_state.params,
            unwrapped_env,
            do_render=False,
        )

        params = (
            training_state.actor_state.params,
            training_state.sa_encoder_state.params,
            training_state.g_encoder_state.params,
        )
        logging.info("CPPO total steps: %d", int(training_state.env_steps))
        return None, params, eval_metrics
