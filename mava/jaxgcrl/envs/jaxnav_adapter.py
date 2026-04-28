"""JaxNav adapter — exposes JaxNav (jaxmarl multi-agent) as a brax-style
single-agent env that JaxGCRL's CPPO / CRL agents can consume.

The agents expect a brax `PipelineEnv`-shaped object with:
  - `reset(rng) -> brax.State`  / `step(state, action) -> brax.State`
  - `state_dim`, `goal_indices`, `observation_size`, `action_size`, `dt`, `sys`
  - Flat 1-D `obs` whose final `goal_size` dims are the target goal, and whose
    `goal_indices` positions are the achieved goal (current state in goal-space).

JaxNav's native obs (`lidar + vel + goal_dist + goal_orient + λ`, default 205-D)
omits the agent's (x, y) and the goal's (x, y), so HER would have nothing
meaningful to relabel against. We append both:

    obs = [ agents_view | pos_xy | goal_xy ]
            205           2        2          = 209-D total

  state_dim       = 207                (= agents_view + pos_xy)
  goal_size       = 2                  (target_goal = goal_xy)
  goal_indices    = (205, 206)         (achieved_goal = pos_xy, sliced from state)
  observation_size = 209

Constraints:
  - num_agents=1 is enforced (multi-agent CPPO is out of scope).
  - Continuous action space (`act_type="Continuous"` by default).
  - Truncation detection is approximate: we set `truncation=1` on the LAST
    step of an episode when goal wasn't reached. Good enough for HER.

Requires `jaxmarl` to be installed in the venv (it isn't a hard dep of
jaxgcrl; the import is deferred so the adapter only loads when `env=jaxnav`
is selected).
"""

from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State as BraxState
from flax import struct


def _import_jaxnav():
    """Defer the jaxmarl import so machines without jaxmarl can still
    import the rest of mava.jaxgcrl."""
    from mava.jaxmarl.environments.jaxnav.jaxnav_env import JaxNav  # noqa: I001
    return JaxNav


@struct.dataclass
class _JaxNavWrapState:
    """Combined JaxNav env state + RNG key. Lives in BraxState.pipeline_state.
    JaxNav's step_env needs a key per call; brax's step(state, action)
    doesn't take one, so we thread the RNG through pipeline_state."""

    jaxnav_state: Any
    key: jnp.ndarray
    step: jnp.ndarray   # int32 — episode step counter (for truncation detection)


class JaxNavSingleAgent(Env):
    """Brax-compatible single-agent wrapper over JaxNav (1 agent, continuous)."""

    def __init__(self, **jaxnav_kwargs):
        JaxNav = _import_jaxnav()
        jaxnav_kwargs["num_agents"] = 1               # we don't support multi-agent
        jaxnav_kwargs.setdefault("act_type", "Continuous")
        self._env = JaxNav(**jaxnav_kwargs)

        # Dimensions:
        #   agents_view (jaxnav default) = lidar + 5 (vel x2, goal_dist, goal_orient, λ)
        self._base_obs_size = int(self._env.lidar_num_beams) + 5
        self._goal_size = 2                            # (x, y)
        self._state_dim = self._base_obs_size + self._goal_size   # +2 for achieved (x, y)
        self._observation_size = self._state_dim + self._goal_size
        self._action_size = 2                          # continuous: [v, omega]

        # achieved_goal = obs[..., goal_indices]; we put pos_xy at the end of the
        # state portion, just before target_goal.
        self._goal_indices = (
            self._base_obs_size,
            self._base_obs_size + 1,
        )

        self._max_steps = int(self._env.max_steps)

    # --- Attributes that JaxGCRL train_fn reads on the unwrapped env -------
    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def goal_indices(self):
        return self._goal_indices

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def dt(self) -> float:
        return float(self._env.dt)

    @property
    def sys(self):
        # brax envs expose sys for rendering; jaxnav has no brax system.
        # Returning None means rendering helpers will skip jaxnav silently.
        return None

    @property
    def backend(self) -> str:
        return "jaxmarl"

    # --- Core ---------------------------------------------------------------
    def _compose_obs(self, agents_view_dict, env_state) -> jnp.ndarray:
        """Build the flat 1-D obs vector: [agents_view | pos_xy | goal_xy]."""
        agents_view = agents_view_dict["agent_0"]      # (205,)
        pos = env_state.pos[0, :2]                     # (2,)  current x,y
        goal = env_state.goal[0]                       # (2,)  target x,y
        return jnp.concatenate([agents_view, pos, goal])

    def reset(self, rng: jnp.ndarray) -> BraxState:
        rng, reset_key = jax.random.split(rng)
        agents_view_dict, env_state = self._env.reset(reset_key)
        obs = self._compose_obs(agents_view_dict, env_state)
        wrap = _JaxNavWrapState(
            jaxnav_state=env_state,
            key=rng,
            step=jnp.int32(0),
        )
        # NOTE: brax's EpisodeWrapper / AutoResetWrapper add their own keys
        # to info and metrics (`steps`, `truncation`, `first_obs`,
        # `first_pipeline_state`, `reward`). We seed the keys we own
        # (`success`, `truncation`); the wrappers will add/overwrite the rest.
        return BraxState(
            pipeline_state=wrap,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics={},
            info={
                "success": jnp.float32(0.0),
                "truncation": jnp.float32(0.0),
            },
        )

    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        wrap: _JaxNavWrapState = state.pipeline_state
        key, step_key = jax.random.split(wrap.key)

        # JaxNav step_env wants a per-agent action dict.
        action_dict = {"agent_0": action}
        agents_view_dict, env_state, reward_dict, done_dict, info_dict = (
            self._env.step_env(step_key, wrap.jaxnav_state, action_dict)
        )

        obs = self._compose_obs(agents_view_dict, env_state)
        reward = jnp.asarray(reward_dict["agent_0"], dtype=jnp.float32)
        done = jnp.asarray(done_dict["__all__"], dtype=jnp.float32)

        # Success indicator: GoalR from info, may be per-agent.
        goal_r = info_dict.get("GoalR", jnp.zeros((1,), dtype=jnp.float32))
        goal_r = jnp.asarray(goal_r, dtype=jnp.float32)
        success = jnp.where(jnp.any(goal_r > 0), 1.0, 0.0).astype(jnp.float32)

        # Truncation: episode ended (done=1) but agent didn't reach the goal.
        new_step = wrap.step + 1
        truncation = jnp.where(
            (new_step >= self._max_steps) & (success < 0.5),
            1.0,
            0.0,
        ).astype(jnp.float32)

        new_wrap = _JaxNavWrapState(
            jaxnav_state=env_state,
            key=key,
            step=new_step,
        )

        # Preserve any wrapper-added keys (steps, first_obs, etc.) by merging
        # into the existing info/metrics rather than rebuilding from scratch.
        # We only OWN `success` and `truncation`; everything else passes through.
        new_info = {**state.info, "success": success, "truncation": truncation}
        return state.replace(
            pipeline_state=new_wrap,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
        )
