import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


class TerminateOnSuccessWrapper(Wrapper):
    """Sets `state.done = 1.0` when `metrics["success"] > 0`.

    JaxGCRL envs hardcode `done = 0.0` so trajectories always run the full
    episode_length. With this wrapper, episodes end as soon as the goal is
    reached, yielding cleaner advantage signal for on-policy PPO and a more
    meaningful `success_any` metric (no inflation from random touches).

    The outer wrappers (envs.training.wrap, EvalWrapper) consume `state.done`
    to drive resets, so this slots in cleanly upstream of them.
    """

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        done = jnp.asarray(nstate.done, dtype=jnp.float32)
        success = jnp.asarray(nstate.metrics.get("success", jnp.zeros_like(done)), dtype=jnp.float32)
        success_flag = (success > 0).astype(done.dtype)
        new_done = jnp.maximum(done, success_flag)
        return nstate.replace(done=new_done)
