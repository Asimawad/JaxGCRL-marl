"""Hydra entry point for the mava-hosted JaxGCRL agents.

Usage:
    python -m mava.jaxgcrl.run --config-name=cppo env=ant
    python -m mava.jaxgcrl.run --config-name=crl env=humanoid system.actor_lr=1e-4

The agent (CPPO vs CRL) is selected by the top-level config file you load
(`cppo.yaml` or `crl.yaml`); the YAML's `agent` field tells this script which
class to construct.

The agents themselves are unchanged — this file only:
  1. Resolves a hydra `DictConfig` into a `RunConfig` dataclass + agent kwargs.
  2. Builds the env via `mava.jaxgcrl.utils.env.create_env`.
  3. Wraps `MavaLogger.log` as the `progress_fn` callback that the agent's
     `train_fn` expects (replacing JaxGCRL's `MetricsRecorder.progress`).
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from mava.jaxgcrl.utils.logger import LogEvent, MavaLogger
from mava.jaxgcrl.utils.env import create_env
from mava.jaxgcrl.utils.config import RunConfig


_AGENT_REGISTRY = {
    "cppo": "mava.jaxgcrl.agents.cppo.CPPO",
    "crl": "mava.jaxgcrl.agents.crl.CRL",
}


def _resolve_agent_cls(name: str):
    """Look up an agent class by name without forcing both to import at module load."""
    if name not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent {name!r}. Choose from {list(_AGENT_REGISTRY)}.")
    mod_path, cls_name = _AGENT_REGISTRY[name].rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def _make_progress_fn(logger: MavaLogger):
    """Adapt JaxGCRL's `progress_fn(num_steps, metrics, make_policy, params, env, do_render)`
    to MavaLogger.

    JaxGCRL emits a single `metrics` dict per eval that contains both `eval/*`
    and `training/*` keys. We split them and dispatch to LogEvent.EVAL and
    LogEvent.TRAIN respectively (and `timestep` to LogEvent.MISC).
    """
    state = {"eval_idx": 0, "last_metrics": {}, "last_step": 0}

    def progress_fn(num_steps, metrics, make_policy=None, params=None, env=None, do_render=False):  # noqa: ARG001
        eval_only = {}
        train_only = {}
        for k, v in metrics.items():
            if k.startswith("eval/"):
                eval_only[k[len("eval/"):]] = v
            elif k.startswith("training/"):
                train_only[k[len("training/"):]] = v
            else:
                eval_only[k] = v  # uncategorised goes with eval

        logger.log({"timestep": int(num_steps)}, int(num_steps), state["eval_idx"], LogEvent.MISC)
        if train_only:
            logger.log(train_only, int(num_steps), state["eval_idx"], LogEvent.TRAIN)
        if eval_only:
            logger.log(eval_only, int(num_steps), state["eval_idx"], LogEvent.EVAL)
            state["last_metrics"] = dict(eval_only)
            state["last_step"] = int(num_steps)
        state["eval_idx"] += 1

    return progress_fn, state


def run_experiment(cfg: DictConfig) -> float:
    """Resolve cfg, build env + agent + logger, run training, return scalar metric."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -------- Env --------
    train_env = create_env(cfg.env.env_name, backend=cfg.env.get("backend", None))
    if cfg.env.get("eval_env", None):
        eval_env = create_env(cfg.env.eval_env, backend=cfg.env.get("backend", None))
    else:
        eval_env = train_env

    # -------- Logger --------
    logger = MavaLogger(cfg)
    logger.log_config(OmegaConf.to_container(cfg, resolve=True))

    progress_fn, progress_state = _make_progress_fn(logger)

    # -------- RunConfig from cfg.run --------
    run_dict = OmegaConf.to_container(cfg.run, resolve=True)
    # `env` is required by RunConfig and is referenced from cfg.env.env_name
    run_dict.setdefault("env", cfg.env.env_name)
    run_dict.setdefault("backend", cfg.env.get("backend", None))
    run_cfg = RunConfig(**run_dict)

    # -------- Agent --------
    agent_name = cfg.get("agent", "cppo")
    AgentCls = _resolve_agent_cls(agent_name)
    agent_kwargs = OmegaConf.to_container(cfg.system, resolve=True)
    agent = AgentCls(**agent_kwargs)

    # -------- Train --------
    _, _, _ = agent.train_fn(
        config=run_cfg,
        train_env=train_env,
        eval_env=eval_env,
        progress_fn=progress_fn,
    )

    # -------- Absolute metric --------
    # Re-emit the LAST eval metrics as a LogEvent.ABSOLUTE event. This populates
    # the marl-eval `absolute_metrics` block in metrics.json (a "final headline"
    # number per (env, algo, seed) used by rliable / sample-efficiency plotters).
    # Note: this is the simple version — it copies the last eval values rather
    # than running a separate higher-precision re-eval.
    if progress_state["last_metrics"]:
        logger.log(
            progress_state["last_metrics"],
            progress_state["last_step"],
            progress_state["eval_idx"],
            LogEvent.ABSOLUTE,
        )

    logger.stop()

    # Return a scalar so optuna sweepers can rank trials.
    last = progress_state["last_metrics"]
    for key in ("episode_success_any", "episode_success", "episode_reward"):
        if key in last:
            try:
                return float(last[key])
            except Exception:
                pass
    return 0.0


@hydra.main(config_path="../configs/jaxgcrl", config_name="cppo", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    return run_experiment(cfg)


if __name__ == "__main__":
    hydra_entry_point()
