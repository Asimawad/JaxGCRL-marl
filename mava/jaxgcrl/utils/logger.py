"""Mava-compatible logger for the JaxGCRL port — without the heavy Mava deps.

Mava's `mava.utils.logger` pulls in jumanji, tensorflow_probability, neptune,
tensorboard_logger, etc. The JaxGCRL agents only need a subset. This module
replicates the public API of `mava.utils.logger` (`LogEvent`, `MavaLogger`,
`log_config`, `log`, `stop`) and ships three backends:

  - ConsoleLogger — coloured single-line-per-event output (matches Mava format)
  - WandbLogger   — `wandb.init / wandb.log` per LogEvent
  - JsonLogger    — wraps `marl_eval.json_tools.JsonLogger` so the resulting
                    `metrics.json` plays with marl-eval's plotting tools.

Public surface:
    LogEvent          enum of {ACT, TRAIN, EVAL, ABSOLUTE, MISC}
    MavaLogger(cfg)   constructed from a hydra DictConfig
        .log_config(cfg_dict)
        .log(metrics_dict, t, t_eval, event)
        .stop()
"""

import logging
from enum import Enum
from typing import Any, Dict, List

import jax
import numpy as np
from omegaconf import OmegaConf

try:
    from colorama import Fore, Style
except ImportError:  # graceful fallback if colorama isn't installed
    class _Dummy:
        def __getattr__(self, _):
            return ""
    Fore = Style = _Dummy()  # type: ignore


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"


_EVENT_COLOURS = {
    LogEvent.TRAIN: Fore.MAGENTA,
    LogEvent.EVAL: Fore.GREEN,
    LogEvent.ABSOLUTE: Fore.BLUE,
    LogEvent.ACT: Fore.CYAN,
    LogEvent.MISC: Fore.YELLOW,
}


def _flatten(d: Dict[str, Any], sep: str = "/", parent_key: str = "") -> Dict[str, Any]:
    """Flatten a nested dict with `sep` as the key separator."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(_flatten(v, sep=sep, parent_key=kk))
        else:
            out[kk] = v
    return out


def _to_scalar(v):
    if isinstance(v, (jax.Array, np.ndarray)):
        try:
            return v.item() if v.ndim == 0 else float(np.mean(np.asarray(v)))
        except Exception:
            return float(np.asarray(v).flatten()[0])
    return v


class _ConsoleLogger:
    """One coloured line per `log_dict` call, formatted `EVENT - k1: v1 | k2: v2 | …`."""

    def __init__(self) -> None:
        self.logger = logging.getLogger()

        # Hydra also installs a StreamHandler on the root logger when @hydra.main
        # runs — that's the `[YYYY-MM-DD HH:MM:SS][root][INFO] - …` line you see.
        # If we add ours alongside, every record prints twice. Strip any existing
        # StreamHandlers (but keep FileHandlers so hydra's per-run log file still
        # gets written) and install our coloured one as the sole console output.
        kept = []
        for h in self.logger.handlers:
            if isinstance(h, logging.FileHandler):
                kept.append(h)
            elif isinstance(h, logging.StreamHandler) and getattr(h, "_jaxgcrl_console", False):
                kept.append(h)  # already ours, idempotent on re-init
        self.logger.handlers = kept

        if not any(getattr(h, "_jaxgcrl_console", False) for h in self.logger.handlers):
            ch = logging.StreamHandler()
            ch._jaxgcrl_console = True  # type: ignore[attr-defined]
            ch.setFormatter(logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s{Style.RESET_ALL}", "%H:%M:%S"))
            self.logger.addHandler(ch)
        self.logger.setLevel("INFO")

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        flat = _flatten(data, sep=" ")
        colour = _EVENT_COLOURS[event]
        parts: List[str] = []
        for k, v in flat.items():
            kk = k.replace("_", " ").capitalize()
            v = _to_scalar(v)
            parts.append(f"{kk}: {v:.3f}" if isinstance(v, float) else f"{kk}: {v}")
        msg = " | ".join(parts)
        self.logger.info(f"{colour}{Style.BRIGHT}{event.value.upper()} - step={step} | {msg}{Style.RESET_ALL}")

    def log_config(self, cfg: Dict[str, Any]) -> None:
        colour = _EVENT_COLOURS[LogEvent.MISC]
        self.logger.info(f"{colour}{Style.BRIGHT}CONFIG{Style.RESET_ALL}")
        try:
            from rich.pretty import pprint
            pprint(cfg)
        except ImportError:
            import pprint as _pp
            self.logger.info(_pp.pformat(cfg))

    def stop(self) -> None:
        return None


class _WandbLogger:
    """Wraps the standard `wandb.init / wandb.log` API. Reads run metadata
    from the hydra config via fields like `run.wandb_project_name`."""

    def __init__(self, cfg) -> None:
        import wandb  # local import — keeps the module importable on machines without wandb

        self._wandb = wandb
        run = cfg.get("run", {}) if hasattr(cfg, "get") else cfg.run
        wandb_cfg = cfg.logger.loggers.wandb

        # Disabled mode for tests / no-network runs.
        mode = OmegaConf.to_container(run, resolve=True).get("wandb_mode", "online")  # type: ignore[arg-type]

        self.run = wandb.init(
            project=wandb_cfg.get("project", None) or run.get("wandb_project_name", "jaxgcrl"),
            entity=wandb_cfg.get("entity", None),
            group=wandb_cfg.get("group", None) or run.get("wandb_group", None),
            name=wandb_cfg.get("run_name", None) or run.get("exp_name", None),
            tags=list(wandb_cfg.get("tags", []) or []),
            mode=mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        flat = _flatten(data, sep="/")
        payload = {f"{event.value}/{k}": _to_scalar(v) for k, v in flat.items()}
        self._wandb.log(payload, step=int(step))

    def log_config(self, _cfg: Dict[str, Any]) -> None:
        # config was already passed at wandb.init time
        return None

    def stop(self) -> None:
        try:
            self.run.finish()
        except Exception:
            pass


class _JsonLogger:
    """marl-eval-compatible JSON writer (env→task→algo→seed_N→step_M tree).

    Wraps `marl_eval.json_tools.JsonLogger`. Writes ONLY on `LogEvent.EVAL`
    and `LogEvent.ABSOLUTE`. Filters incoming metrics through a key allowlist
    (with optional rename) defined in YAML — usually mapping JaxGCRL's
    `episode_reward` → marl-eval's `mean_episode_return` so plotters work.
    """

    def __init__(self, cfg) -> None:
        from marl_eval.json_tools import JsonLogger as _MELJsonLogger

        json_cfg = cfg.logger.loggers.json
        run = cfg.run

        path = json_cfg.get("path", None) or f"results/json/{cfg.env.env_name}"
        task_name = json_cfg.get("task_name", None) or cfg.env.env_name
        env_name = json_cfg.get("env_name", None) or cfg.env.env_name
        algo = cfg.get("agent", "agent")
        seed = int(run.seed)

        self._logger = _MELJsonLogger(
            path=path,
            algorithm_name=algo,
            task_name=task_name,
            environment_name=env_name,
            seed=seed,
        )

        # Allowlist: dict mapping incoming key -> json-output key (or None to keep).
        # Defaults match marl-eval plotting conventions.
        default_map = {
            "episode_reward": "mean_episode_return",
            "episode_success_any": "mean_episode_success",
            "sps": "steps_per_second",
        }
        user_map = json_cfg.get("metrics_to_log", None)
        self._key_map: Dict[str, str] = (
            dict(user_map) if user_map else default_map
        )

    def log_dict(self, data: Dict[str, Any], step: int, eval_step: int, event: LogEvent) -> None:
        if event not in (LogEvent.EVAL, LogEvent.ABSOLUTE):
            return
        is_absolute = event == LogEvent.ABSOLUTE
        for k_in, k_out in self._key_map.items():
            if k_in not in data:
                continue
            v = _to_scalar(data[k_in])
            if not isinstance(v, (int, float)):
                continue
            self._logger.write(int(step), k_out, float(v), eval_step, is_absolute)

    def log_config(self, _cfg: Dict[str, Any]) -> None:
        return None

    def stop(self) -> None:
        return None  # marl_eval.JsonLogger flushes on every write


class MavaLogger:
    """Mava-compatible logger that fans out to enabled backends.

    Reads `cfg.logger.loggers.{console,wandb}.enabled` to pick what to construct.
    Other backend keys (neptune, tensorboard, json) are ignored — this logger is
    intentionally minimal for the JaxGCRL port.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backends: List[Any] = []

        loggers_cfg = cfg.logger.loggers if hasattr(cfg, "logger") else {}
        if loggers_cfg.console.get("enabled", True):
            self.backends.append(_ConsoleLogger())
        if loggers_cfg.wandb.get("enabled", False):
            try:
                self.backends.append(_WandbLogger(cfg))
            except Exception as e:  # missing wandb api key, network down, etc.
                logging.warning("WandbLogger disabled: %s", e)
        if loggers_cfg.json.get("enabled", False):
            try:
                self.backends.append(_JsonLogger(cfg))
            except Exception as e:
                logging.warning("JsonLogger disabled: %s", e)

    def log_config(self, cfg_dict: Dict[str, Any]) -> None:
        for b in self.backends:
            b.log_config(cfg_dict)

    def log(self, metrics: Dict[str, Any], t: int, t_eval: int, event: LogEvent) -> None:
        # Drop non-numeric / unprocessable fields silently; reduce arrays to scalar.
        clean = {}
        for k, v in metrics.items():
            try:
                clean[k] = _to_scalar(v)
            except Exception:
                continue
        for b in self.backends:
            b.log_dict(clean, t, t_eval, event)

    def stop(self) -> None:
        for b in self.backends:
            try:
                b.stop()
            except Exception:
                pass
