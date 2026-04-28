"""Run-level configuration dataclass for the mava-hosted JaxGCRL port.

Differences from `jaxgcrl/utils/config.py`:
- No `tyro` (the hydra entry constructs this dataclass via `RunConfig(**dict)`).
- No `AgentConfig` union — agent kwargs come from a separate `cfg.system` block in the YAML.
- No `Config` wrapper class.

The agents' `train_fn(config: RunConfig, ...)` signature is unchanged, so this
dataclass must keep all the same field names and defaults as the original.
"""

from typing import Literal, Optional

from flax.struct import dataclass

from .env import legal_envs


@dataclass
class RunConfig:
    """Run-level args passed to every agent's `train_fn`.

    Field documentation matches the original at `jaxgcrl/utils/config.py`.
    """

    env: Literal[legal_envs]

    total_env_steps: int = 50_000_000
    episode_length: int = 1001
    eval_env: Optional[Literal[legal_envs]] = None
    num_envs: int = 256
    num_eval_envs: int = 256
    action_repeat: int = 1
    num_evals: int = 200
    seed: int = 0
    backend: Optional[Literal["mjx", "spring", "positional", "generalized"]] = None

    # wandb / experiment metadata (consumed by the hydra entry, not the agents)
    exp_name: str = "run"
    log_wandb: bool = True
    wandb_project_name: str = "jaxgcrl"
    wandb_group: str = "."
    wandb_mode: Literal["online", "offline"] = "online"

    visualization_interval: int = 5
    vis_length: int = 1000
    checkpoint_logdir: Optional[str] = None
    max_devices_per_host: int = 1
    cuda: bool = True
