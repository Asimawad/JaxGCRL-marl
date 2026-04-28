"""Slimmed env factory for the mava-hosted JaxGCRL port.

Only `create_env` and `legal_envs` survive. Dropped from the original:
- MetricsRecorder (replaced by MavaLogger in `mava/jaxgcrl/run.py`)
- get_env_config (argparse-based, unused in the hydra path)
- render / render_policy helpers (depended on wandb + matplotlib;
  not used by the hydra entry point — re-add later if needed)
"""

from mava.jaxgcrl.envs.ant import Ant
from mava.jaxgcrl.envs.ant_ball import AntBall
from mava.jaxgcrl.envs.ant_ball_maze import AntBallMaze
from mava.jaxgcrl.envs.ant_maze import AntMaze
from mava.jaxgcrl.envs.ant_push import AntPush
from mava.jaxgcrl.envs.half_cheetah import Halfcheetah
from mava.jaxgcrl.envs.humanoid import Humanoid
from mava.jaxgcrl.envs.humanoid_maze import HumanoidMaze
from mava.jaxgcrl.envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from mava.jaxgcrl.envs.manipulation.arm_binpick_hard import ArmBinpickHard
from mava.jaxgcrl.envs.manipulation.arm_grasp import ArmGrasp
from mava.jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy
from mava.jaxgcrl.envs.manipulation.arm_push_hard import ArmPushHard
from mava.jaxgcrl.envs.manipulation.arm_reach import ArmReach
from mava.jaxgcrl.envs.pusher import Pusher, PusherReacher
from mava.jaxgcrl.envs.pusher2 import Pusher2
from mava.jaxgcrl.envs.reacher import Reacher
from mava.jaxgcrl.envs.simple_maze import SimpleMaze

legal_envs = (
    "ant",
    "ant_random_start",
    "ant_ball",
    "ant_push",
    "humanoid",
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "arm_reach",
    "arm_grasp",
    "arm_push_easy",
    "arm_push_hard",
    "arm_binpick_easy",
    "arm_binpick_hard",
    "ant_ball_maze",
    "ant_u_maze",
    "ant_big_maze",
    "ant_hardest_maze",
    "humanoid_u_maze",
    "humanoid_big_maze",
    "humanoid_hardest_maze",
    "simple_u_maze",
    "simple_big_maze",
    "simple_hardest_maze",
)


def create_env(env_name: str, backend: str = None, **kwargs) -> object:
    """Instantiate a JaxGCRL env by name.

    Args:
        env_name: env identifier (must be in `legal_envs`).
        backend: brax backend (`"mjx"`, `"spring"`, `"positional"`, `"generalized"`).
                 None lets each env pick its own default.

    Returns:
        A brax `PipelineEnv` subclass with `state_dim`, `goal_indices`,
        `observation_size`, and `action_size` attributes.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring")
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_push":
        # Stable only on mjx backend.
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "ant" in env_name:
            env = AntMaze(backend=backend or "spring", maze_layout_name=env_name[4:])
        elif "humanoid" in env_name:
            env = HumanoidMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        else:
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env
