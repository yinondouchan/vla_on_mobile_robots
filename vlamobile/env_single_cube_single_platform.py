import gymnasium as gym
import vlamobile.lift_env  # noqa: F401 — registers LiftMobileRobot-v0


def make_env(n_envs: int = 1, use_async_envs: bool = False, cfg=None, **env_kwargs):
    """
    Create vectorized environments for your custom task.

    Args:
        n_envs: Number of parallel environments
        use_async_envs: Whether to use AsyncVectorEnv or SyncVectorEnv

    Returns:
        gym.vector.VectorEnv or dict mapping suite names to vectorized envs
    """
    env_kwargs = {
        "hide_red_platform": True,
        "hide_blue_platform": True,
        "hide_medium_cube": True,
        "hide_small_cube": True
    }

    def _make_single_env():
        # Create your custom environment
        return gym.make("LiftMobileRobot-v0", **env_kwargs)

    # Choose vector environment type
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    # Create vectorized environment
    vec_env = env_cls([_make_single_env for _ in range(n_envs)])

    return vec_env
