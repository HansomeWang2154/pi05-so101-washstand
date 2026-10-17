"""Optional simulation package; heavy runtime imports stay behind env access."""

from .scene import JOINT_NAMES, TASKS, load_config

__all__ = ["WashstandEnv", "JOINT_NAMES", "TASKS", "load_config"]


def __getattr__(name):
    if name == "WashstandEnv":
        from .env import WashstandEnv
        return WashstandEnv
    raise AttributeError(name)
