"""Gymnasium-compliant adapter around safe-control-gym's cartpole environment."""
import gymnasium as gym
import yaml
from safe_control_gym.utils.registration import make


class SafeCartpoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path="configs/cartpole_stab.yaml"):
        super().__init__()
        with open(config_path, "r") as f:
            task_config = yaml.safe_load(f)["task_config"]
        self._env = make("cartpole", **task_config)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, bool(done), False, info  # terminated, truncated

    def close(self):
        self._env.close()