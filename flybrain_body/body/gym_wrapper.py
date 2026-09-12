"""Gymnasium wrapper around a dm_control environment.

Flattens the observation dict to one float32 vector (and remembers where each
observable lives, so the sensory encoder can find joints and touch sensors),
rescales actions from [-1, 1] to the real actuator ranges, and maps dm_env's
discount to Gymnasium's terminated / truncated.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DmEnvToGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, dm_env, camera_id: int = 1, width: int = 640, height: int = 480,
                 render_mode: str | None = None):
        self._env = dm_env
        self.render_mode = render_mode
        self._camera_id, self._width, self._height = camera_id, width, height

        aspec = dm_env.action_spec()
        self._act_lo = np.asarray(aspec.minimum, dtype=np.float32)
        self._act_hi = np.asarray(aspec.maximum, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=aspec.shape, dtype=np.float32)

        ospec = dm_env.observation_spec()
        self.obs_keys = list(ospec.keys())
        self.obs_slices: dict[str, slice] = {}
        start = 0
        for k, s in ospec.items():
            size = int(np.prod(s.shape))
            self.obs_slices[k] = slice(start, start + size)
            start += size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(start,), dtype=np.float32)

        walker = dm_env.task._walker
        try:
            self.joint_names = [j.name for j in walker.observable_joints]
        except Exception:  # pragma: no cover
            self.joint_names = None
        self.control_timestep = float(dm_env.control_timestep())

    def flatten_observation(self, obs: dict) -> np.ndarray:
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in self.obs_keys])

    def unscale_action(self, action) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self._act_lo + (a + 1.0) * 0.5 * (self._act_hi - self._act_lo)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ts = self._env.reset()
        return self.flatten_observation(ts.observation), {}

    def step(self, action):
        ts = self._env.step(self.unscale_action(action))
        obs = self.flatten_observation(ts.observation)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = bool(ts.last() and ts.discount == 0.0)
        truncated = bool(ts.last() and not terminated)
        return obs, reward, terminated, truncated, {}

    def render(self):
        return self._env.physics.render(camera_id=self._camera_id,
                                        width=self._width, height=self._height)

    def close(self):
        self._env.close()

    @property
    def dm_env(self):
        return self._env

    @property
    def physics(self):
        return self._env.physics

    def root_position(self) -> np.ndarray:
        walker = self._env.task._walker
        return np.array(self._env.physics.bind(walker.root_body).xpos)
