"""The fly body with the brain in the loop, as a Gymnasium environment.

Each control step (2 ms), the body observation is encoded into drive on the
nerve cord's sensory neurons, the brain runs for one control interval, and the
agent sees the firing rates of a readout population (motor and/or descending
neurons). The agent's action still goes straight to the 59 actuators.
Optionally reward acts as dopamine on the synapses onto the leg motor neurons.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neurofly_core.brain.lif import LIFBrain
from neurofly_core.brain.plasticity import DopamineHebbian
from neurofly_training.body.encoder import ProprioEncoder
from neurofly_training.body.gym_wrapper import DmEnvToGym
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations

ENCODER_SEED = 0   # the sensory map must be identical across parallel workers


class BrainInLoopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, body: DmEnvToGym, cx: Connectome, *,
                 readout: str | np.ndarray = "motor+descending",
                 include_proprio: bool = False, dt: float = 0.5, brain_gain: float = 1.0,
                 encoder_gain: float = 12.0, plasticity: bool = False, device: str = "cpu",
                 warmup_ms: float = 20.0, seed: int = 0):
        self.body = body
        self.cx = cx
        self.pops = Populations(cx)
        self.brain = LIFBrain(cx.W, dt=dt, gain=brain_gain, device=device)
        self.encoder = ProprioEncoder(self.pops, cx.n, body.obs_slices, body.joint_names,
                                      gain=encoder_gain, seed=ENCODER_SEED, device=device)
        self.readout_idx = (self.pops.readout(readout) if isinstance(readout, str)
                            else np.asarray(readout))
        self.include_proprio = include_proprio
        self.substeps = max(1, int(round(body.control_timestep * 1000.0 / dt)))
        self.warmup_steps = int(round(warmup_ms / dt))
        self.plasticity = None
        if plasticity:
            self.plasticity = DopamineHebbian(self.brain, pre_idx=np.arange(cx.n),
                                              post_idx=self.pops.all_leg_motor)
        n_feat = len(self.readout_idx) + (body.observation_space.shape[0] if include_proprio else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_feat,), dtype=np.float32)
        self.action_space = body.action_space
        self.render_mode = body.render_mode
        self._last_body_obs = None

    def _features(self) -> np.ndarray:
        feats = self.brain.rates(self.readout_idx) / 100.0
        if self.include_proprio:
            feats = np.concatenate([feats, self._last_body_obs])
        return feats.astype(np.float32)

    def _run_brain(self, body_obs: np.ndarray, n_steps: int, dopamine: float = 0.0) -> int:
        drive = self.encoder(body_obs)
        before = self.brain.total_spikes
        for _ in range(n_steps):
            self.brain.step(drive)
            if self.plasticity is not None:
                self.plasticity.step(dopamine)
        return self.brain.total_spikes - before

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        body_obs, info = self.body.reset(seed=seed)
        self._last_body_obs = body_obs
        self.brain.reset()
        self._run_brain(body_obs, self.warmup_steps)
        return self._features(), info

    def step(self, action):
        body_obs, reward, terminated, truncated, info = self.body.step(action)
        self._last_body_obs = body_obs
        spikes = self._run_brain(body_obs, self.substeps,
                                 dopamine=reward if self.plasticity is not None else 0.0)
        info["brain_spikes"] = spikes
        return self._features(), reward, terminated, truncated, info

    def render(self):
        return self.body.render()

    def close(self):
        self.body.close()

    def root_position(self):
        return self.body.root_position()
