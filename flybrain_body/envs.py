"""Environment factories, including the brain-in-the-loop environment.

``BrainInLoopEnv`` puts the spiking connectome between the body's sensors and
the agent: each control step, the body observation is encoded into drive on
sensory neurons, the brain runs for one control interval, and the agent sees
the firing rates of a readout population (motor and/or descending neurons).
The agent's action still goes straight to the 59 body actuators. Learning a
policy on top of that readout is what "training the fly brain" means here;
optionally, reward also acts as dopamine on a plastic synapse subset.
"""
from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flybrain_body.body.gym_wrapper import DmEnvToGym
from flybrain_body.body.tasks import make_env as make_dm_env
from flybrain_body.brain.lif import LIFBrain
from flybrain_body.brain.plasticity import DopamineHebbian
from flybrain_body.data.connectome import Connectome
from flybrain_body.interface.encoder import ProprioEncoder
from flybrain_body.interface.populations import Populations

DEFAULT_DATA_DIR = os.environ.get(
    "FLYBRAIN_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "malecns"))

# Which env-construction arguments train scripts must store so watch.py can
# rebuild the same environment.
ENV_ARGS = ["task", "brain", "subset", "dt", "readout", "include_proprio", "plasticity",
            "brain_gain", "encoder_gain", "synthetic_n"]


def make_body_env(task: str = "forward", seed: int = 0, render_mode: str | None = None,
                  camera_id: int = 1, width: int = 640, height: int = 480,
                  **task_kwargs) -> DmEnvToGym:
    return DmEnvToGym(make_dm_env(task, seed=seed, **task_kwargs), camera_id=camera_id,
                      width=width, height=height, render_mode=render_mode)


def load_connectome(brain: str = "malecns", subset: str = "vnc",
                    data_dir: str | None = None, synthetic_n: int = 3000) -> Connectome:
    """The connectome must not depend on the env seed: parallel workers and
    the playback script all need identical neuron populations."""
    if brain == "synthetic":
        return Connectome.synthetic(n=synthetic_n, seed=0)
    if brain == "malecns":
        cx = Connectome.load(data_dir or DEFAULT_DATA_DIR, verbose=False)
        return cx.subset(subset)
    raise ValueError(f"brain must be 'none', 'malecns' or 'synthetic', got {brain!r}")


class BrainInLoopEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, body: DmEnvToGym, cx: Connectome, *, readout: str | np.ndarray = "motor+descending",
                 include_proprio: bool = False, dt: float = 0.5, brain_gain: float = 1.0,
                 encoder_gain: float = 12.0, plasticity: bool = False, device: str = "cpu",
                 warmup_ms: float = 20.0, seed: int = 0):
        self.body = body
        self.cx = cx
        self.pops = Populations(cx)
        self.brain = LIFBrain(cx.W, dt=dt, gain=brain_gain, device=device)
        self.encoder = ProprioEncoder(self.pops, cx.n, body.obs_slices, body.joint_names,
                                      gain=encoder_gain, seed=seed, device=device)
        self.readout_idx = (self.pops.readout(readout) if isinstance(readout, str)
                            else np.asarray(readout))
        self.include_proprio = include_proprio
        self.substeps = max(1, int(round(body.control_timestep * 1000.0 / dt)))
        self.warmup_steps = int(round(warmup_ms / dt))
        self.plasticity = None
        if plasticity:
            # plastic synapses: descending -> leg motor and intrinsic -> motor
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


def make_env(task: str = "forward", seed: int = 0, brain: str = "none", subset: str = "vnc",
             dt: float = 0.5, readout: str = "motor+descending", include_proprio: bool = False,
             plasticity: bool = False, brain_gain: float = 1.0, encoder_gain: float = 12.0,
             synthetic_n: int = 3000, data_dir: str | None = None, device: str = "cpu",
             render_mode: str | None = None, camera_id: int = 1, width: int = 640,
             height: int = 480, **task_kwargs) -> gym.Env:
    """Build a body-only env (brain='none') or a brain-in-the-loop env."""
    body = make_body_env(task, seed=seed, render_mode=render_mode, camera_id=camera_id,
                         width=width, height=height, **task_kwargs)
    if brain == "none":
        return body
    cx = load_connectome(brain, subset=subset, data_dir=data_dir, synthetic_n=synthetic_n)
    return BrainInLoopEnv(body, cx, readout=readout, include_proprio=include_proprio, dt=dt,
                          brain_gain=brain_gain, encoder_gain=encoder_gain,
                          plasticity=plasticity, device=device, seed=seed)
