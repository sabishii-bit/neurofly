"""The PC with the fly brain in the loop, as a Gymnasium environment.

``PCEnv`` wraps a ``neurofly_core.Model``: each step the agent's action vector
is unpacked into keys, mouse buttons, mouse motion and scroll and applied to
the PC; the environment waits for the next tick (live sources only); the new
frame and the sound since the last step go through the model's encoders and
brain; the agent sees the model's features. Reward and episode ends come from
a ``Task``. After training, the policy goes into the same model and the
artifact runs without this class.
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neurofly_core.io.audio import AudioSource
from neurofly_core.io.controls import Controls, NullControls
from neurofly_core.io.video import VideoSource
from neurofly_core.model import Model
from neurofly_training.pc.task import NoTask, Task


class Clock:
    """Holds a loop at ``fps`` steps per second."""

    def __init__(self, fps: float):
        self.period = 1.0 / float(fps)
        self._next = None

    def reset(self) -> None:
        self._next = time.perf_counter() + self.period

    def wait(self) -> None:
        if self._next is None:
            self.reset()
        lag = self._next - time.perf_counter()
        if lag > 0:
            time.sleep(lag)
        self._next = max(self._next + self.period, time.perf_counter())


class PCEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, model: Model, *, video: VideoSource, controls: Controls | None = None,
                 audio: AudioSource | None = None, task: Task | None = None, fps: float = 10.0,
                 realtime: bool | None = None, max_steps: int | None = None):
        self.model = model
        self.video, self.audio = video, audio
        self.controls = controls if controls is not None else NullControls()
        self.task = task if task is not None else NoTask()
        self.fps = float(fps)
        if realtime is None:
            realtime = video.live or (audio is not None and audio.live)
        self.realtime = bool(realtime)
        self.clock = Clock(self.fps) if self.realtime else None
        self.max_steps = max_steps
        if audio is not None and model.audition is None:
            raise ValueError("the model has no audition encoder; build it with audio=True")
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(model.n_features,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(model.n_actions,), dtype=np.float32)
        self.render_mode = "rgb_array"
        self.metadata = dict(self.metadata, render_fps=max(1, int(round(self.fps))))
        self._frame = None
        self._chunk = None
        self.t = 0

    # --- convenience --------------------------------------------------------------

    @property
    def layout(self):
        return self.model.layout

    @property
    def brain(self):
        return self.model.brain

    @property
    def readout_idx(self):
        return self.model.readout_idx

    # --- gym API -------------------------------------------------------------------

    def _read(self) -> bool:
        """Next frame and audio chunk; False when the video is over."""
        frame = self.video.read()
        if self.audio is not None:
            self._chunk = self.audio.read()
        if frame is None:
            return False
        self._frame = frame
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.controls.release_all()
        self.task.reset(self.controls, self.video, self.audio)
        frame = self.video.reset()
        if frame is None:
            raise RuntimeError("the video source produced no frame")
        self._frame = frame
        self._chunk = self.audio.reset() if self.audio is not None else None
        self.t = 0
        self.model.reset()
        if self.clock is not None:
            self.clock.reset()
        return self.model.observe(self._frame, self._chunk), {}

    def step(self, action):
        state = self.model.layout.decode(action)
        self.controls.apply(state)
        if self.clock is not None:
            self.clock.wait()
        more = self._read()
        self.t += 1
        info = {"t": self.t, "held": state.held, "state": state}
        reward = float(self.task.reward(self._frame, self._chunk, state, info))
        terminated = bool(self.task.done(self._frame, self._chunk, info))
        truncated = (not more) or (self.max_steps is not None and self.t >= self.max_steps)
        obs = self.model.observe(self._frame, self._chunk, reward=reward)
        info["brain_spikes"] = self.model.last_spikes
        if terminated or truncated:
            self.controls.release_all()
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._frame

    def close(self):
        self.controls.close()
        self.video.close()
        if self.audio is not None:
            self.audio.close()
