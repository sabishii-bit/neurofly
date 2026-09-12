"""What the brain is rewarded for, and when an episode ends.

The raw interface only sees pixels and sound; a ``Task`` turns those into
reward and episode structure. Subclass it in your project and point the
scripts at it with ``--reward mypackage.module:MyTask`` (or
``path/to/file.py:MyTask``). The three hooks:

    reset(controls, video, audio)   start an episode, e.g. press the restart key
    reward(frame, audio, state, info) -> float
    done(frame, audio, info) -> bool
    odours(frame, audio, info) -> {"channel": value} | vector | None   (for --odours)
    tastes(frame, audio, info), thermo(frame, audio, info)               (--tastes, --thermo)
    touch(frame, audio, info)                                             (--touch)
    pulses(frame, audio, info) -> {"population": mV} | None   one-step drives by name

``PatchBrightness`` is a worked example that reads a rectangle of the screen
(a health bar, a score, a target) and rewards its brightness.
"""
from __future__ import annotations

import importlib
import importlib.util
import os

import numpy as np

from neurofly_core.controls import ControlState


class Task:
    def reset(self, controls, video, audio) -> None:
        pass

    def reward(self, frame: np.ndarray, audio: np.ndarray | None, state: ControlState,
               info: dict) -> float:
        return 0.0

    def done(self, frame: np.ndarray, audio: np.ndarray | None, info: dict) -> bool:
        return False

    def odours(self, frame: np.ndarray, audio: np.ndarray | None, info: dict):
        """What the fly should smell this step, for a model built with ``--odours``: a
        dict of channel name to value in [0, 1] (missing channels keep their last value),
        a vector in channel order, or None to change nothing."""
        return None

    def tastes(self, frame: np.ndarray, audio: np.ndarray | None, info: dict):
        """Taste channel values for a model built with ``--tastes``, as ``odours``."""
        return None

    def thermo(self, frame: np.ndarray, audio: np.ndarray | None, info: dict):
        """Temperature and humidity channel values for ``--thermo``, as ``odours``."""
        return None

    def touch(self, frame: np.ndarray, audio: np.ndarray | None, info: dict):
        """Touch channel values for ``--touch`` (bristles, grooming, leg contact), as
        ``odours``."""
        return None

    def pulses(self, frame: np.ndarray, audio: np.ndarray | None, info: dict):
        """Drives for this step only, by population name: ``{"giantfibre": 20}`` startles
        the fly, ``{"clock": 5}`` nudges the circadian neurons; any name the model lists
        (``model.populations()``) works. None for nothing."""
        return None


class NoTask(Task):
    """Reward 0, never ends. For imitation and for just watching."""


class PatchBrightness(Task):
    """Reward = mean brightness of a rectangle of the frame, given as fractions
    (left, top, width, height) of the frame size. ``target`` flips it into a
    distance-to-target reward; ``end_below`` ends the episode when the patch
    goes darker than a threshold (a health bar running out)."""

    def __init__(self, patch=(0.0, 0.0, 1.0, 1.0), target: float | None = None,
                 end_below: float | None = None):
        self.patch, self.target, self.end_below = patch, target, end_below

    def brightness(self, frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        x0, y0, pw, ph = self.patch
        x0, y0 = int(x0 * w), int(y0 * h)
        x1, y1 = max(x0 + 1, int((x0 / w + pw) * w)), max(y0 + 1, int((y0 / h + ph) * h))
        return float(np.asarray(frame[y0:y1, x0:x1], np.float32).mean() / 255.0)

    def reward(self, frame, audio, state, info) -> float:
        b = self.brightness(frame)
        info["brightness"] = b
        return b if self.target is None else 1.0 - abs(b - self.target)

    def done(self, frame, audio, info) -> bool:
        return self.end_below is not None and self.brightness(frame) < self.end_below


def load_task(spec: str | None, **kwargs) -> Task:
    """``None`` -> NoTask. ``'pkg.module:Name'`` or ``'file.py:Name'`` -> that class
    (instantiated with ``kwargs``), instance, or factory."""
    if not spec:
        return NoTask()
    if ":" not in spec:
        raise ValueError(f"task spec {spec!r} must look like module:Name or file.py:Name")
    where, name = spec.rsplit(":", 1)
    if where.endswith(".py"):
        modspec = importlib.util.spec_from_file_location(
            os.path.splitext(os.path.basename(where))[0], where)
        mod = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(where)
    obj = getattr(mod, name)
    if isinstance(obj, Task):
        return obj
    task = obj(**kwargs)
    if not isinstance(task, Task):
        raise TypeError(f"{spec} produced {type(task).__name__}, not a Task")
    return task
