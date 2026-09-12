"""A Task that reads a bar on the screen: reward for keeping it full, restart when it empties.

Adjust the three constants to your screen: where the bar is (fractions of the frame),
which colour counts as "filled", and which key restarts. Everything else is the Task
interface described in docs/pc.md.

    neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d \
        --reward examples/health_bar_task.py:HealthBar --max-steps 600
"""
from __future__ import annotations

import time

import numpy as np

from neurofly_core.controls import ControlState
from neurofly_training.pc.task import Task

BAR = (0.05, 0.92, 0.30, 0.03)       # left, top, width, height as fractions of the frame
FILLED = np.array([200, 40, 40])     # RGB of the bar when it is filled
RESTART_KEY = "r"


class HealthBar(Task):
    def __init__(self, bar=BAR, filled=FILLED, restart_key=RESTART_KEY, tolerance=80.0):
        self.bar, self.filled = bar, np.asarray(filled)
        self.restart_key, self.tolerance = restart_key, tolerance

    def fill(self, frame: np.ndarray) -> float:
        """Fraction of the bar's width whose colour matches ``filled``."""
        h, w = frame.shape[:2]
        x0, y0, bw, bh = self.bar
        x0, x1 = int(x0 * w), max(int(x0 * w) + 1, int((x0 + bw) * w))
        y0, y1 = int(y0 * h), max(int(y0 * h) + 1, int((y0 + bh) * h))
        strip = frame[y0:y1, x0:x1].astype(np.float32)
        match = np.linalg.norm(strip - self.filled, axis=-1) < self.tolerance
        return float(match.any(axis=0).mean())

    def reset(self, controls, video, audio) -> None:
        """Tap the restart key and give the program a moment to redraw."""
        controls.apply(ControlState(keys=frozenset({self.restart_key})))
        time.sleep(0.05)
        controls.release_all()
        time.sleep(0.5)

    def reward(self, frame, audio, state, info) -> float:
        fill = self.fill(frame)
        info["fill"] = fill
        return fill - 0.5            # positive while more than half full, negative below

    def done(self, frame, audio, info) -> bool:
        return info.get("fill", self.fill(frame)) <= 0.0
