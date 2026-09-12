"""An open-loop tripod gait: a policy that ignores the observation and swings the legs.

Insects walk with two alternating tripods (front-left, middle-right, hind-left against
the other three). This produces that pattern as sinusoids on each leg's coxa (the
stride), femur and tibia (the lift) and, optionally, claw adhesion during the stance,
so the limbs can be seen working in a viewer or a video without any training at all.
It is open loop: the fly stays upright and cycles its legs but does not go anywhere
much; a policy that walks comes from training.

    neurofly watch --task forward --policy gait --poses assets/gait.json --video videos/gait.mp4
"""
from __future__ import annotations

import numpy as np

from neurofly_training.body.actuators import ACTUATOR_NAMES, LEG_JOINTS, leg_actuator_indices

TRIPOD_A = [("T1", "L"), ("T2", "R"), ("T3", "L")]
TRIPOD_B = [("T1", "R"), ("T2", "L"), ("T3", "R")]


class TripodGait:
    def __init__(self, control_hz: float = 500.0, stride_hz: float = 2.0, stride: float = 0.3,
                 lift: float = 0.3, adhesion: float = 0.0):
        """``control_hz``: control steps per second (flybody: 500); ``stride_hz``: full
        gait cycles per second; ``stride`` and ``lift``: amplitudes in [0, 1] of the
        action range; ``adhesion``: how much the claws grip during the stance, 0 for
        none (grip switching launches the body if it is abrupt, so it ramps)."""
        self.control_hz, self.stride_hz = float(control_hz), float(stride_hz)
        self.stride, self.lift, self.adhesion = float(stride), float(lift), float(adhesion)
        self.n = len(ACTUATOR_NAMES)
        j = {name: k for k, name in enumerate(LEG_JOINTS)}
        self.legs = []
        for phase, tripod in ((0.0, TRIPOD_A), (np.pi, TRIPOD_B)):
            for t, side in tripod:
                idx = leg_actuator_indices(t, side)
                sign = -1.0 if side == "L" else 1.0        # mirror the stride across the body
                self.legs.append((phase, idx[j["coxa"]], idx[j["femur"]], idx[j["tibia"]],
                                  idx[-1], sign))

    def __call__(self, step: int) -> np.ndarray:
        """The action for control step ``step`` (an integer since the episode start)."""
        theta = 2.0 * np.pi * self.stride_hz * step / self.control_hz
        a = np.zeros(self.n, np.float32)
        for phase, coxa, femur, tibia, claw, sign in self.legs:
            ph = theta + phase
            up = max(0.0, np.sin(ph))                      # half the cycle in the air
            a[coxa] = -sign * self.stride * np.cos(ph)     # back during stance, forward in swing
            a[femur] = self.lift * up                      # lift only while swinging
            a[tibia] = -self.lift * up
            a[claw] = -1.0 + 2.0 * self.adhesion * (1.0 - up)   # grip ramps up during the stance
        return a

    def actions(self, n_steps: int) -> np.ndarray:
        return np.stack([self(k) for k in range(n_steps)])
