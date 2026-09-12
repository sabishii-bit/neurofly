"""An open-loop tripod gait: a policy that ignores the observation and swings the legs.

Insects walk with two alternating tripods (front-left, middle-right, hind-left against
the other three). This produces that pattern as sinusoids on each leg's coxa (the
stride), femur and tibia (the lift) and, optionally, claw adhesion during the stance,
so the limbs can be seen working in a viewer or a video without any training at all.
It is open loop: the fly stays upright and cycles its legs but does not go anywhere
much; a policy that walks comes from training.

    neurofly watch --task forward --policy gait --poses assets/gait.json --video videos/gait.mp4

Through the physics, pass ``rest=rest_action(physics.model)``: the environment maps the
action range onto each actuator's control range, so action 0 is mid-range, and for the
fly's asymmetric joint ranges mid-range is a crouch with the hind legs tucked under the
body, not the standing pose. ``rest_action`` is the action that holds every joint at its
standing angle; the gait's sinusoids are added to it.
"""
from __future__ import annotations

import numpy as np

from neurofly_training.body.actuators import (ACTION_TO_MODEL, ACTUATOR_NAMES, LEG_JOINTS,
                                              leg_actuator_indices)

TRIPOD_A = [("T1", "L"), ("T2", "R"), ("T3", "L")]
TRIPOD_B = [("T1", "R"), ("T2", "L"), ("T3", "R")]
JOINT_ACTUATOR = 0        # MuJoCo transmission type: acts on one joint


def rest_action(model) -> np.ndarray:
    """The action (in [-1, 1] per actuator) that holds every joint actuator at joint
    angle 0, the standing pose, with the claws not gripping; tendon actuators stay at 0."""
    a = np.zeros(model.nu, np.float32)
    a[[i for i, n in enumerate(ACTUATOR_NAMES) if n.startswith("adhere_claw")]] = -1.0
    rng = np.asarray(model.actuator_ctrlrange)[ACTION_TO_MODEL]      # in action order
    lo, hi = rng[:, 0], rng[:, 1]
    pos = np.asarray(model.actuator_trntype)[ACTION_TO_MODEL] == JOINT_ACTUATOR
    a[pos] = -(lo[pos] + hi[pos]) / (hi[pos] - lo[pos])
    return np.clip(a, -1.0, 1.0)


class TripodGait:
    def __init__(self, control_hz: float = 500.0, stride_hz: float = 2.0, stride: float = 0.3,
                 lift: float = 0.3, adhesion: float = 0.0, rest=None):
        """``control_hz``: control steps per second (flybody: 500); ``stride_hz``: full
        gait cycles per second; ``stride`` and ``lift``: amplitudes in [0, 1] of the
        action range; ``adhesion``: how much the claws grip during the stance, 0 for
        none (grip switching launches the body if it is abrupt, so it ramps);
        ``rest``: the action the sinusoids are added to (``rest_action`` for the
        physics; zeros, the default, when the actions are read as joint angles)."""
        self.control_hz, self.stride_hz = float(control_hz), float(stride_hz)
        self.stride, self.lift, self.adhesion = float(stride), float(lift), float(adhesion)
        self.n = len(ACTUATOR_NAMES)
        self.rest = (np.zeros(self.n, np.float32) if rest is None
                     else np.asarray(rest, np.float32).copy())
        assert self.rest.shape == (self.n,)
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
        a = self.rest.copy()
        for phase, coxa, femur, tibia, claw, sign in self.legs:
            ph = theta + phase
            up = max(0.0, np.sin(ph))                      # half the cycle in the air
            a[coxa] += -sign * self.stride * np.cos(ph)    # back during stance, forward in swing
            a[femur] += self.lift * up                     # lift only while swinging
            a[tibia] += -self.lift * up
            a[claw] = -1.0 + 2.0 * self.adhesion * (1.0 - up)   # grip ramps up during the stance
        return np.clip(a, -1.0, 1.0)

    def actions(self, n_steps: int) -> np.ndarray:
        return np.stack([self(k) for k in range(n_steps)])
