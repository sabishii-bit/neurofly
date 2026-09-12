"""flybody walking action layout: 59 actuators in this fixed order."""
from __future__ import annotations

import numpy as np

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur",
              "tibia", "tarsus", "tarsus2"]
SIDE_NAME = {"L": "left", "R": "right"}

ACTUATOR_NAMES = ["head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen"]
for _t in ("T1", "T2", "T3"):
    for _s in ("left", "right"):
        ACTUATOR_NAMES += [f"{j}_{_t}_{_s}" for j in LEG_JOINTS]
for _t in ("T1", "T2", "T3"):
    for _s in ("left", "right"):
        ACTUATOR_NAMES.append(f"adhere_claw_{_t}_{_s}")
assert len(ACTUATOR_NAMES) == 59

HEAD_ABDOMEN_INDICES = np.arange(5)


def leg_actuator_indices(neuromere: str, side: str) -> np.ndarray:
    """Indices of the 8 joint actuators + 1 adhesion actuator of one leg."""
    s = SIDE_NAME[side]
    names = [f"{j}_{neuromere}_{s}" for j in LEG_JOINTS] + [f"adhere_claw_{neuromere}_{s}"]
    return np.array([ACTUATOR_NAMES.index(n) for n in names])
