"""flybody walking action layout: 59 actuators in this fixed order.

The order of the *action vector* is not the order of the actuators in the MuJoCo model.
flybody's walker groups its actuators and lays the action out group by group: the six
adhesion claws first, then the head, the abdomen, then the 48 leg joints (T1 to T3, left
before right, coxa to tarsus2). The model lists the head first and the claws last. Every
index in this module is an action index; ``MODEL_ACTUATOR_NAMES`` is the model's order
and ``ACTION_TO_MODEL`` maps between them.
"""
from __future__ import annotations

import numpy as np

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur",
              "tibia", "tarsus", "tarsus2"]
SIDE_NAME = {"L": "left", "R": "right"}
NEUROMERES = ("T1", "T2", "T3")

CLAW_NAMES = [f"adhere_claw_{t}_{s}" for t in NEUROMERES for s in ("left", "right")]
HEAD_NAMES = ["head_abduct", "head_twist", "head"]
ABDOMEN_NAMES = ["abdomen_abduct", "abdomen"]
LEG_NAMES = [f"{j}_{t}_{s}" for t in NEUROMERES for s in ("left", "right") for j in LEG_JOINTS]

ACTUATOR_NAMES = CLAW_NAMES + HEAD_NAMES + ABDOMEN_NAMES + LEG_NAMES      # action order
MODEL_ACTUATOR_NAMES = HEAD_NAMES + ABDOMEN_NAMES + LEG_NAMES + CLAW_NAMES  # MuJoCo order
assert len(ACTUATOR_NAMES) == len(MODEL_ACTUATOR_NAMES) == 59

ACTION_TO_MODEL = np.array([MODEL_ACTUATOR_NAMES.index(n) for n in ACTUATOR_NAMES])
MODEL_TO_ACTION = np.argsort(ACTION_TO_MODEL)

ADHESION_INDICES = np.arange(0, 6)
HEAD_ABDOMEN_INDICES = np.arange(6, 11)
LEG_INDICES = np.arange(11, 59)


def leg_actuator_indices(neuromere: str, side: str) -> np.ndarray:
    """Action indices of the 8 joint actuators + 1 adhesion actuator of one leg."""
    s = SIDE_NAME[side]
    names = [f"{j}_{neuromere}_{s}" for j in LEG_JOINTS] + [f"adhere_claw_{neuromere}_{s}"]
    return np.array([ACTUATOR_NAMES.index(n) for n in names])


def action_order(walker) -> list[str]:
    """The action layout a flybody walker actually uses, actuator name per action index,
    read from the walker itself (what ``ACTUATOR_NAMES`` must equal)."""
    names = [None] * sum(walker._num_actions.values())
    for key, action_idx in walker._action_indices.items():
        ctrl_idx = walker._ctrl_indices.get(key)
        if not action_idx or ctrl_idx is None:
            continue
        for ai, ci in zip(action_idx, ctrl_idx):
            names[ai] = walker.actuators[ci].name
    return names
