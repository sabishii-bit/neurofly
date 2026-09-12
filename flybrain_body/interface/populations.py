"""Named neuron populations that the body interface talks to.

Uses the MaleCNS annotation columns:
  * leg motor neurons:   superclass vnc_motor, somaNeuromere T1/T2/T3, somaSide L/R
  * leg sensory neurons: superclass vnc_sensory, entryNerve ProLN/MesoLN/MetaLN
                         (the T1/T2/T3 leg nerves), rootSide L/R; split by
                         class into proprioceptive vs tactile
  * descending neurons:  superclass descending_neuron (brain -> nerve cord)
  * head mechanosensory: cb_sensory with subclass wind_gravity (Johnston's
                         organ) and haltere afferents
  * dopamine neurons:    class DAN
"""
from __future__ import annotations

import numpy as np

from flybrain_body.data.connectome import LEG_NERVE, LEGS, Connectome


class Populations:
    def __init__(self, cx: Connectome):
        self.cx = cx
        self.leg_motor: dict[tuple[str, str], np.ndarray] = {}
        self.leg_sensory: dict[tuple[str, str], np.ndarray] = {}
        self.leg_proprio: dict[tuple[str, str], np.ndarray] = {}
        self.leg_tactile: dict[tuple[str, str], np.ndarray] = {}
        for t, side in LEGS:
            self.leg_motor[(t, side)] = cx.select(superclass="vnc_motor",
                                                  somaNeuromere=t, somaSide=side)
            sens = cx.select(superclass="vnc_sensory", entryNerve=LEG_NERVE[t], rootSide=side)
            self.leg_sensory[(t, side)] = sens
            cls = cx.neurons["class"].values[sens]
            self.leg_proprio[(t, side)] = sens[cls == "mechanosensory_proprioceptive"]
            self.leg_tactile[(t, side)] = sens[cls == "mechanosensory_tactile"]
        self.descending = cx.select(superclass="descending_neuron")
        self.ascending = cx.select(superclass="ascending_neuron")
        self.wind_gravity = cx.select(superclass="cb_sensory", subclass="wind_gravity")
        self.haltere = cx.select(subclass="haltere")
        self.dopamine = cx.select(class_="DAN")
        self.all_leg_motor = np.unique(np.concatenate(list(self.leg_motor.values())))

    def readout(self, which: str = "motor+descending") -> np.ndarray:
        parts = []
        if "motor" in which:
            parts.append(self.all_leg_motor)
        if "descending" in which:
            parts.append(self.descending)
        if "ascending" in which:
            parts.append(self.ascending)
        if not parts:
            raise ValueError(f"unknown readout {which!r}")
        return np.unique(np.concatenate(parts))

    def summary(self) -> str:
        lines = []
        for t, side in LEGS:
            lines.append(f"  leg {t}{side}: motor {len(self.leg_motor[(t, side)]):4d}  "
                         f"proprio {len(self.leg_proprio[(t, side)]):4d}  "
                         f"tactile {len(self.leg_tactile[(t, side)]):4d}")
        lines.append(f"  descending {len(self.descending)}, ascending {len(self.ascending)}, "
                     f"wind/gravity {len(self.wind_gravity)}, haltere {len(self.haltere)}, "
                     f"dopamine {len(self.dopamine)}")
        return "\n".join(lines)
