"""Named neuron populations that the body and game interfaces talk to.

Uses the MaleCNS annotation columns:
  * leg motor neurons:   superclass vnc_motor, somaNeuromere T1/T2/T3, somaSide L/R
  * leg sensory neurons: superclass vnc_sensory, entryNerve ProLN/MesoLN/MetaLN
                         (the T1/T2/T3 leg nerves), rootSide L/R; split by
                         class into proprioceptive vs tactile
  * descending neurons:  superclass descending_neuron (brain -> nerve cord)
  * head mechanosensory: cb_sensory with subclass wind_gravity (Johnston's
                         organ) and haltere afferents
  * dopamine neurons:    class DAN; the PPL1 cluster is the punishment input
  * auditory neurons:    cb_sensory with subclass auditory (Johnston's organ)
  * visual input:        columnar optic-lobe neurons (L1, L2, Mi1, Tm1, ...) carry
                         their hex column (assignedOlHex1/2) and eye (somaSide);
                         that is the retinotopic map the frame encoder uses
  * visual projection:   superclass visual_projection (LC, LPLC, MeTu, ... cells
                         leaving the optic lobe), fallback visual input
"""
from __future__ import annotations

import numpy as np

from neurofly_training.data.connectome import HEX_COLS, LEG_NERVE, LEGS, Connectome

RETINA_TYPES = ("L1", "L2", "L3", "Mi1", "Tm1", "Tm2")  # default columnar input cells
ON_TYPES = frozenset({"L1", "L3", "Mi1", "Tm3", "Mi4"})  # the rest are treated as OFF cells


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
        self.auditory = cx.select(superclass="cb_sensory", subclass="auditory")
        self.haltere = cx.select(subclass="haltere")
        self.olfactory = cx.select(class_="olfactory")             # receptor neurons, by glomerulus
        self.glomeruli = self._by_type(self.olfactory)
        self.gustatory = cx.select(class_re="gustat")
        self.taste_types = self._by_type(self.gustatory)
        self.thermo = cx.select(type_re=r"^(?:TRN|HRN)_")            # temperature and humidity
        self.thermo_types = self._by_type(self.thermo)
        # touch: the head's bristle and grooming mechanosensory neurons by subclass, and
        # each leg's tactile neurons
        self.touch_types: dict[str, np.ndarray] = {}
        head_touch = cx.select(superclass="cb_sensory", class_="mechanosensory",
                               subclass_re="bristle|grooming|pharyngeal|taste peg")
        subs = cx.neurons["subclass"].values[head_touch]
        for sub in sorted({str(s) for s in subs if s}):
            self.touch_types[f"head:{sub}"] = head_touch[subs == sub]
        for (t, side), idx in self.leg_tactile.items():
            if len(idx):
                self.touch_types[f"leg:{t}{side}"] = idx
        self.giantfibre = cx.select(type_re=r"^GF")                  # the escape circuit
        self.clock = cx.select(type_re=r"^(?:l-LNv|s-LNv|LNd|DN1|DN2|DN3|LPN)")   # circadian
        self.visual_feedback = cx.select(superclass="visual_centrifugal")
        self.dopamine = cx.select(class_="DAN")
        self.ppl1 = cx.select(class_="DAN", type_re=r"^PPL1")      # punishment
        self.pam = cx.select(class_="DAN", type_re=r"^PAM")        # reward
        self.kenyon = cx.select(type_re=r"^KC")                    # the mushroom body ...
        self.mbon = cx.select(type_re=r"^MBON")                    # ... and its outputs
        self.compass = cx.select(type_re=r"^(?:EPG|PEN|PEG|Delta7|EL)\b")   # the heading circuit
        self.all_leg_motor = np.unique(np.concatenate(list(self.leg_motor.values())))
        # game side
        self.cb_motor = cx.select(superclass="cb_motor")
        self.cb_intrinsic = cx.select(superclass="cb_intrinsic")
        self.visual_projection = cx.select(superclass="visual_projection")
        self.photoreceptors = cx.select(superclass="ol_sensory")
        self._readouts = {
            "motor": self.all_leg_motor, "descending": self.descending,
            "ascending": self.ascending, "cbmotor": self.cb_motor,
            "visual": self.visual_projection, "mbon": self.mbon, "compass": self.compass,
            "clock": self.clock,
        }

    def _by_type(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Neurons of ``idx`` grouped by their type annotation, in name order."""
        types = self.cx.neurons["type"].values[idx]
        names = sorted({str(t) for t in types if t is not None and str(t)})
        return {t: idx[types == t] for t in names}

    def readout(self, which: str = "motor+descending") -> np.ndarray:
        """Neurons whose rates the agent sees; ``+``-joined names from
        motor, descending, ascending, cbmotor, visual."""
        parts = []
        for token in which.split("+"):
            if token not in self._readouts:
                raise ValueError(f"unknown readout {token!r}; "
                                 f"choose from {sorted(self._readouts)}")
            parts.append(self._readouts[token])
        return np.unique(np.concatenate(parts))

    def retina(self, types=RETINA_TYPES) -> dict[str, dict]:
        """Columnar visual neurons with a known eye column, per eye.

        Returns ``{"L": d, "R": d}`` with ``d["idx"]`` neuron indices,
        ``d["hex"]`` an (n, 2) float array of hex coordinates, and
        ``d["on"]`` a bool array (True for ON-pathway cell types).
        """
        if not all(c in self.cx.neurons.columns for c in HEX_COLS):
            return {"L": self._empty_retina(), "R": self._empty_retina()}
        has_hex = self.cx.neurons[HEX_COLS].notna().all(axis=1).values
        out = {}
        for side in ("L", "R"):
            idx = self.cx.select(somaSide=side, type=list(types))
            idx = idx[has_hex[idx]]
            out[side] = {
                "idx": idx,
                "hex": self.cx.neurons[HEX_COLS].values[idx].astype(np.float64),
                "on": np.array([t in ON_TYPES for t in self.cx.neurons["type"].values[idx]],
                               dtype=bool),
            }
        return out

    @staticmethod
    def _empty_retina() -> dict:
        return {"idx": np.zeros(0, np.int64), "hex": np.zeros((0, 2)), "on": np.zeros(0, bool)}

    @property
    def n_retina(self) -> int:
        return int(sum(len(d["idx"]) for d in self.retina().values()))

    def summary(self) -> str:
        lines = []
        for t, side in LEGS:
            lines.append(f"  leg {t}{side}: motor {len(self.leg_motor[(t, side)]):4d}  "
                         f"proprio {len(self.leg_proprio[(t, side)]):4d}  "
                         f"tactile {len(self.leg_tactile[(t, side)]):4d}")
        lines.append(f"  descending {len(self.descending)}, ascending {len(self.ascending)}, "
                     f"wind/gravity {len(self.wind_gravity)}, haltere {len(self.haltere)}, "
                     f"auditory {len(self.auditory)}, olfactory {len(self.olfactory)} in "
                     f"{len(self.glomeruli)} glomeruli, gustatory {len(self.gustatory)}, "
                     f"dopamine {len(self.dopamine)} (PPL1 {len(self.ppl1)}, PAM {len(self.pam)})")
        lines.append(f"  mushroom body: {len(self.kenyon)} Kenyon cells, {len(self.mbon)} output "
                     f"neurons; compass {len(self.compass)}; taste {len(self.gustatory)} in "
                     f"{len(self.taste_types)} types; thermo/hygro {len(self.thermo)}; touch "
                     f"{sum(len(v) for v in self.touch_types.values())} in "
                     f"{len(self.touch_types)} groups; giant fibre {len(self.giantfibre)}; clock "
                     f"{len(self.clock)}; visual feedback {len(self.visual_feedback)}")
        ret = self.retina()
        lines.append(f"  retina columns: left {len(ret['L']['idx'])}, "
                     f"right {len(ret['R']['idx'])}; "
                     f"visual projection {len(self.visual_projection)}, "
                     f"photoreceptors {len(self.photoreceptors)}, head motor {len(self.cb_motor)}")
        return "\n".join(lines)
