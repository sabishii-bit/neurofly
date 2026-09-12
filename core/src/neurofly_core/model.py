"""The assembled controllers: brain + encoders + readout + policy.

Two kinds share one base:

* ``Model`` (kind ``pc``): frames and sound in through the retina and the
  auditory neurons, a ``ControlLayout`` of keys, mouse and gamepad out.
* ``BodyModel`` (kind ``body``, in ``neurofly_core.body``): a body observation
  vector in through the proprioceptive map, actuator commands out. The
  simulation stays wherever MuJoCo runs; only the controller is here.

    features = model.observe(input, reward=0)   # drive the neurons, run the brain, read out
    action   = model.act(features)              # the policy, if the artifact has one
    out, info = model.step(input)               # both

Training environments call ``observe`` and supply their own actions; a deployed
controller calls ``step``. ``reset`` starts a fresh episode (the next ``observe``
runs the warm-up).

Experiments on the brain while it runs, on either kind:

    model.stimulate({"type_re": "^PPL1"}, 20.0)   # extra drive, mV, every step
    model.silence({"superclass": "descending_neuron"})
    model.probe({"name": "readout"})              # model.last_probe after each observe
    model.watch_activity(True)                    # model.last_activity: every neuron that fired
    model.clear()                                 # undo all of the above
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from neurofly_core.activity import SpikeAccumulator
from neurofly_core.brain.lif import LIFBrain
from neurofly_core.brain.plasticity import DopamineHebbian
from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.encode.audition import AuditionEncoder
from neurofly_core.encode.detection import DetectionEncoder
from neurofly_core.encode.vision import RetinaEncoder, luminance
from neurofly_core.selection import resolve


@dataclass
class ModelConfig:
    dt: float = 0.5                 # brain step, ms
    brain_ms: float = 10.0          # brain time per observation
    warmup_ms: float = 20.0         # brain time on the first frame of an episode
    readout_scale: float = 0.01     # rates in Hz times this = features
    include_frame: bool = False     # PC: append a luminance grid of the frame to the features
    frame_grid: tuple = (12, 16)
    include_audio: bool = False     # PC: append the audio band levels to the features
    include_detections: bool = False  # PC: append the detection grids to the features
    include_proprio: bool = False   # body: append the raw observation to the features
    plasticity: bool = False        # reward acts as dopamine on synapses onto the readout
    plasticity_lr: float = 1e-3
    dopamine_punish: float = 0.0    # mV on the punishment neurons while reward is negative
    name: str = "neurofly"
    meta: dict = field(default_factory=dict)   # provenance: subset, run, options

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frame_grid"] = list(self.frame_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["frame_grid"] = tuple(d.get("frame_grid", (12, 16)))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BrainModel:
    """What every controller shares: the brain, the readout, the policy slot, plasticity,
    punishment, neuron annotations and the experiment operations."""
    kind = "brain"

    def __init__(self, brain: LIFBrain, *, readout_idx, policy=None,
                 config: ModelConfig | None = None, punish_idx=None, neuron_ids=None,
                 neuron_types=None, neuron_superclass=None, neuron_positions=None,
                 positions_known=None):
        self.brain = brain
        self.readout_idx = np.asarray(readout_idx, dtype=np.int64)
        self.policy = policy
        self.config = config or ModelConfig()
        self.punish_idx = None if punish_idx is None else np.asarray(punish_idx, dtype=np.int64)
        self.neuron_ids = None if neuron_ids is None else np.asarray(neuron_ids, dtype=np.int64)
        self.neuron_types = None if neuron_types is None else np.asarray(neuron_types, dtype=object)
        self.neuron_superclass = (None if neuron_superclass is None
                                  else np.asarray(neuron_superclass, dtype=object))
        self.neuron_positions = None
        self.positions_known = None
        if neuron_positions is not None:
            self.neuron_positions = np.asarray(neuron_positions, dtype=np.float32).reshape(-1, 3)
            if self.neuron_positions.shape[0] != brain.n:
                raise ValueError("neuron_positions must have one row per neuron")
            known = (np.isfinite(self.neuron_positions).all(axis=1) if positions_known is None
                     else np.asarray(positions_known, dtype=bool))
            self.positions_known = known
        c = self.config
        self.substeps = max(1, int(round(c.brain_ms / brain.dt)))
        self.warmup_steps = int(round(c.warmup_ms / brain.dt))
        self.plasticity = None
        if c.plasticity:
            self.plasticity = DopamineHebbian(brain, pre_idx=np.arange(brain.n),
                                              post_idx=self.readout_idx, lr=c.plasticity_lr)
        self._punish = None
        if c.dopamine_punish > 0 and self.punish_idx is not None and len(self.punish_idx):
            self._punish = brain.drive(self.punish_idx, c.dopamine_punish)
        self._fresh = True
        self.t = 0
        self.last_spikes = 0
        self.probe_idx: np.ndarray | None = None
        self.last_probe: dict | None = None
        self.activity: SpikeAccumulator | None = None
        self.last_activity: dict | None = None
        self.manipulations: list[dict] = []

    # --- shapes (subclasses extend) ----------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.readout_idx)

    @property
    def n_actions(self) -> int:
        raise NotImplementedError

    # --- neuron selection and experiments ----------------------------------------------

    def populations(self) -> dict[str, np.ndarray]:
        named = {"readout": self.readout_idx}
        if self.punish_idx is not None:
            named["punish"] = self.punish_idx
        return named

    def select(self, sel: dict) -> np.ndarray:
        """Neuron indices for a selection dict (see ``neurofly_core.selection``)."""
        return resolve(sel, n=self.brain.n, ids=self.neuron_ids, types=self.neuron_types,
                       superclass=self.neuron_superclass, named=self.populations())

    def stimulate(self, sel: dict, mv: float) -> int:
        """Add ``mv`` of drive to the selected neurons on every brain step. Returns how many."""
        idx = self.select(sel)
        self.brain.stimulate(idx, mv)
        self.manipulations.append({"stimulate": sel, "mv": float(mv), "n": int(len(idx))})
        return int(len(idx))

    def silence(self, sel: dict) -> int:
        """Keep the selected neurons from spiking. Returns how many."""
        idx = self.select(sel)
        self.brain.silence(idx)
        self.manipulations.append({"silence": sel, "n": int(len(idx))})
        return int(len(idx))

    def probe(self, sel: dict | None) -> int:
        """Record the selected neurons: after every observe, ``last_probe`` holds their
        spike counts over the observation and their firing rates. None turns it off."""
        if sel is None:
            self.probe_idx, self.last_probe = None, None
            return 0
        self.probe_idx = self.select(sel)
        return int(len(self.probe_idx))

    def watch_activity(self, on: bool = True, substeps: bool = False) -> int:
        """Record every neuron's spikes: after each observe, ``last_activity`` holds the
        indices that fired and their counts (and, with ``substeps``, the set that fired
        on each brain step). Returns the neuron count, 0 when turned off."""
        if not on:
            self.activity, self.last_activity = None, None
            return 0
        self.activity = SpikeAccumulator(self.brain.n, substeps=substeps,
                                         device=self.brain.device)
        return int(self.brain.n)

    def activity_map(self) -> dict:
        """What a viewer needs to draw the brain: positions (micrometres), which of them
        are real somas, superclasses and the named populations."""
        sc = None
        if self.neuron_superclass is not None:
            sc = ["" if t is None else str(t) for t in self.neuron_superclass]
        return {"n": int(self.brain.n), "unit": "micrometre",
                "positions": self.neuron_positions, "known": self.positions_known,
                "superclass": sc, "populations": self.populations()}

    def clear(self) -> None:
        """Undo every stimulation and silencing; keep the probe."""
        self.brain.clear_manipulations()
        self.manipulations = []

    # --- the shared loop -----------------------------------------------------------------

    def reset(self) -> None:
        self.brain.reset()
        self._fresh = True
        self.t = 0
        self.last_spikes = 0
        self.last_probe = None
        self.last_activity = None

    def _run(self, drive: torch.Tensor, reward: float = 0.0) -> None:
        """Run the brain for one observation under ``drive`` (plus punishment when the
        reward is negative), with plasticity and the probe."""
        if reward < 0 and self._punish is not None:
            drive = drive + self._punish
        n_steps = self.substeps + (self.warmup_steps if self._fresh else 0)
        self._fresh = False
        before = self.brain.total_spikes
        probe_t = None
        if self.probe_idx is not None:
            probe_t = torch.as_tensor(self.probe_idx, device=self.brain.device)
            counts = torch.zeros(len(self.probe_idx), device=self.brain.device)
        if self.activity is not None:
            self.activity.begin()
        for _ in range(n_steps):
            spikes = self.brain.step(drive)
            if self.plasticity is not None:
                self.plasticity.step(reward)
            if probe_t is not None:
                counts += spikes[probe_t].to(counts.dtype)
            if self.activity is not None:
                self.activity.add(spikes)
        self.last_spikes = self.brain.total_spikes - before
        if probe_t is not None:
            self.last_probe = {"spikes": counts.cpu().numpy().astype(np.int64),
                               "rates": self.brain.rates(self.probe_idx)}
        if self.activity is not None:
            self.last_activity = self.activity.finish()
        self.t += 1

    def readout_features(self) -> np.ndarray:
        return self.brain.rates(self.readout_idx) * self.config.readout_scale

    def act(self, features: np.ndarray) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("this model has no policy; train one or pass actions yourself")
        return np.asarray(self.policy(features), dtype=np.float32)

    def describe_brain(self) -> list[str]:
        c = self.config
        pol = "none" if self.policy is None else self.policy.kind
        lines = [f"{c.name} ({self.kind}): {self.brain.n:,} neurons, {self.brain.n_edges:,} "
                 f"synapses, dt {self.brain.dt} ms, {c.brain_ms:g} ms per observation "
                 f"({self.brain.backend})",
                 f"  policy: {pol}; plasticity {'on' if self.plasticity else 'off'}; "
                 f"punishment {c.dopamine_punish:g} mV; annotations "
                 f"{'yes' if self.neuron_types is not None else 'no'}; positions "
                 f"{'yes' if self.neuron_positions is not None else 'no'}"]
        for m in self.manipulations:
            lines.append(f"  manipulation: {m}")
        return lines

    @property
    def device(self) -> torch.device:
        return self.brain.device


class Model(BrainModel):
    """The PC controller: frames and sound in, keys, mouse and gamepad out."""
    kind = "pc"

    def __init__(self, brain: LIFBrain, *, readout_idx, layout: ControlLayout,
                 retina: RetinaEncoder, audition: AuditionEncoder | None = None,
                 detection: DetectionEncoder | None = None,
                 policy=None, config: ModelConfig | None = None, punish_idx=None,
                 neuron_ids=None, neuron_types=None, neuron_superclass=None,
                 neuron_positions=None, positions_known=None):
        super().__init__(brain, readout_idx=readout_idx, policy=policy, config=config,
                         punish_idx=punish_idx, neuron_ids=neuron_ids, neuron_types=neuron_types,
                         neuron_superclass=neuron_superclass, neuron_positions=neuron_positions,
                         positions_known=positions_known)
        self.layout = layout
        self.retina = retina
        self.audition = audition
        self.detection = detection
        self.include_audio = self.config.include_audio and audition is not None
        self.include_detections = self.config.include_detections and detection is not None
        self._frame = None
        self._chunk = None

    @property
    def n_features(self) -> int:
        n = len(self.readout_idx)
        if self.config.include_frame:
            n += int(np.prod(self.config.frame_grid))
        if self.include_audio:
            n += self.audition.n_bands
        if self.include_detections:
            n += self.detection.n_inputs
        return n

    @property
    def n_actions(self) -> int:
        return self.layout.n

    def populations(self) -> dict[str, np.ndarray]:
        named = super().populations()
        named["retina"] = self.retina.indices if self.retina.mode == "hex" else self.retina.targets
        if self.audition is not None:
            named["audition"] = self.audition.targets
        if self.detection is not None:
            named["detection"] = self.detection.targets
        return named

    def reset(self) -> None:
        super().reset()
        self.retina.reset()
        if self.audition is not None:
            self.audition.reset()
        if self.detection is not None:
            self.detection.reset()
        self._frame = self._chunk = None

    def features(self) -> np.ndarray:
        parts = [self.readout_features()]
        if self.config.include_frame:
            parts.append(luminance(self._frame, self.config.frame_grid).ravel())
        if self.include_audio:
            parts.append(self.audition.bands(self._chunk))
        if self.include_detections:
            parts.append(self.detection.levels)
        return np.concatenate(parts).astype(np.float32)

    def observe(self, frame: np.ndarray, audio: np.ndarray | None = None,
                reward: float = 0.0, detections=None) -> np.ndarray:
        """Drive the neurons with a frame (RGB uint8), the sound since the last
        observation and, when the model has a detection encoder, the objects a detector
        found in the frame; run the brain for ``brain_ms``; return the feature vector.
        ``reward`` is dopamine for plasticity and, when negative, punishment."""
        self._frame = np.asarray(frame)
        self._chunk = audio
        drive = self.retina(self._frame)
        if self.audition is not None:
            drive = drive + self.audition(audio)
        if self.detection is not None:
            drive = drive + self.detection(detections)
        self._run(drive, reward)
        return self.features()

    def step(self, frame: np.ndarray, audio: np.ndarray | None = None,
             reward: float = 0.0, detections=None) -> tuple[ControlState, dict]:
        features = self.observe(frame, audio, reward, detections)
        action = self.act(features)
        state = self.layout.decode(action)
        info = {"t": self.t, "spikes": self.last_spikes, "action": action.tolist(),
                "held": state.held}
        if self.last_probe is not None:
            info["probe"] = self.last_probe
        return state, info

    def describe(self) -> str:
        c = self.config
        lines = self.describe_brain()
        r = self.retina
        lines.insert(1, f"  retina: {r.mode}, {r.n_driven} neurons driven, grid "
                        f"{r.grid[0]}x{r.grid[1]}, gain {r.gain:g} mV")
        if self.audition is not None:
            a = self.audition
            lines.insert(2, f"  audition: {a.n_driven} neurons driven, {a.n_bands} bands "
                            f"{a.fmin:g}-{a.fmax:g} Hz at {a.sample_rate} Hz, gain {a.gain:g} mV")
        lines.insert(-0 if not self.manipulations else -len(self.manipulations),
                     f"  readout: {len(self.readout_idx)} neurons -> {self.n_features} features"
                     + (" (+frame)" if c.include_frame else "")
                     + (" (+audio)" if self.include_audio else "")
                     + f"; controls: {self.layout.names}")
        return "\n".join(lines)
