"""Connectome + options -> a runtime Model.

This is where the annotations become tables: which columnar neurons sample
which pixel, which neurons the projection encoders target, which neurons are
the readout and the punishment input. The resulting ``neurofly_core.Model``
carries only arrays, so it can be saved as an artifact and run anywhere.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from neurofly_core.controls import ControlLayout
from neurofly_core.brain.lif import LIFBrain
from neurofly_core.encode.audition import AuditionEncoder
from neurofly_core.encode.detection import DetectionEncoder
from neurofly_core.encode.olfaction import OlfactionEncoder
from neurofly_core.encode.vision import RetinaEncoder, hex_to_unit_square
from neurofly_core.model import Model, ModelConfig
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import RETINA_TYPES, Populations

ENCODER_SEED = 0   # encoders must be identical across parallel workers and at export


def _random_projection(rng, n_neurons: int, targets, n_inputs: int, k: int) -> sp.csr_matrix:
    """Each target neuron reads ``k`` random inputs with positive weights summing to about 1."""
    k = min(k, n_inputs)
    r_, c_, v_ = [], [], []
    for nrn in targets:
        pick = rng.choice(n_inputs, size=k, replace=False)
        r_ += [int(nrn)] * k
        c_ += [int(p) for p in pick]
        v_ += list(np.abs(rng.normal(1.0, 0.3, size=k)) / k)
    M = sp.csr_matrix((np.asarray(v_, np.float32), (r_, c_)), shape=(n_neurons, n_inputs))
    M.sum_duplicates()
    M.sort_indices()
    return M


def build_retina(pops: Populations, n_neurons: int, *, mode: str = "auto", types=RETINA_TYPES,
                 grid=(24, 32), gain: float = 15.0, temporal: float = 0.0,
                 inputs_per_neuron: int = 4, seed: int = ENCODER_SEED,
                 device: str = "cpu") -> RetinaEncoder:
    """``hex`` when the subset has columnar neurons with eye coordinates: each samples the
    pixel where its column looks, left eye on the left half of the frame. Otherwise
    ``projection`` onto the visual projection neurons (or the photoreceptors)."""
    retina = pops.retina(types)
    n_hex = sum(len(d["idx"]) for d in retina.values())
    if mode == "auto":
        mode = "hex" if n_hex >= 32 else "projection"
    rows, cols = grid
    if mode == "hex":
        if n_hex == 0:
            raise ValueError("retina mode 'hex' needs columnar optic-lobe neurons with "
                             "hex coordinates; use the 'visual' or 'brain' subset")
        idx, pix, on = [], [], []
        half = cols // 2
        for side, d in retina.items():
            if len(d["idx"]) == 0:
                continue
            uv = hex_to_unit_square(d["hex"])
            c = np.clip((uv[:, 0] * (half - 1)).round().astype(int), 0, half - 1)
            r = np.clip(((1.0 - uv[:, 1]) * (rows - 1)).round().astype(int), 0, rows - 1)
            if side == "R":
                c = c + half
            idx.append(d["idx"])
            pix.append(r * cols + c)
            on.append(d["on"])
        return RetinaEncoder(n_neurons, mode="hex", grid=grid, gain=gain, temporal=temporal,
                             indices=np.concatenate(idx), pixels=np.concatenate(pix),
                             on=np.concatenate(on), device=device)
    if mode != "projection":
        raise ValueError(f"unknown retina mode {mode!r}")
    targets = pops.visual_projection if len(pops.visual_projection) else pops.photoreceptors
    if len(targets) == 0:
        raise ValueError("no visual neurons in this connectome subset")
    M = _random_projection(np.random.default_rng(seed), n_neurons, targets, rows * cols,
                           inputs_per_neuron)
    return RetinaEncoder(n_neurons, mode="projection", grid=grid, gain=gain, temporal=temporal,
                         matrix=M, targets=targets, device=device)


def build_audition(pops: Populations, n_neurons: int, *, sample_rate: int = 16000,
                   n_bands: int = 16, fmin: float = 50.0, fmax: float | None = None,
                   gain: float = 15.0, loud_ref: float = 0.05, inputs_per_neuron: int = 3,
                   seed: int = ENCODER_SEED, device: str = "cpu") -> AuditionEncoder:
    """Band levels onto the auditory neurons (Johnston's organ), or the wind/gravity
    neurons when the subset has no auditory ones."""
    targets = pops.auditory if len(pops.auditory) else pops.wind_gravity
    if len(targets) == 0:
        raise ValueError("no auditory neurons in this connectome subset "
                         "(needs the head: subsets central, visual or brain)")
    M = _random_projection(np.random.default_rng(seed + 1), n_neurons, targets, n_bands,
                           inputs_per_neuron)
    return AuditionEncoder(n_neurons, matrix=M, targets=targets, sample_rate=sample_rate,
                           n_bands=n_bands, fmin=fmin, fmax=fmax, gain=gain, loud_ref=loud_ref,
                           device=device)


def build_detection(pops: Populations, n_neurons: int, classes, *, grid=(6, 8),
                    gain: float = 15.0, inputs_per_neuron: int = 1, seed: int = ENCODER_SEED,
                    device: str = "cpu") -> DetectionEncoder:
    """Detected objects onto the central-brain interneurons (or the visual projection
    neurons without a central brain): each target neuron is tuned to one class at one
    place, a labelled line, so the readout can tell what is where."""
    targets = pops.cb_intrinsic if len(pops.cb_intrinsic) else pops.visual_projection
    if len(targets) == 0:
        raise ValueError("no central-brain or visual projection neurons for detections "
                         "(needs the head: subsets central, visual or brain)")
    n_in = len(classes) * int(grid[0]) * int(grid[1])
    M = _random_projection(np.random.default_rng(seed + 2), n_neurons, targets, n_in,
                           inputs_per_neuron)
    return DetectionEncoder(n_neurons, classes=classes, matrix=M, targets=targets, grid=grid,
                            gain=gain, device=device)


def build_sense(groups: dict, n_neurons: int, channels, *, gain: float = 15.0,
                adapt: float = 0.0, device: str = "cpu", what: str = "sense") -> OlfactionEncoder:
    """Each named channel onto one group of receptor neurons (a glomerulus, a taste
    neuron type, a thermo/hygro type), in name order; more channels than groups wrap
    around and share."""
    if not groups:
        raise ValueError(f"no {what} receptor neurons in this connectome subset "
                         "(needs the head: subsets central, visual or brain)")
    names = list(groups)
    r_, c_ = [], []
    for j, _ in enumerate(channels):
        for nrn in groups[names[j % len(names)]]:
            r_.append(int(nrn))
            c_.append(j)
    M = sp.csr_matrix((np.ones(len(r_), np.float32), (r_, c_)),
                      shape=(n_neurons, len(channels)))
    M.sum_duplicates()
    M.sort_indices()
    targets = np.unique(np.asarray(r_, dtype=np.int64))
    return OlfactionEncoder(n_neurons, channels=list(channels), matrix=M, targets=targets,
                            gain=gain, adapt=adapt, device=device)


def build_olfaction(pops: Populations, n_neurons: int, channels, **kw) -> OlfactionEncoder:
    """Odour channels onto the glomeruli of the antennal lobe."""
    return build_sense(pops.glomeruli, n_neurons, channels, what="olfactory", **kw)


def build_gustation(pops: Populations, n_neurons: int, channels, **kw) -> OlfactionEncoder:
    """Taste channels onto the gustatory receptor neuron types (legs and proboscis)."""
    return build_sense(pops.taste_types, n_neurons, channels, what="gustatory", **kw)


def build_thermo(pops: Populations, n_neurons: int, channels, **kw) -> OlfactionEncoder:
    """Temperature and humidity channels onto the thermo- and hygrosensory types."""
    return build_sense(pops.thermo_types, n_neurons, channels, what="thermo/hygro", **kw)


def build_touch(pops: Populations, n_neurons: int, channels, **kw) -> OlfactionEncoder:
    """Touch channels onto the head's bristle and grooming neurons and the legs' tactile
    neurons, one group per channel (a channel named like a group, e.g. ``leg:T1L`` or
    ``head:grooming``, gets that group; others take groups in order)."""
    groups = dict(pops.touch_types)
    if not groups:
        raise ValueError("no touch neurons in this connectome subset")
    ordered = {}
    rest = [g for g in groups if g not in channels]
    for ch in channels:
        if ch in groups:
            ordered[ch] = groups[ch]
        else:
            ordered[ch] = groups[rest.pop(0) if rest else list(groups)[len(ordered) % len(groups)]]
    return build_sense(ordered, n_neurons, channels, what="touch", **kw)


def build_model(cx: Connectome, layout: ControlLayout, *, readout="descending", dt: float = 0.5,
                brain_ms: float = 10.0, brain_gain: float = 1.0, warmup_ms: float = 20.0,
                retina_mode: str = "auto", retina_gain: float = 15.0,
                retina_temporal: float = 0.0, retina_grid=(24, 32), audio: bool = False,
                sample_rate: int = 16000, audio_bands: int = 16, audio_gain: float = 15.0,
                include_frame: bool = False, frame_grid=(12, 16), include_audio: bool = False,
                detect_classes=None, detection_grid=(6, 8), detection_gain: float = 15.0,
                include_detections: bool = False, odour_channels=None,
                odour_gain: float = 15.0, odour_adapt: float = 0.0,
                include_odours: bool = False, taste_channels=None, taste_gain: float = 15.0,
                include_tastes: bool = False, thermo_channels=None, thermo_gain: float = 15.0,
                include_thermo: bool = False, touch_channels=None, touch_gain: float = 15.0,
                include_touch: bool = False, plasticity_target: str = "readout",
                plasticity: bool = False, dopamine_punish: float = 0.0,
                dopamine_reward: float = 0.0, policy=None,
                name: str | None = None, meta: dict | None = None, device: str = "cpu",
                backend: str = "auto") -> Model:
    """Assemble a ``Model`` from a connectome subset and the interface options."""
    pops = Populations(cx)
    brain = LIFBrain(cx.W, dt=dt, gain=brain_gain, device=device, backend=backend)
    retina = build_retina(pops, cx.n, mode=retina_mode, grid=retina_grid, gain=retina_gain,
                          temporal=retina_temporal, device=device)
    audition = None
    if audio:
        audition = build_audition(pops, cx.n, sample_rate=sample_rate, n_bands=audio_bands,
                                  gain=audio_gain, device=device)
    detection = None
    if detect_classes:
        detection = build_detection(pops, cx.n, list(detect_classes), grid=detection_grid,
                                    gain=detection_gain, device=device)
    olfaction = gustation = thermo = None
    if odour_channels:
        olfaction = build_olfaction(pops, cx.n, list(odour_channels), gain=odour_gain,
                                    adapt=odour_adapt, device=device)
    if taste_channels:
        gustation = build_gustation(pops, cx.n, list(taste_channels), gain=taste_gain,
                                    device=device)
    if thermo_channels:
        thermo = build_thermo(pops, cx.n, list(thermo_channels), gain=thermo_gain, device=device)
    touch = None
    if touch_channels:
        touch = build_touch(pops, cx.n, list(touch_channels), gain=touch_gain, device=device)
    readout_idx = pops.readout(readout) if isinstance(readout, str) else np.asarray(readout)
    punish = pops.ppl1 if len(pops.ppl1) else pops.dopamine
    config = ModelConfig(dt=dt, brain_ms=brain_ms, warmup_ms=warmup_ms,
                         include_frame=include_frame, frame_grid=tuple(frame_grid),
                         include_audio=include_audio, include_detections=include_detections,
                         include_odours=include_odours, include_tastes=include_tastes,
                         include_thermo=include_thermo, include_touch=include_touch,
                         plasticity=plasticity,
                         plasticity_target=plasticity_target,
                         dopamine_punish=dopamine_punish, dopamine_reward=dopamine_reward,
                         name=name or cx.name,
                         meta=dict(meta or {}, connectome=cx.name, n_neurons=cx.n,
                                   readout=readout if isinstance(readout, str) else "custom"))
    positions, known = cx.positions()
    return Model(brain, readout_idx=readout_idx, layout=layout, retina=retina,
                 audition=audition, detection=detection, olfaction=olfaction,
                 gustation=gustation, thermo=thermo, touch=touch, policy=policy, config=config,
                 punish_idx=punish, reward_idx=pops.pam,
                 neuron_ids=cx.neurons["bodyId"].values,
                 neuron_types=cx.neurons["type"].values,
                 neuron_superclass=cx.neurons["superclass"].values,
                 neuron_positions=positions, positions_known=known)
