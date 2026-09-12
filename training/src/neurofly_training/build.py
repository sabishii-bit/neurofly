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


def build_model(cx: Connectome, layout: ControlLayout, *, readout="descending", dt: float = 0.5,
                brain_ms: float = 10.0, brain_gain: float = 1.0, warmup_ms: float = 20.0,
                retina_mode: str = "auto", retina_gain: float = 15.0,
                retina_temporal: float = 0.0, retina_grid=(24, 32), audio: bool = False,
                sample_rate: int = 16000, audio_bands: int = 16, audio_gain: float = 15.0,
                include_frame: bool = False, frame_grid=(12, 16), include_audio: bool = False,
                plasticity: bool = False, dopamine_punish: float = 0.0, policy=None,
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
    readout_idx = pops.readout(readout) if isinstance(readout, str) else np.asarray(readout)
    punish = pops.ppl1 if len(pops.ppl1) else pops.dopamine
    config = ModelConfig(dt=dt, brain_ms=brain_ms, warmup_ms=warmup_ms,
                         include_frame=include_frame, frame_grid=tuple(frame_grid),
                         include_audio=include_audio, plasticity=plasticity,
                         dopamine_punish=dopamine_punish, name=name or cx.name,
                         meta=dict(meta or {}, connectome=cx.name, n_neurons=cx.n,
                                   readout=readout if isinstance(readout, str) else "custom"))
    return Model(brain, readout_idx=readout_idx, layout=layout, retina=retina,
                 audition=audition, policy=policy, config=config, punish_idx=punish,
                 neuron_ids=cx.neurons["bodyId"].values,
                 neuron_types=cx.neurons["type"].values,
                 neuron_superclass=cx.neurons["superclass"].values)
