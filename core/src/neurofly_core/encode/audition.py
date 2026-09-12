"""Sound -> external drive on auditory neurons.

The chunk of audio since the last step is turned into a coarse log-spaced band
spectrum (the shape of the spectrum, scaled by loudness) and projected through
a fixed sparse matrix onto the auditory neurons. Which neuron hears which band
is a table built at training time.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from neurofly_core.encode.vision import _csr_to_torch


class AuditionEncoder:
    def __init__(self, n_neurons: int, *, matrix: sp.spmatrix, targets, sample_rate: int = 16000,
                 n_bands: int = 16, fmin: float = 50.0, fmax: float | None = None,
                 gain: float = 15.0, loud_ref: float = 0.05, device: str = "cpu"):
        """
        Args:
            matrix: (n_neurons, n_bands) sparse projection.
            n_bands: log-spaced bands between ``fmin`` and ``fmax`` (default Nyquist).
            gain: mV of drive at full band level.
            loud_ref: RMS amplitude (of samples in [-1, 1]) that counts as full loudness.
        """
        self.n = int(n_neurons)
        self.sample_rate = int(sample_rate)
        self.n_bands = int(n_bands)
        self.fmin = float(fmin)
        self.fmax = float(fmax or self.sample_rate / 2.0)
        self.gain, self.loud_ref = float(gain), float(loud_ref)
        self.device = torch.device(device)
        self.edges = np.geomspace(self.fmin, self.fmax, self.n_bands + 1)
        self.matrix = sp.csr_matrix(matrix, dtype=np.float32)
        self.matrix.sort_indices()
        self.targets = np.asarray(targets, dtype=np.int64)
        self._M = _csr_to_torch(self.matrix, self.device)
        self.n_driven = int(len(self.targets))
        self._last = np.zeros(self.n_bands, np.float32)

    def reset(self) -> None:
        self._last[:] = 0

    def bands(self, chunk: np.ndarray | None) -> np.ndarray:
        """Band levels in [0, 1]: spectral shape times loudness. Empty chunk -> last value."""
        if chunk is None:
            return np.zeros(self.n_bands, np.float32)
        x = np.asarray(chunk, np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if len(x) < 32:
            return self._last
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < 1e-5:
            self._last = np.zeros(self.n_bands, np.float32)
            return self._last
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
        e = np.zeros(self.n_bands)
        for j, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            sel = (freqs >= lo) & (freqs < hi)
            e[j] = spec[sel].mean() if sel.any() else 0.0
        lvl = np.log10(e + 1e-12)
        shape = (lvl - lvl.min()) / (np.ptp(lvl) + 1e-6)
        self._last = (shape * min(1.0, rms / self.loud_ref)).astype(np.float32)
        return self._last

    def __call__(self, chunk: np.ndarray | None) -> torch.Tensor:
        b = torch.from_numpy(self.bands(chunk)).to(self.device)
        return torch.relu((self._M @ b.unsqueeze(1)).squeeze(1)) * self.gain

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"sample_rate": self.sample_rate, "n_bands": self.n_bands, "fmin": self.fmin,
                "fmax": self.fmax, "gain": self.gain, "loud_ref": self.loud_ref}

    def tables(self) -> dict[str, np.ndarray]:
        return {"matrix_indptr": self.matrix.indptr.astype(np.int64),
                "matrix_indices": self.matrix.indices.astype(np.int64),
                "matrix_values": self.matrix.data.astype(np.float32),
                "targets": self.targets}

    @classmethod
    def from_tables(cls, n_neurons: int, params: dict, tables: dict, device: str = "cpu"):
        M = sp.csr_matrix((tables["matrix_values"], tables["matrix_indices"],
                           tables["matrix_indptr"]), shape=(n_neurons, int(params["n_bands"])))
        return cls(n_neurons, matrix=M, targets=tables["targets"], device=device,
                   **{k: params[k] for k in ("sample_rate", "n_bands", "fmin", "fmax", "gain",
                                             "loud_ref")})
