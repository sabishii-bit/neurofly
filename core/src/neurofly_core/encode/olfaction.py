"""Odours -> external drive on olfactory receptor neurons.

The fly smells through about fifty kinds of receptor neuron, one kind per glomerulus of
the antennal lobe, and its learning circuit (the mushroom body, taught by the dopamine
neurons this project already drives with punishment) is built around them. Here an
"odour" is any slow scalar a task cares about: health, ammo, distance to the goal, "an
enemy is in view". Each named channel drives the receptor neurons of one glomerulus;
which glomerulus is a table built at training time and named in the artifact, so the
brain's own associative memory has something to associate the punishment with.

Input: a vector with one value per channel in [0, 1] (or a dict of channel name to
value; missing channels are 0). Values are held between observations, so a channel that
is set once stays on until it is set again or the model is reset.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from neurofly_core.encode.vision import _csr_to_torch


class OlfactionEncoder:
    def __init__(self, n_neurons: int, *, channels, matrix: sp.spmatrix, targets,
                 gain: float = 15.0, adapt: float = 0.0, device: str = "cpu"):
        """
        Args:
            channels: the odour names, in order; the artifact carries them.
            matrix: (n_neurons, n_channels) sparse projection.
            targets: the neurons the matrix drives (for selections and populations).
            gain: mV of drive at a channel value of 1.
            adapt: 0 to 1: how much of the drive fades while a channel stays constant
                (receptor adaptation); 0 keeps the drive steady.
        """
        self.n = int(n_neurons)
        self.channels = [str(c) for c in channels]
        self.gain, self.adapt = float(gain), float(adapt)
        self.device = torch.device(device)
        self.matrix = sp.csr_matrix(matrix, dtype=np.float32)
        self.matrix.sort_indices()
        if self.matrix.shape != (self.n, self.n_channels):
            raise ValueError(f"matrix must be {self.n} x {self.n_channels}, "
                             f"got {self.matrix.shape}")
        self.targets = np.asarray(targets, dtype=np.int64)
        self._M = _csr_to_torch(self.matrix, self.device)
        self._last = np.zeros(self.n_channels, np.float32)
        self._adapted = np.zeros(self.n_channels, np.float32)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def reset(self) -> None:
        self._last[:] = 0
        self._adapted[:] = 0

    def vector(self, odours) -> np.ndarray:
        """Odours in any accepted form -> the held channel vector in [0, 1]."""
        if odours is None:
            return self._last
        if isinstance(odours, dict):
            v = self._last.copy()
            for k, val in odours.items():
                if k in self.channels:
                    v[self.channels.index(k)] = float(val)
        else:
            v = np.zeros(self.n_channels, np.float32)
            arr = np.asarray(odours, np.float32).ravel()
            v[:min(len(arr), self.n_channels)] = arr[:self.n_channels]
        self._last = np.clip(v, 0.0, 1.0).astype(np.float32)
        return self._last

    @property
    def levels(self) -> np.ndarray:
        """The held channel vector, for appending to the feature vector."""
        return self._last

    def __call__(self, odours) -> torch.Tensor:
        v = self.vector(odours)
        if self.adapt > 0:
            # a slow trace of each channel; the drive is what is left after adaptation
            self._adapted += 0.1 * (v - self._adapted)
            v = np.clip(v - self.adapt * self._adapted, 0.0, 1.0).astype(np.float32)
        x = torch.from_numpy(v).to(self.device)
        return torch.relu((self._M @ x.unsqueeze(1)).squeeze(1)) * self.gain

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"channels": list(self.channels), "gain": self.gain, "adapt": self.adapt}

    def tables(self) -> dict[str, np.ndarray]:
        return {"matrix_indptr": self.matrix.indptr.astype(np.int64),
                "matrix_indices": self.matrix.indices.astype(np.int64),
                "matrix_values": self.matrix.data.astype(np.float32),
                "targets": self.targets}

    @classmethod
    def from_tables(cls, n_neurons: int, params: dict, tables: dict, device: str = "cpu"):
        M = sp.csr_matrix((tables["matrix_values"], tables["matrix_indices"],
                           tables["matrix_indptr"]), shape=(n_neurons, len(params["channels"])))
        return cls(n_neurons, channels=params["channels"], matrix=M, targets=tables["targets"],
                   gain=params["gain"], adapt=params.get("adapt", 0.0), device=device)
