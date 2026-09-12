"""Video frame -> external drive on visual neurons (the fly's retina).

Two modes, both defined by tables:

* ``hex``: each columnar optic-lobe neuron samples one pixel of a luminance
  grid; the left half of the frame is the left eye's view, the right half the
  right eye's. ON-pathway cells are driven by brightness, OFF-pathway cells by
  darkness. Tables: ``indices`` (neuron ids), ``pixels`` (flat grid index per
  neuron), ``on`` (bool per neuron).
* ``projection``: the luminance grid is projected through a fixed sparse
  matrix onto a set of target neurons, with the values contrast-normalised
  across the targets so overall brightness does not matter. Tables: the
  matrix (n_neurons x n_pixels, CSR) and ``targets``.

Either way one frame becomes one (n_neurons,) drive vector in mV.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image

SQRT3_2 = np.sqrt(3.0) / 2.0


def luminance(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Grayscale frame downsampled by area averaging to ``size = (rows, cols)``, in [0, 1]."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    img = Image.fromarray(frame)
    if img.mode != "L":
        img = img.convert("L")
    rows, cols = size
    img = img.resize((cols, rows), Image.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


def hex_to_unit_square(hex_xy: np.ndarray) -> np.ndarray:
    """Hex (p, q) coordinates -> (u, v) in the unit square, axes at 60 degrees."""
    p, q = hex_xy[:, 0], hex_xy[:, 1]
    x = p + 0.5 * q
    y = q * SQRT3_2
    uv = np.stack([x, y], axis=1)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    return (uv - lo) / np.maximum(hi - lo, 1e-9)


def _csr_to_torch(M: sp.csr_matrix, device) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(M.indptr.astype(np.int64)),
        torch.from_numpy(M.indices.astype(np.int64)),
        torch.from_numpy(M.data.astype(np.float32)), size=M.shape).to(device)


class RetinaEncoder:
    def __init__(self, n_neurons: int, *, mode: str, grid=(24, 32), gain: float = 15.0,
                 temporal: float = 0.0, indices=None, pixels=None, on=None,
                 matrix: sp.spmatrix | None = None, targets=None, device: str = "cpu"):
        """
        Args:
            grid: (rows, cols) of the luminance grid the frame is reduced to.
            gain: mV of drive at full brightness.
            temporal: 0 = respond to brightness; 1 = respond only to brightness
                change between frames (ON cells to increases, OFF to decreases).
        """
        self.n = int(n_neurons)
        self.mode = mode
        self.grid = tuple(int(g) for g in grid)
        self.gain, self.temporal = float(gain), float(temporal)
        self.device = torch.device(device)
        if mode == "hex":
            self.indices = np.asarray(indices, dtype=np.int64)
            self.pixels = np.asarray(pixels, dtype=np.int64)
            self.on = np.asarray(on, dtype=bool)
            self._idx_t = torch.from_numpy(self.indices).to(self.device)
            self.n_driven = int(len(self.indices))
        elif mode == "projection":
            self.matrix = sp.csr_matrix(matrix, dtype=np.float32)
            self.matrix.sort_indices()
            self.targets = np.asarray(targets, dtype=np.int64)
            self._M = _csr_to_torch(self.matrix, self.device)
            self._targets_t = torch.from_numpy(self.targets).to(self.device)
            self.n_driven = int(len(self.targets))
        else:
            raise ValueError(f"unknown retina mode {mode!r}")
        self._prev = None

    def reset(self) -> None:
        self._prev = None

    def signals(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(on, off) luminance-grid signals in [0, 1], flattened."""
        lum = luminance(frame, self.grid).ravel()
        prev = lum if self._prev is None else self._prev
        self._prev = lum
        d = lum - prev
        t = self.temporal
        on = (1 - t) * lum + t * np.clip(4.0 * np.maximum(d, 0), 0, 1)
        off = (1 - t) * (1 - lum) + t * np.clip(4.0 * np.maximum(-d, 0), 0, 1)
        return on.astype(np.float32), off.astype(np.float32)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        on, off = self.signals(frame)
        drive = torch.zeros(self.n, device=self.device)
        if self.mode == "hex":
            s = np.where(self.on, on[self.pixels], off[self.pixels]) * self.gain
            drive[self._idx_t] = torch.from_numpy(s.astype(np.float32)).to(self.device)
            return drive
        x = torch.from_numpy(on).to(self.device)
        y = (self._M @ x.unsqueeze(1)).squeeze(1)
        yt = y[self._targets_t]
        z = (yt - yt.mean()) / (yt.std() + 1e-6)      # contrast across neurons
        drive[self._targets_t] = torch.relu(z + 0.5) * self.gain
        return drive

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"mode": self.mode, "grid": list(self.grid), "gain": self.gain,
                "temporal": self.temporal}

    def tables(self) -> dict[str, np.ndarray]:
        if self.mode == "hex":
            return {"indices": self.indices, "pixels": self.pixels, "on": self.on}
        return {"matrix_indptr": self.matrix.indptr.astype(np.int64),
                "matrix_indices": self.matrix.indices.astype(np.int64),
                "matrix_values": self.matrix.data.astype(np.float32),
                "targets": self.targets}

    @classmethod
    def from_tables(cls, n_neurons: int, params: dict, tables: dict, device: str = "cpu"):
        if params["mode"] == "hex":
            return cls(n_neurons, mode="hex", grid=params["grid"], gain=params["gain"],
                       temporal=params["temporal"], indices=tables["indices"],
                       pixels=tables["pixels"], on=tables["on"], device=device)
        n_pix = int(np.prod(params["grid"]))
        M = sp.csr_matrix((tables["matrix_values"], tables["matrix_indices"],
                           tables["matrix_indptr"]), shape=(n_neurons, n_pix))
        return cls(n_neurons, mode="projection", grid=params["grid"], gain=params["gain"],
                   temporal=params["temporal"], matrix=M, targets=tables["targets"],
                   device=device)
