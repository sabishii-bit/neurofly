"""Detected objects -> external drive on central-brain neurons.

A detector (any detector: it runs outside this package) turns a frame into boxes with
class labels. This encoder paints those boxes onto one coarse grid per class, so a
cell holds how much of it the strongest box of that class covers, and projects the
grids through a fixed sparse matrix onto a set of target neurons. Which neuron sees
which class at which place is a table built at training time; the classes are named
in the artifact so a consumer knows which ids to send.

Detections are a list of ``{"class": id or name, "box": [x0, y0, x1, y1], "score": s}``
with the box in fractions of the frame (0 to 1, left-top to right-bottom), or an
``(k, 6)`` array of ``[class_id, x0, y0, x1, y1, score]``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from neurofly_core.encode.vision import _csr_to_torch


def as_array(detections, classes) -> np.ndarray:
    """Detections in any accepted form -> ``(k, 6)`` float32 rows of
    ``[class_id, x0, y0, x1, y1, score]``; unknown class names are dropped."""
    if detections is None:
        return np.zeros((0, 6), np.float32)
    if isinstance(detections, np.ndarray) or (detections and not isinstance(detections[0], dict)):
        arr = np.asarray(detections, np.float32).reshape(-1, 6)
        return arr
    index = {str(name): i for i, name in enumerate(classes)}
    rows = []
    for d in detections:
        c = d.get("class", d.get("class_id"))
        if isinstance(c, str):
            if c not in index:
                continue
            c = index[c]
        box = d.get("box") or [d.get("x0"), d.get("y0"), d.get("x1"), d.get("y1")]
        rows.append([float(c), *[float(v) for v in box], float(d.get("score", 1.0))])
    return np.asarray(rows, np.float32).reshape(-1, 6)


def paint(dets: np.ndarray, n_classes: int, grid) -> np.ndarray:
    """``(n_classes, rows, cols)`` in [0, 1]: for every cell, the best score-weighted
    coverage by a box of that class."""
    rows, cols = int(grid[0]), int(grid[1])
    out = np.zeros((n_classes, rows, cols), np.float32)
    ys = np.linspace(0.0, 1.0, rows + 1)
    xs = np.linspace(0.0, 1.0, cols + 1)
    for c, x0, y0, x1, y1, score in dets:
        c = int(c)
        if not 0 <= c < n_classes or x1 <= x0 or y1 <= y0:
            continue
        x0, x1 = np.clip([x0, x1], 0.0, 1.0)
        y0, y1 = np.clip([y0, y1], 0.0, 1.0)
        # overlap of the box with every cell, as a fraction of the cell
        ox = np.clip(np.minimum(x1, xs[1:]) - np.maximum(x0, xs[:-1]), 0.0, None) * cols
        oy = np.clip(np.minimum(y1, ys[1:]) - np.maximum(y0, ys[:-1]), 0.0, None) * rows
        cover = np.outer(oy, ox) * float(np.clip(score, 0.0, 1.0))
        out[c] = np.maximum(out[c], cover.astype(np.float32))
    return out


class DetectionEncoder:
    def __init__(self, n_neurons: int, *, classes, matrix: sp.spmatrix, targets, grid=(6, 8),
                 gain: float = 15.0, device: str = "cpu"):
        """
        Args:
            classes: the class names, in id order; the artifact carries them.
            matrix: (n_neurons, n_classes * rows * cols) sparse projection.
            targets: the neurons the matrix drives (for selections and populations).
            grid: (rows, cols) cells per class.
            gain: mV of drive at full coverage.
        """
        self.n = int(n_neurons)
        self.classes = [str(c) for c in classes]
        self.grid = (int(grid[0]), int(grid[1]))
        self.gain = float(gain)
        self.device = torch.device(device)
        self.matrix = sp.csr_matrix(matrix, dtype=np.float32)
        self.matrix.sort_indices()
        if self.matrix.shape != (self.n, self.n_inputs):
            raise ValueError(f"matrix must be {self.n} x {self.n_inputs}, got {self.matrix.shape}")
        self.targets = np.asarray(targets, dtype=np.int64)
        self._M = _csr_to_torch(self.matrix, self.device)
        self._last = np.zeros(self.n_inputs, np.float32)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_inputs(self) -> int:
        return self.n_classes * self.grid[0] * self.grid[1]

    def reset(self) -> None:
        self._last[:] = 0

    def grids(self, detections) -> np.ndarray:
        """The painted grids, flattened (class-major, then row, then column)."""
        dets = as_array(detections, self.classes)
        self._last = paint(dets, self.n_classes, self.grid).ravel()
        return self._last

    @property
    def levels(self) -> np.ndarray:
        """The last grids, for appending to the feature vector."""
        return self._last

    def __call__(self, detections) -> torch.Tensor:
        g = torch.from_numpy(self.grids(detections)).to(self.device)
        return torch.relu((self._M @ g.unsqueeze(1)).squeeze(1)) * self.gain

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"classes": list(self.classes), "grid": list(self.grid), "gain": self.gain}

    def tables(self) -> dict[str, np.ndarray]:
        return {"matrix_indptr": self.matrix.indptr.astype(np.int64),
                "matrix_indices": self.matrix.indices.astype(np.int64),
                "matrix_values": self.matrix.data.astype(np.float32),
                "targets": self.targets}

    @classmethod
    def from_tables(cls, n_neurons: int, params: dict, tables: dict, device: str = "cpu"):
        n_in = len(params["classes"]) * int(params["grid"][0]) * int(params["grid"][1])
        M = sp.csr_matrix((tables["matrix_values"], tables["matrix_indices"],
                           tables["matrix_indptr"]), shape=(n_neurons, n_in))
        return cls(n_neurons, classes=params["classes"], matrix=M, targets=tables["targets"],
                   grid=params["grid"], gain=params["gain"], device=device)
