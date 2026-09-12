"""Brain readout rates -> one control action vector, as a linear map.

The output is the action vector of a ``ControlLayout``: keys and buttons are
held while their entry is positive, mouse axes and scroll are the entry
itself. The weights are a flat parameter vector, so evolution strategies can
move them, and imitation can fit them to a recorded human.
"""
from __future__ import annotations

import json

import numpy as np

from neurofly_core.controls import ControlLayout, ControlState


class ControlDecoder:
    kind = "linear"

    def __init__(self, n_features: int, layout: ControlLayout, *, scale: float = 0.05,
                 seed: int = 0, W=None, b=None):
        rng = np.random.default_rng(seed)
        self.layout = layout
        self.n_features = int(n_features)
        self.W = (np.asarray(W, dtype=np.float64) if W is not None
                  else rng.normal(scale=scale, size=(layout.n, self.n_features)))
        self.b = np.asarray(b, dtype=np.float64) if b is not None else np.zeros(layout.n)

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(features, dtype=np.float64) + self.b

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Action vector in [-1, 1]."""
        return np.tanh(self.logits(features)).astype(np.float32)

    def state(self, features: np.ndarray) -> ControlState:
        return self.layout.decode(self(features))

    # --- flat parameter vector, for evolution strategies -----------------------

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b])

    def set_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float64)
        self.W = theta[:self.W.size].reshape(self.W.shape).copy()
        self.b = theta[self.W.size:self.W.size + self.b.size].copy()

    def save(self, path: str) -> None:
        np.savez(path, theta=self.get_params(), n_features=self.n_features,
                 layout=json.dumps(self.layout.to_dict()))

    def load(self, path: str) -> None:
        z = np.load(path)
        if "layout" in z:
            names = ControlLayout.from_dict(json.loads(str(z["layout"]))).names
            if names != self.layout.names:
                raise ValueError(f"decoder was trained for controls {names}, "
                                 f"this layout has {self.layout.names}")
        self.set_params(z["theta"])

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"type": "linear", "output": "tanh"}

    def tables(self) -> dict[str, np.ndarray]:
        return {"W": self.W.astype(np.float32), "b": self.b.astype(np.float32)}

    @classmethod
    def from_tables(cls, layout: ControlLayout, params: dict, tables: dict):
        return cls(tables["W"].shape[1], layout, W=tables["W"], b=tables["b"])
