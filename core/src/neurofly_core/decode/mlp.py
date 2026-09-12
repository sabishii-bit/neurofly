"""A trained policy network as plain arrays.

Observation normalisation (mean, variance, clip) followed by dense layers with
tanh or relu, and the output clipped to [-1, 1]. This is what a PPO policy
from the training package becomes when exported: no framework needed to run it.
"""
from __future__ import annotations

import numpy as np

from neurofly_core.controls import ControlLayout, ControlState

_ACT = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0), "linear": lambda x: x}


class MLPPolicy:
    kind = "mlp"

    def __init__(self, layout: ControlLayout, layers: list[tuple[np.ndarray, np.ndarray]],
                 activation: str = "tanh", obs_mean=None, obs_var=None,
                 obs_clip: float = 10.0, obs_eps: float = 1e-8):
        """``layers`` is a list of (W, b) with W of shape (out, in); the activation is
        applied after every layer but the last. The last layer's width is ``layout.n``."""
        self.layout = layout
        self.layers = [(np.asarray(W, np.float64), np.asarray(b, np.float64)) for W, b in layers]
        if self.layers[-1][0].shape[0] != layout.n:
            raise ValueError(f"policy outputs {self.layers[-1][0].shape[0]} values, "
                             f"layout needs {layout.n}")
        if activation not in _ACT:
            raise ValueError(f"unknown activation {activation!r}")
        self.activation = activation
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, np.float64)
        self.obs_var = None if obs_var is None else np.asarray(obs_var, np.float64)
        self.obs_clip, self.obs_eps = float(obs_clip), float(obs_eps)

    @property
    def n_features(self) -> int:
        return int(self.layers[0][0].shape[1])

    def normalize(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, np.float64)
        if self.obs_mean is not None:
            x = (x - self.obs_mean) / np.sqrt(self.obs_var + self.obs_eps)
            x = np.clip(x, -self.obs_clip, self.obs_clip)
        return x

    def __call__(self, features: np.ndarray) -> np.ndarray:
        x = self.normalize(features)
        act = _ACT[self.activation]
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = act(x)
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    def state(self, features: np.ndarray) -> ControlState:
        return self.layout.decode(self(features))

    # --- for the artifact ---------------------------------------------------------

    def params(self) -> dict:
        return {"type": "mlp", "activation": self.activation, "n_layers": len(self.layers),
                "obs_clip": self.obs_clip, "obs_eps": self.obs_eps,
                "obs_normalized": self.obs_mean is not None, "output": "clip"}

    def tables(self) -> dict[str, np.ndarray]:
        t = {}
        for i, (W, b) in enumerate(self.layers):
            t[f"W{i}"] = W.astype(np.float32)
            t[f"b{i}"] = b.astype(np.float32)
        if self.obs_mean is not None:
            t["obs_mean"] = self.obs_mean.astype(np.float32)
            t["obs_var"] = self.obs_var.astype(np.float32)
        return t

    @classmethod
    def from_tables(cls, layout: ControlLayout, params: dict, tables: dict):
        layers = [(tables[f"W{i}"], tables[f"b{i}"]) for i in range(int(params["n_layers"]))]
        return cls(layout, layers, activation=params["activation"],
                   obs_mean=tables.get("obs_mean"), obs_var=tables.get("obs_var"),
                   obs_clip=params["obs_clip"], obs_eps=params["obs_eps"])
