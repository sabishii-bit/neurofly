"""Brain readout rates -> body actions, as a structured linear map.

Each leg's 9 actuators read only that leg's motor neurons; the head and
abdomen actuators read the descending neurons. This is the "fixed
neuron-to-button interface" of the game demos, except that here its weights
are a parameter vector you can train (see scripts/train_es.py) instead of
being hand-picked.
"""
from __future__ import annotations

import numpy as np

from flybrain_body.body.actuators import HEAD_ABDOMEN_INDICES, leg_actuator_indices
from flybrain_body.data.connectome import LEGS
from flybrain_body.interface.populations import Populations


class LinearDecoder:
    def __init__(self, pops: Populations, readout_idx: np.ndarray, n_actions: int = 59,
                 scale: float = 0.05, seed: int = 0):
        """
        Args:
            pops: populations of the connectome the brain is running.
            readout_idx: the neuron indices whose rates form the feature
                vector (the env's ``readout_idx``); blocks are located
                inside it.
        """
        rng = np.random.default_rng(seed)
        pos = {int(n): i for i, n in enumerate(readout_idx)}
        self.n_actions = n_actions
        self.blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (act_idx, feat_idx, W)
        for t, side in LEGS:
            feats = np.array([pos[int(n)] for n in pops.leg_motor[(t, side)] if int(n) in pos])
            if len(feats):
                acts = leg_actuator_indices(t, side)
                self.blocks.append((acts, feats, rng.normal(scale=scale, size=(len(acts), len(feats)))))
        feats = np.array([pos[int(n)] for n in pops.descending if int(n) in pos])
        if len(feats):
            self.blocks.append((HEAD_ABDOMEN_INDICES, feats,
                                rng.normal(scale=scale, size=(len(HEAD_ABDOMEN_INDICES), len(feats)))))
        self.bias = np.zeros(n_actions)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """features: the env feature vector (rates / 100 Hz). Returns action in [-1, 1]."""
        a = self.bias.copy()
        for acts, feats, W in self.blocks:
            a[acts] += W @ features[feats]
        return np.tanh(a).astype(np.float32)

    # --- flat parameter vector, for evolution strategies -----------------------

    @property
    def n_params(self) -> int:
        return int(sum(W.size for _, _, W in self.blocks) + self.bias.size)

    def get_params(self) -> np.ndarray:
        return np.concatenate([W.ravel() for _, _, W in self.blocks] + [self.bias])

    def set_params(self, theta: np.ndarray) -> None:
        i = 0
        for k, (acts, feats, W) in enumerate(self.blocks):
            self.blocks[k] = (acts, feats, theta[i:i + W.size].reshape(W.shape).copy())
            i += W.size
        self.bias = theta[i:i + self.bias.size].copy()

    def save(self, path: str) -> None:
        np.savez(path, theta=self.get_params())

    def load(self, path: str) -> None:
        self.set_params(np.load(path)["theta"])
