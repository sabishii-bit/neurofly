"""The body controller: a body observation vector in, actuator commands out.

The fly body is a MuJoCo model; the simulation runs wherever MuJoCo runs (the
training package with ``flybody``, a game engine plugin, MuJoCo's WebAssembly
build in a browser). What travels in a body artifact is the controller:

    observation (joints, contacts, gyro, ...) -> proprio map -> brain -> readout -> policy
    -> 59 actuators

``ProprioMap`` applies the sensory tables built at training time (a fixed sparse
projection of the observation onto sensory neurons, contact sensors squashed by
tanh). ``ActuatorDecoder`` is the linear neuron-to-actuator table;
``MLPPolicy`` from ``neurofly_core.decode.mlp`` is the exported PPO policy.
The artifact records the observation layout (names and slices) and the actuator
names, so a consumer assembling observations from its own MuJoCo instance knows
what to put where.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from neurofly_core.encode.vision import _csr_to_torch
from neurofly_core.model import BrainModel, ModelConfig


class ProprioMap:
    def __init__(self, n_neurons: int, *, matrix: sp.spmatrix, gain: float = 12.0,
                 touch: tuple[int, int] | None = None, touch_scale: float = 1.0,
                 device: str = "cpu"):
        self.n = int(n_neurons)
        self.matrix = sp.csr_matrix(matrix, dtype=np.float32)
        self.matrix.sort_indices()
        self.obs_dim = int(self.matrix.shape[1])
        self.gain, self.touch_scale = float(gain), float(touch_scale)
        self.touch = None if touch is None else (int(touch[0]), int(touch[1]))
        self.device = torch.device(device)
        self._M = _csr_to_torch(self.matrix, self.device)
        self.n_driven = int(np.count_nonzero(np.diff(self.matrix.indptr)))

    def __call__(self, obs: np.ndarray) -> torch.Tensor:
        x = np.asarray(obs, dtype=np.float32).copy()
        if len(x) != self.obs_dim:
            raise ValueError(f"observation has {len(x)} entries, the map expects {self.obs_dim}")
        if self.touch is not None:
            a, b = self.touch
            x[a:b] = np.tanh(x[a:b] * self.touch_scale)
        xt = torch.from_numpy(x).to(self.device)
        return torch.relu((self._M @ xt.unsqueeze(1)).squeeze(1)) * self.gain

    def params(self) -> dict:
        return {"gain": self.gain, "touch_scale": self.touch_scale,
                "touch": list(self.touch) if self.touch else None, "obs_dim": self.obs_dim}

    def tables(self) -> dict[str, np.ndarray]:
        return {"matrix_indptr": self.matrix.indptr.astype(np.int64),
                "matrix_indices": self.matrix.indices.astype(np.int64),
                "matrix_values": self.matrix.data.astype(np.float32)}

    @classmethod
    def from_tables(cls, n_neurons: int, params: dict, tables: dict, device: str = "cpu"):
        M = sp.csr_matrix((tables["matrix_values"], tables["matrix_indices"],
                           tables["matrix_indptr"]), shape=(n_neurons, int(params["obs_dim"])))
        touch = tuple(params["touch"]) if params.get("touch") else None
        return cls(n_neurons, matrix=M, gain=params["gain"], touch=touch,
                   touch_scale=params["touch_scale"], device=device)


class ActuatorLayout:
    """The output side of a body controller: named actuators, all continuous in [-1, 1]."""

    def __init__(self, names):
        self.names = list(names)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def n_binary(self) -> int:
        return 0

    @property
    def binary_mask(self) -> np.ndarray:
        return np.zeros(self.n, dtype=bool)

    def to_dict(self) -> dict:
        return {"actuators": list(self.names)}


class ActuatorDecoder:
    """action = tanh(W f + b) over the actuators."""
    kind = "linear"

    def __init__(self, n_features: int, layout: ActuatorLayout, *, W=None, b=None):
        self.layout = layout
        self.n_features = int(n_features)
        self.W = (np.asarray(W, np.float64) if W is not None
                  else np.zeros((layout.n, self.n_features)))
        self.b = np.asarray(b, np.float64) if b is not None else np.zeros(layout.n)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(self.W @ np.asarray(features, np.float64) + self.b).astype(np.float32)

    def params(self) -> dict:
        return {"type": "linear", "output": "tanh"}

    def tables(self) -> dict[str, np.ndarray]:
        return {"W": self.W.astype(np.float32), "b": self.b.astype(np.float32)}

    @classmethod
    def from_tables(cls, layout: ActuatorLayout, params: dict, tables: dict):
        return cls(tables["W"].shape[1], layout, W=tables["W"], b=tables["b"])


class BodyModel(BrainModel):
    kind = "body"

    def __init__(self, brain, *, readout_idx, proprio: ProprioMap, layout: ActuatorLayout,
                 obs_keys, obs_slices: dict, policy=None, config: ModelConfig | None = None,
                 punish_idx=None, neuron_ids=None, neuron_types=None, neuron_superclass=None,
                 neuron_positions=None, positions_known=None):
        super().__init__(brain, readout_idx=readout_idx, policy=policy, config=config,
                         punish_idx=punish_idx, neuron_ids=neuron_ids, neuron_types=neuron_types,
                         neuron_superclass=neuron_superclass, neuron_positions=neuron_positions,
                         positions_known=positions_known)
        self.proprio = proprio
        self.layout = layout
        self.obs_keys = list(obs_keys)
        self.obs_slices = {k: (int(v[0]), int(v[1])) for k, v in obs_slices.items()}
        self._obs = None

    @property
    def n_obs(self) -> int:
        return self.proprio.obs_dim

    @property
    def n_features(self) -> int:
        return len(self.readout_idx) + (self.n_obs if self.config.include_proprio else 0)

    @property
    def n_actions(self) -> int:
        return self.layout.n

    def populations(self) -> dict[str, np.ndarray]:
        named = super().populations()
        named["sensory"] = np.flatnonzero(np.diff(self.proprio.matrix.indptr))
        return named

    def reset(self) -> None:
        super().reset()
        self._obs = None

    def features(self) -> np.ndarray:
        parts = [self.readout_features()]
        if self.config.include_proprio:
            parts.append(np.asarray(self._obs, np.float32))
        return np.concatenate(parts).astype(np.float32)

    def observe(self, obs: np.ndarray, reward: float = 0.0) -> np.ndarray:
        """Drive the sensory neurons with a body observation, run the brain for one
        control interval, return the feature vector."""
        self._obs = np.asarray(obs, dtype=np.float32)
        self._run(self.proprio(self._obs), reward)
        return self.features()

    def step(self, obs: np.ndarray, reward: float = 0.0) -> tuple[np.ndarray, dict]:
        """Observation in, actuator commands in [-1, 1] out (the layout says which)."""
        features = self.observe(obs, reward)
        action = self.act(features)
        info = {"t": self.t, "spikes": self.last_spikes, "action": action.tolist()}
        if self.last_probe is not None:
            info["probe"] = self.last_probe
        return action, info

    def describe(self) -> str:
        lines = self.describe_brain()
        lines.insert(1, f"  proprioception: {self.proprio.n_driven} sensory neurons driven from "
                        f"{self.n_obs} observation entries ({', '.join(self.obs_keys)}), "
                        f"gain {self.proprio.gain:g} mV")
        lines.insert(2, f"  readout: {len(self.readout_idx)} neurons -> {self.n_features} features"
                        + (" (+observation)" if self.config.include_proprio else "")
                        + f"; actuators: {self.layout.n}")
        return "\n".join(lines)
