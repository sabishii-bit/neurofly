"""Body observation -> external drive on sensory neurons.

Each leg's proprioceptive neurons get a fixed random sparse projection of that
leg's joint angles and velocities (rectified, so more deviation = more drive),
its tactile neurons get that leg's ground-contact signal, and the head's
wind/gravity and haltere neurons get the gyro and accelerometer. The
projection is a random reservoir-style map, not a learned one; training
happens downstream of the brain. Everything is a single sparse matrix, so
encoding costs one sparse mat-vec per control step.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from flybrain_body.data.connectome import LEGS
from flybrain_body.interface.populations import Populations

TOUCH_ORDER = LEGS  # flybody's 6 touch sensors are ordered T1L, T1R, T2L, T2R, T3L, T3R


class ProprioEncoder:
    def __init__(self, pops: Populations, n_neurons: int, obs_slices: dict[str, slice],
                 joint_names: list[str] | None, *, gain: float = 12.0,
                 vel_scale: float = 0.02, touch_scale: float = 1.0,
                 inputs_per_neuron: int = 3, seed: int = 0, device: str = "cpu"):
        self.gain, self.touch_scale = gain, touch_scale
        self.device = torch.device(device)
        self.obs_dim = max(s.stop for s in obs_slices.values())
        self.touch_slice = obs_slices.get("walker/touch")
        rng = np.random.default_rng(seed)
        rows, cols, vals = [], [], []

        def connect(neurons, features, k):
            for nrn in neurons:
                pick = rng.choice(len(features), size=min(k, len(features)), replace=False)
                for p in pick:
                    f, scale = features[p]
                    rows.append(nrn)
                    cols.append(f)
                    vals.append(rng.normal() * scale)

        jp, jv = obs_slices["walker/joints_pos"], obs_slices["walker/joints_vel"]
        n_joints = jp.stop - jp.start
        for li, (t, side) in enumerate(LEGS):
            if joint_names is not None and len(joint_names) == n_joints:
                leg_joints = [i for i, nm in enumerate(joint_names) if f"{t}_{side}" in nm
                              or f"{t}_{'left' if side == 'L' else 'right'}" in nm]
            else:
                leg_joints = []
            if not leg_joints:  # fall back to an even split of the joint vector
                chunk = n_joints // 6
                leg_joints = list(range(li * chunk, (li + 1) * chunk))
            feats = [(jp.start + i, 1.0) for i in leg_joints] + \
                    [(jv.start + i, vel_scale) for i in leg_joints]
            connect(pops.leg_proprio[(t, side)], feats, inputs_per_neuron)
            if self.touch_slice is not None:
                tf = self.touch_slice.start + TOUCH_ORDER.index((t, side))
                for nrn in pops.leg_tactile[(t, side)]:
                    rows.append(nrn)
                    cols.append(tf)
                    vals.append(abs(rng.normal(1.0, 0.3)))

        head_feats = []
        for key, scale in (("walker/gyro", 0.05), ("walker/accelerometer", 0.002)):
            if key in obs_slices:
                s = obs_slices[key]
                head_feats += [(i, scale) for i in range(s.start, s.stop)]
        if head_feats:
            connect(pops.wind_gravity, head_feats, 2)
            connect(pops.haltere, head_feats, 2)

        M = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                          shape=(n_neurons, self.obs_dim), dtype=np.float32)
        M.sum_duplicates()
        M.sort_indices()
        self.n_inputs = int(M.nnz)
        self.n_driven = int(np.count_nonzero(np.diff(M.indptr)))
        self._M = torch.sparse_csr_tensor(
            torch.from_numpy(M.indptr.astype(np.int64)),
            torch.from_numpy(M.indices.astype(np.int64)),
            torch.from_numpy(M.data), size=M.shape).to(self.device)

    def __call__(self, obs: np.ndarray) -> torch.Tensor:
        x = np.asarray(obs, dtype=np.float32).copy()
        if self.touch_slice is not None:
            x[self.touch_slice] = np.tanh(x[self.touch_slice] * self.touch_scale)
        xt = torch.from_numpy(x).to(self.device)
        drive = (self._M @ xt.unsqueeze(1)).squeeze(1)
        return torch.relu(drive) * self.gain
