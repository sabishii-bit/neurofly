"""An inverse dynamics model: what was pressed, from the frames around each moment.

Given a labelled recording (frames and the human's actions) it learns to
predict the action at frame t from the luminance grids of frames t-k .. t+k.
Seeing the future makes this far easier than acting, so a little labelled
footage suffices; the trained model then labels footage that has no input
log, and ``imitate`` treats those labels like recorded ones. The model sits
outside the brain: it only recovers what a human did.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from neurofly_core.controls import ControlLayout
from neurofly_core.encode.vision import luminance


def frame_features(frames, grid=(12, 16)) -> np.ndarray:
    """(T, rows*cols) luminance grids."""
    return np.stack([luminance(f, grid).ravel() for f in frames]).astype(np.float32)


def windowed(feats: np.ndarray, k: int) -> np.ndarray:
    """(T, (2k+1)*d): each row is frames t-k..t+k, edges padded by repetition."""
    T = len(feats)
    pad = np.concatenate([np.repeat(feats[:1], k, 0), feats, np.repeat(feats[-1:], k, 0)])
    return np.stack([pad[t:t + 2 * k + 1].ravel() for t in range(T)])


class IDM(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class InverseDynamics:
    def __init__(self, layout: ControlLayout, grid=(12, 16), k: int = 2, hidden: int = 256):
        self.layout, self.grid, self.k, self.hidden = layout, tuple(grid), int(k), int(hidden)
        n_in = (2 * self.k + 1) * int(np.prod(self.grid))
        self.model = IDM(n_in, layout.n, hidden)

    def inputs(self, frames) -> torch.Tensor:
        return torch.as_tensor(windowed(frame_features(frames, self.grid), self.k))

    def fit(self, frames, actions, *, epochs: int = 200, lr: float = 1e-3, batch: int = 256,
            seed: int = 0, verbose: bool = False) -> list[float]:
        torch.manual_seed(seed)
        X = self.inputs(frames)
        Y = torch.as_tensor(np.asarray(actions, np.float32))[:len(X)]
        mask = torch.as_tensor(self.layout.binary_mask)
        pos = Y[:, mask].gt(0).float().mean(0).clamp(1e-3, 1 - 1e-3) if mask.any() else None
        bce = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos) if pos is not None else None
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        history = []
        n = len(X)
        for ep in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                logits = self.model(X[idx])
                loss = torch.zeros(())
                if bce is not None:
                    loss = loss + bce(logits[:, mask], (Y[idx][:, mask] > 0).float())
                if (~mask).any():
                    loss = loss + ((torch.tanh(logits[:, ~mask]) - Y[idx][:, ~mask]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            history.append(total / n)
            if verbose and (ep % 50 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  loss {total / n:.4f}")
        return history

    @torch.no_grad()
    def predict(self, frames) -> np.ndarray:
        """(T, n_actions) actions in the layout's space, keys as +1 / -1."""
        logits = self.model(self.inputs(frames)).numpy()
        mask = self.layout.binary_mask
        out = np.tanh(logits).astype(np.float32)
        out[:, mask] = np.where(logits[:, mask] > 0, 1.0, -1.0)
        return out

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "idm.pt"))
        with open(os.path.join(path, "idm.json"), "w") as f:
            json.dump({"layout": self.layout.to_dict(), "grid": list(self.grid), "k": self.k,
                       "hidden": self.hidden}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "InverseDynamics":
        with open(os.path.join(path, "idm.json")) as f:
            meta = json.load(f)
        idm = cls(ControlLayout.from_dict(meta["layout"]), grid=meta["grid"], k=meta["k"],
                  hidden=meta["hidden"])
        idm.model.load_state_dict(torch.load(os.path.join(path, "idm.pt"), map_location="cpu"))
        idm.model.eval()
        return idm
