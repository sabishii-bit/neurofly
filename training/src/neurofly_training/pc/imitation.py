"""Fit the neuron-to-control map to a recorded human.

Run the recording through the brain, collect the readout features on every
frame, and fit a linear map from features to what the human did on that
frame: logistic regression for keys and buttons, least squares (through
tanh) for mouse motion and scroll. The result is a ``ControlDecoder``.
"""
from __future__ import annotations

import numpy as np
import torch

from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.controls import ControlLayout


def collect_features(env, actions: np.ndarray | None = None, max_steps: int | None = None,
                     progress: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Play a file-backed env to the end with no controls applied. ``X[t]`` is the
    feature vector after the brain has seen frame ``t``; ``Y[t]`` is ``actions[t]``."""
    obs, _ = env.reset()
    X = [obs]
    noop = np.zeros(env.action_space.shape, dtype=np.float32)
    while max_steps is None or len(X) < max_steps:
        obs, _, term, trunc, info = env.step(noop)
        if term or trunc:
            break
        X.append(obs)
        if progress and len(X) % 100 == 0:
            print(f"  {len(X)} frames, {info.get('brain_spikes', 0)} spikes on the last one")
    X = np.stack(X).astype(np.float32)
    Y = None if actions is None else np.asarray(actions, dtype=np.float32)[:len(X)]
    if Y is not None and len(Y) < len(X):
        X = X[:len(Y)]
    return X, Y


def fit_control_decoder(X: np.ndarray, Y: np.ndarray, layout: ControlLayout, *,
                        epochs: int = 300, lr: float = 1e-2, l2: float = 1e-4, seed: int = 0,
                        verbose: bool = False) -> ControlDecoder:
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    Yt = torch.as_tensor(np.asarray(Y, np.float32))
    n, f = Xt.shape
    mask = torch.as_tensor(layout.binary_mask)
    Yb = (Yt[:, mask] > 0).float()
    pos = Yb.mean(0).clamp(1e-3, 1 - 1e-3)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    W = torch.zeros(layout.n, f, requires_grad=True)
    b = torch.zeros(layout.n, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        logits = Xt @ W.T + b
        loss = l2 * (W ** 2).sum()
        if mask.any():
            loss = loss + bce(logits[:, mask], Yb)
        if (~mask).any():
            loss = loss + ((torch.tanh(logits[:, ~mask]) - Yt[:, ~mask]) ** 2).mean()
        loss.backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}")
    dec = ControlDecoder(f, layout)
    dec.W = W.detach().numpy().astype(np.float64)
    dec.b = b.detach().numpy().astype(np.float64)
    return dec


def evaluate(dec: ControlDecoder, X: np.ndarray, Y: np.ndarray) -> dict[str, dict[str, float]]:
    """Per control: precision / recall / F1 and base rate for keys and buttons,
    correlation and mean absolute error for mouse and scroll."""
    P = np.stack([dec(x) for x in X])
    Y = np.asarray(Y, np.float32)
    out = {}
    for j, name in enumerate(dec.layout.names):
        if j < dec.layout.n_binary:
            p, y = P[:, j] > 0, Y[:, j] > 0
            tp, fp, fn = np.sum(p & y), np.sum(p & ~y), np.sum(~p & y)
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            out[name] = {"precision": float(prec), "recall": float(rec),
                         "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
                         "rate": float(y.mean())}
        else:
            p, y = P[:, j], Y[:, j]
            corr = float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 and y.std() > 0 else 0.0
            out[name] = {"corr": corr, "mae": float(np.abs(p - y).mean()),
                         "rate": float(np.abs(y).mean())}
    return out
