"""Gradient training through the brain, with surrogate gradients.

The readout-only trainers treat the brain as a fixed feature extractor. This
module backpropagates through the spiking dynamics instead: the spike is a
step function in the forward pass and a fast-sigmoid in the backward pass
(the standard surrogate-gradient trick of spiking-network training), so a loss
on the readout can move parameters upstream of the neurons.

What is trained: the retina's input map (the projection matrix in projection
mode, a per-neuron gain in hex mode) and a linear head from features to
controls. The synapses of the connectome stay fixed. Truncated backpropagation
runs within each frame's brain sub-steps; the brain state carries across frames
without gradient.

    history = train_surrogate(model, frames, chunks, actions, epochs=5)
    # model.retina and model.policy are replaced by the trained ones
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.encode.vision import RetinaEncoder
from neurofly_core.model import Model

BETA = 5.0          # slope of the surrogate
V_SCALE = 2.0       # mV over which the surrogate is wide


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward, 1 / (1 + beta |x|)^2 backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + BETA * x.abs()) ** 2


spike_fn = FastSigmoidSpike.apply


class DifferentiableLIF:
    """The same dynamics as ``LIFBrain.step``, in autograd-friendly form, with the
    synapses as a fixed sparse COO matrix (post x pre)."""

    def __init__(self, brain, device="cpu"):
        pre, post = brain.edges()
        vals = brain.vals.cpu().numpy()
        M = sp.coo_matrix((vals, (post, pre)), shape=(brain.n, brain.n))
        idx = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
        self.W = torch.sparse_coo_tensor(idx, torch.from_numpy(M.data.astype(np.float32)),
                                         (brain.n, brain.n)).coalesce().to(device)
        self.n = brain.n
        self.device = torch.device(device)
        self.dt, self.tau_m, self.tau_syn = brain.dt, brain.tau_m, brain.tau_syn
        self.v_rest, self.v_reset = brain.v_rest, brain.v_reset
        self.v_th, self.t_ref = brain.v_th, brain.t_ref
        self.rate_tau = brain.rate_tau
        self.active = brain.active.to(device).to(torch.float32)
        self.i_extra = brain.i_extra.to(device)

    def initial_state(self) -> dict:
        z = torch.zeros(self.n, device=self.device)
        return {"v": torch.full((self.n,), self.v_rest, device=self.device), "i_syn": z.clone(),
                "ref": z.clone(), "s": z.clone(), "rate": z.clone()}

    def run(self, drive: torch.Tensor, n_steps: int, state: dict | None = None):
        """``n_steps`` of constant ``drive`` (n,) from ``state``; returns (rate, new state)."""
        st = state or self.initial_state()
        v, i_syn, ref, s, rate = st["v"], st["i_syn"], st["ref"], st["s"], st["rate"]
        dt = self.dt
        for _ in range(n_steps):
            syn_in = torch.sparse.mm(self.W, s.unsqueeze(1)).squeeze(1)
            i_syn = i_syn * (1.0 - dt / self.tau_syn) + syn_in
            v = v + (self.v_rest - v + i_syn + drive + self.i_extra) * (dt / self.tau_m)
            refractory = (ref > 0).to(v.dtype)
            v = v * (1 - refractory) + self.v_reset * refractory
            ref = torch.clamp(ref - dt, min=0.0)
            s = spike_fn((v - self.v_th) / V_SCALE) * self.active
            reset = s.detach()
            v = v * (1 - reset) + self.v_reset * reset
            ref = ref * (1 - reset) + self.t_ref * reset
            rate = rate * (1.0 - dt / self.rate_tau) + s * (1000.0 / self.rate_tau)
        new_state = {"v": v.detach(), "i_syn": i_syn.detach(), "ref": ref.detach(),
                     "s": s.detach(), "rate": rate.detach()}
        return rate, new_state


class TrainableRetina(nn.Module):
    """The retina's input map as parameters: the projection matrix, or per-neuron gains."""

    def __init__(self, retina: RetinaEncoder, device="cpu"):
        super().__init__()
        self.mode, self.n, self.grid = retina.mode, retina.n, retina.grid
        self.gain, self.temporal = retina.gain, retina.temporal
        self.device = torch.device(device)
        if retina.mode == "hex":
            self.indices = torch.from_numpy(retina.indices).to(device)
            self.pixels = torch.from_numpy(retina.pixels).to(device)
            self.on_mask = torch.from_numpy(retina.on).to(device)
            self.gains = nn.Parameter(torch.full((len(retina.indices),), float(retina.gain),
                                                 device=device))
        else:
            self.targets = torch.from_numpy(retina.targets).to(device)
            M = retina.matrix.toarray()[retina.targets]           # (n_targets, n_pix), small
            self.M = nn.Parameter(torch.from_numpy(M.astype(np.float32)).to(device))

    def forward(self, on: torch.Tensor, off: torch.Tensor) -> torch.Tensor:
        drive = torch.zeros(self.n, device=self.device)
        if self.mode == "hex":
            s = torch.where(self.on_mask, on[self.pixels], off[self.pixels]) * self.gains
            return drive.index_put((self.indices,), s)
        y = self.M @ on
        z = (y - y.mean()) / (y.std() + 1e-6)
        return drive.index_put((self.targets,), torch.relu(z + 0.5) * self.gain)

    def export(self, retina: RetinaEncoder) -> RetinaEncoder:
        """A RetinaEncoder with the trained tables."""
        if self.mode == "hex":
            return RetinaEncoder(self.n, mode="hex", grid=self.grid, gain=retina.gain,
                                 temporal=self.temporal, indices=retina.indices,
                                 pixels=retina.pixels, on=retina.on, device=str(retina.device)) \
                ._with_gains(self.gains.detach().cpu().numpy())
        M = np.zeros((self.n, int(np.prod(self.grid))), np.float32)
        M[retina.targets] = np.clip(self.M.detach().cpu().numpy(), 0.0, None)
        return RetinaEncoder(self.n, mode="projection", grid=self.grid, gain=retina.gain,
                             temporal=self.temporal, matrix=sp.csr_matrix(M),
                             targets=retina.targets, device=str(retina.device))


def _with_gains(self: RetinaEncoder, gains: np.ndarray) -> RetinaEncoder:
    """Hex mode has one gain; fold per-neuron gains in by scaling with a per-neuron table
    is not part of the artifact, so bake them into ON/OFF polarity and a shared gain: the
    closest representable encoder keeps the mean gain and drops neurons whose gain fell
    to zero."""
    keep = gains > 1e-3
    self.indices, self.pixels, self.on = self.indices[keep], self.pixels[keep], self.on[keep]
    self._idx_t = torch.from_numpy(self.indices).to(self.device)
    self.n_driven = int(len(self.indices))
    self.gain = float(gains[keep].mean()) if keep.any() else self.gain
    return self


RetinaEncoder._with_gains = _with_gains


def _losses(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor):
    loss = torch.zeros((), device=logits.device)
    if mask.any():
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            logits[mask], (y[mask] > 0).float(), pos_weight=pos_weight)
    if (~mask).any():
        loss = loss + ((torch.tanh(logits[~mask]) - y[~mask]) ** 2).mean()
    return loss


def train_surrogate(model: Model, frames, chunks, actions, *, epochs: int = 5, lr: float = 1e-2,
                    window: int = 8, l2: float = 1e-4, device: str = "cpu",
                    verbose: bool = False, detections=None, odours=None) -> dict:
    """Train the retina's input map and a linear head end to end on (frames, actions).

    ``frames``: list of RGB uint8 arrays; ``chunks``: list of audio chunks or None;
    ``actions``: (T, n_actions) targets in the layout's action space; ``detections``: a
    list of per-frame detections when the model has a detection encoder; ``odours``: a
    list of per-frame odour vectors when it has an olfaction encoder. The model's
    retina and policy are replaced by the trained ones. Returns a history dict.
    """
    layout = model.layout
    T = min(len(frames), len(actions))
    actions = torch.as_tensor(np.asarray(actions[:T], np.float32), device=device)
    mask = torch.as_tensor(layout.binary_mask, device=device)
    pos = actions[:, mask].gt(0).float().mean(0).clamp(1e-3, 1 - 1e-3) if mask.any() else None
    pos_weight = ((1 - pos) / pos) if pos is not None else None

    lif = DifferentiableLIF(model.brain, device=device)
    retina = TrainableRetina(model.retina, device=device)
    n_read = len(model.readout_idx)
    readout = torch.from_numpy(model.readout_idx).to(device)
    n_extra = model.n_features - n_read
    head = nn.Linear(model.n_features, layout.n).to(device)
    with torch.no_grad():
        if isinstance(model.policy, ControlDecoder):
            head.weight.copy_(torch.as_tensor(model.policy.W, dtype=torch.float32))
            head.bias.copy_(torch.as_tensor(model.policy.b, dtype=torch.float32))
        else:
            head.weight.zero_()
            head.bias.zero_()
    opt = torch.optim.Adam(list(retina.parameters()) + list(head.parameters()), lr=lr)

    # constants per frame: the on/off grids, the audition drive, the extra features
    model.retina.reset()
    if model.audition is not None:
        model.audition.reset()
    if model.detection is not None:
        model.detection.reset()
    if model.olfaction is not None:
        model.olfaction.reset()
    signals, aud, extras = [], [], []
    for t in range(T):
        on, off = model.retina.signals(frames[t])
        signals.append((torch.from_numpy(on).to(device), torch.from_numpy(off).to(device)))
        const = (model.audition(chunks[t] if chunks else None).to(device).detach()
                 if model.audition is not None else None)
        if model.detection is not None:
            d = model.detection(detections[t] if detections else None).to(device).detach()
            const = d if const is None else const + d
        if model.olfaction is not None:
            o = model.olfaction(odours[t] if odours else None).to(device).detach()
            const = o if const is None else const + o
        aud.append(const)
        if n_extra:
            model._frame, model._chunk = frames[t], (chunks[t] if chunks else None)
            extras.append(torch.from_numpy(model.features()[n_read:]).to(device))
    history = {"loss": []}
    scale = model.config.readout_scale
    for ep in range(epochs):
        state, total, pending = None, 0.0, []
        opt.zero_grad()
        for t in range(T):
            on, off = signals[t]
            drive = retina(on, off)
            if aud[t] is not None:
                drive = drive + aud[t]
            n_steps = model.substeps + (model.warmup_steps if t == 0 else 0)
            rate, state = lif.run(drive, n_steps, state)
            feats = rate[readout] * scale
            if n_extra:
                feats = torch.cat([feats, extras[t]])
            logits = head(feats)
            loss = _losses(logits, actions[t], mask, pos_weight) + l2 * (head.weight ** 2).sum()
            pending.append(loss)
            if len(pending) == window or t == T - 1:
                batch = torch.stack(pending).mean()
                batch.backward()
                opt.step()
                opt.zero_grad()
                total += batch.item() * len(pending)
                pending = []
        history["loss"].append(total / T)
        if verbose:
            print(f"  epoch {ep:3d}  loss {total / T:.4f}")
    model.retina = retina.export(model.retina)
    dec = ControlDecoder(model.n_features, layout)
    dec.W = head.weight.detach().cpu().numpy().astype(np.float64)
    dec.b = head.bias.detach().cpu().numpy().astype(np.float64)
    model.policy = dec
    return history
