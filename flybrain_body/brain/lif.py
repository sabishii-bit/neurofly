"""Leaky integrate-and-fire network on a fixed signed connectome.

This is the model every "fly brain plays X" project uses, after Shiu et al.,
Nature 2024 (github.com/philshiu/Drosophila_brain_model):

    tau_m dv/dt      = (v_rest - v) + I_syn + I_ext
    tau_syn dI_syn/dt = -I_syn,     I_syn += w_syn * count  on each presynaptic spike
    spike when v >= v_th, then v = v_reset for t_ref ms

All voltages are in mV and times in ms. ``I_syn`` and ``I_ext`` are expressed
directly in mV of drive (the membrane resistance is folded in). A neuron at
rest needs a sustained drive above ``v_th - v_rest`` (7 mV) to fire.

The weight matrix is a torch sparse CSR tensor, so one step costs one sparse
matrix-vector product; on CPU the full 165k-neuron / 25.6M-edge CNS steps in
tens of milliseconds, the nerve-cord subset in a few.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class LIFBrain:
    def __init__(self, W: sp.spmatrix, *, dt: float = 0.5, tau_m: float = 20.0,
                 tau_syn: float = 5.0, v_rest: float = -52.0, v_reset: float = -52.0,
                 v_th: float = -45.0, t_ref: float = 2.2, w_syn: float = 0.275,
                 gain: float = 1.0, rate_tau: float = 50.0, device: str = "cpu"):
        """
        Args:
            W: (n, n) matrix, W[post, pre] = signed synapse count.
            dt: integration step in ms. Shiu et al. used 0.1; 0.5 is a good
                speed/accuracy trade for closed-loop control.
            w_syn: mV of synaptic drive per synapse per spike.
            gain: global multiplier on all weights (calibration knob).
            rate_tau: time constant (ms) of the exponential firing-rate trace
                exposed as ``self.rate`` (in Hz); this is what readouts use.
        """
        W = sp.csr_matrix(W, dtype=np.float32)
        W.sort_indices()
        self.n = W.shape[0]
        self.dt, self.tau_m, self.tau_syn = float(dt), float(tau_m), float(tau_syn)
        self.v_rest, self.v_reset, self.v_th, self.t_ref = v_rest, v_reset, v_th, t_ref
        self.rate_tau = float(rate_tau)
        self.device = torch.device(device)

        self.crow = torch.from_numpy(W.indptr.astype(np.int64)).to(self.device)
        self.col = torch.from_numpy(W.indices.astype(np.int64)).to(self.device)
        self.vals = torch.from_numpy(W.data * np.float32(w_syn * gain)).to(self.device)
        self._W = None

        n, dev = self.n, self.device
        self.v = torch.full((n,), v_rest, device=dev)
        self.i_syn = torch.zeros(n, device=dev)
        self.ref = torch.zeros(n, device=dev)
        self.spikes = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rate = torch.zeros(n, device=dev)
        self._spike_count = torch.zeros((), device=dev)
        self.t = 0.0

    # --- weights ----------------------------------------------------------------

    @property
    def W(self) -> torch.Tensor:
        if self._W is None:
            self._W = torch.sparse_csr_tensor(self.crow, self.col, self.vals,
                                              size=(self.n, self.n))
        return self._W

    def mark_weights_changed(self) -> None:
        self._W = None

    @property
    def n_edges(self) -> int:
        return int(self.vals.numel())

    # --- dynamics ---------------------------------------------------------------

    def reset(self) -> None:
        self.v.fill_(self.v_rest)
        self.i_syn.zero_()
        self.ref.zero_()
        self.spikes.zero_()
        self.rate.zero_()
        self._spike_count.zero_()
        self.t = 0.0

    def step(self, i_ext: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one ``dt``. ``i_ext`` is an (n,) drive in mV or None."""
        dt = self.dt
        s = self.spikes.to(self.vals.dtype)
        syn_in = (self.W @ s.unsqueeze(1)).squeeze(1)
        self.i_syn.mul_(1.0 - dt / self.tau_syn).add_(syn_in)
        drive = self.i_syn if i_ext is None else self.i_syn + i_ext
        self.v.add_((self.v_rest - self.v + drive) * (dt / self.tau_m))
        self.v.masked_fill_(self.ref > 0, self.v_reset)
        self.ref.sub_(dt).clamp_(min=0.0)
        spikes = self.v >= self.v_th
        self.v.masked_fill_(spikes, self.v_reset)
        self.ref.masked_fill_(spikes, self.t_ref)
        self.spikes = spikes
        self.rate.mul_(1.0 - dt / self.rate_tau).add_(spikes.to(self.rate.dtype),
                                                      alpha=1000.0 / self.rate_tau)
        self._spike_count += spikes.sum()
        self.t += dt
        return spikes

    def run(self, n_steps: int, i_ext: torch.Tensor | None = None) -> torch.Tensor:
        """Run ``n_steps`` with constant drive; returns per-neuron spike counts."""
        counts = torch.zeros(self.n, device=self.device)
        for _ in range(n_steps):
            counts += self.step(i_ext)
        return counts

    # --- helpers ----------------------------------------------------------------

    @property
    def total_spikes(self) -> int:
        return int(self._spike_count.item())

    def drive(self, idx, amount: float) -> torch.Tensor:
        """An (n,) external drive vector with ``amount`` mV on neurons ``idx``."""
        i = torch.zeros(self.n, device=self.device)
        i[torch.as_tensor(np.asarray(idx), device=self.device)] = amount
        return i

    def rates(self, idx=None) -> np.ndarray:
        r = self.rate if idx is None else self.rate[torch.as_tensor(np.asarray(idx), device=self.device)]
        return r.detach().cpu().numpy()
