"""Leaky integrate-and-fire network on a fixed signed connectome.

The model and parameters are those of Shiu et al., Nature 2024:

    tau_m dv/dt      = (v_rest - v) + I_syn + I_ext
    tau_syn dI_syn/dt = -I_syn,     I_syn += w_syn * count  on each presynaptic spike
    spike when v >= v_th, then v = v_reset for t_ref ms

All voltages are in mV and times in ms. ``I_syn`` and ``I_ext`` are expressed
directly in mV of drive (the membrane resistance is folded in). A neuron at
rest needs a sustained drive above ``v_th - v_rest`` (7 mV) to fire.

Two backends compute the synaptic input:

* ``event`` (CPU; the default when numba is installed): only the neurons that
  spiked propagate, through a compiled loop over their outgoing synapses
  (weights stored by presynaptic neuron). A few percent of neurons spike per
  step, so this touches a small fraction of the synapses: the central brain
  (49k neurons, 9.5M synapses) steps in about 1 ms.
* ``torch``: one sparse matrix-vector product per step (weights stored by
  postsynaptic neuron), which streams every synapse every step: about 10 ms
  for the same brain on CPU. This is the backend for CUDA.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @njit(cache=True)
    def _propagate(active, indptr, indices, vals, out):
        """out[post] = sum of weights from the ``active`` presynaptic neurons."""
        out[:] = 0.0
        for a in active:
            for k in range(indptr[a], indptr[a + 1]):
                out[indices[k]] += vals[k]


class LIFBrain:
    def __init__(self, W: sp.spmatrix, *, dt: float = 0.5, tau_m: float = 20.0,
                 tau_syn: float = 5.0, v_rest: float = -52.0, v_reset: float = -52.0,
                 v_th: float = -45.0, t_ref: float = 2.2, w_syn: float = 0.275,
                 gain: float = 1.0, rate_tau: float = 50.0, device: str = "cpu",
                 backend: str = "auto"):
        """
        Args:
            W: (n, n) matrix, W[post, pre] = signed synapse count.
            dt: integration step in ms. Shiu et al. used 0.1; 0.5 is a good
                speed/accuracy trade for closed-loop control.
            w_syn: mV of synaptic drive per synapse per spike.
            gain: global multiplier on all weights (calibration knob).
            rate_tau: time constant (ms) of the exponential firing-rate trace
                exposed as ``self.rate`` (in Hz); this is what readouts use.
            backend: 'event', 'torch' or 'auto' (event on CPU with numba).
        """
        self.device = torch.device(device)
        if backend == "auto":
            backend = "event" if (self.device.type == "cpu" and HAVE_NUMBA) else "torch"
        if backend == "event" and (self.device.type != "cpu" or not HAVE_NUMBA):
            raise ValueError("the event backend needs numba and a CPU device")
        if backend not in ("event", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        as_matrix = sp.csc_matrix if backend == "event" else sp.csr_matrix
        M = as_matrix(W, dtype=np.float32)
        M.sort_indices()
        self.n = M.shape[0]
        self.dt, self.tau_m, self.tau_syn = float(dt), float(tau_m), float(tau_syn)
        self.v_rest, self.v_reset, self.v_th, self.t_ref = v_rest, v_reset, v_th, t_ref
        self.rate_tau = float(rate_tau)

        # by presynaptic neuron (event) or by postsynaptic neuron (torch); ``vals``
        # is the one array of weights, which plasticity edits in place
        self.indptr = torch.from_numpy(M.indptr.astype(np.int64)).to(self.device)
        self.indices = torch.from_numpy(M.indices.astype(np.int64)).to(self.device)
        self.vals = torch.from_numpy(M.data * np.float32(w_syn * gain)).to(self.device)
        self._W = None
        self._out = torch.zeros(self.n, device=self.device)
        if backend == "event":  # numpy views of the same memory, for the compiled kernel
            self._np = (self.indptr.numpy(), self.indices.numpy(), self.vals.numpy(),
                        self._out.numpy())

        n, dev = self.n, self.device
        self.v = torch.full((n,), v_rest, device=dev)
        self.i_syn = torch.zeros(n, device=dev)
        self.ref = torch.zeros(n, device=dev)
        self.spikes = torch.zeros(n, dtype=torch.bool, device=dev)
        self.rate = torch.zeros(n, device=dev)
        self._spike_count = torch.zeros((), device=dev)
        self.t = 0.0
        # experiments: silenced neurons never spike; stimulated ones get extra drive
        self.active = torch.ones(n, dtype=torch.bool, device=dev)
        self.i_extra = torch.zeros(n, device=dev)
        self._manipulated = False

    # --- weights ----------------------------------------------------------------

    @property
    def W(self) -> torch.Tensor:
        """The (n, n) sparse CSR weight matrix (torch backend only)."""
        if self.backend != "torch":
            raise AttributeError("W is only materialised by the torch backend")
        if self._W is None:
            self._W = torch.sparse_csr_tensor(self.indptr, self.indices, self.vals,
                                              size=(self.n, self.n))
        return self._W

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """(pre, post) neuron ids of every synapse, aligned with ``self.vals``."""
        indptr = self.indptr.cpu().numpy()
        indices = self.indices.cpu().numpy()
        outer = np.repeat(np.arange(self.n), np.diff(indptr))
        return (outer, indices) if self.backend == "event" else (indices, outer)

    def mark_weights_changed(self) -> None:
        self._W = None   # the event backend reads ``vals`` directly

    @property
    def n_edges(self) -> int:
        return int(self.vals.numel())

    # --- experiments --------------------------------------------------------------

    def silence(self, idx) -> None:
        """The neurons ``idx`` stop spiking (their inputs still arrive)."""
        self.active[torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)] = False
        self._manipulated = True

    def stimulate(self, idx, mv: float) -> None:
        """Add ``mv`` of drive to the neurons ``idx`` on every step, on top of any input."""
        self.i_extra[torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.device)] += mv
        self._manipulated = True

    def clear_manipulations(self) -> None:
        self.active.fill_(True)
        self.i_extra.zero_()
        self._manipulated = False

    # --- dynamics ---------------------------------------------------------------

    def reset(self) -> None:
        self.v.fill_(self.v_rest)
        self.i_syn.zero_()
        self.ref.zero_()
        self.spikes.zero_()
        self.rate.zero_()
        self._spike_count.zero_()
        self.t = 0.0

    def _synaptic_input(self) -> torch.Tensor:
        if self.backend == "event":
            active = np.flatnonzero(self.spikes.numpy()).astype(np.int64)
            _propagate(active, *self._np)
            return self._out
        s = self.spikes.to(self.vals.dtype)
        return (self.W @ s.unsqueeze(1)).squeeze(1)

    def step(self, i_ext: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one ``dt``. ``i_ext`` is an (n,) drive in mV or None."""
        dt = self.dt
        syn_in = self._synaptic_input()
        self.i_syn.mul_(1.0 - dt / self.tau_syn).add_(syn_in)
        drive = self.i_syn if i_ext is None else self.i_syn + i_ext
        if self._manipulated:
            drive = drive + self.i_extra
        self.v.add_((self.v_rest - self.v + drive) * (dt / self.tau_m))
        self.v.masked_fill_(self.ref > 0, self.v_reset)
        self.ref.sub_(dt).clamp_(min=0.0)
        spikes = self.v >= self.v_th
        if self._manipulated:
            spikes &= self.active
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
        if idx is None:
            return self.rate.detach().cpu().numpy()
        r = self.rate[torch.as_tensor(np.asarray(idx), device=self.device)]
        return r.detach().cpu().numpy()
