"""Dopamine-gated Hebbian plasticity on a subset of existing synapses.

A three-factor rule in the spirit of the mushroom-body learning literature:
every synapse in the chosen subset keeps an eligibility trace that grows when
the presynaptic neuron was recently active and the postsynaptic neuron
spikes. A dopamine signal (here: reward) turns the eligibility into a weight
change. Weights keep their sign and are bounded to a multiple of their
anatomical value.
"""
from __future__ import annotations

import numpy as np
import torch

from neurofly_core.brain.lif import LIFBrain


class DopamineHebbian:
    def __init__(self, brain: LIFBrain, pre_idx, post_idx, *, lr: float = 1e-3,
                 tau_pre: float = 20.0, tau_elig: float = 200.0, w_max_scale: float = 3.0):
        self.brain = brain
        self.lr, self.tau_pre, self.tau_elig = lr, tau_pre, tau_elig
        n = brain.n
        pre, post = brain.edges()
        pre_mask = np.zeros(n, dtype=bool)
        pre_mask[np.asarray(pre_idx)] = True
        post_mask = np.zeros(n, dtype=bool)
        post_mask[np.asarray(post_idx)] = True
        e = np.flatnonzero(pre_mask[pre] & post_mask[post])
        dev = brain.device
        self.e = torch.from_numpy(e).to(dev)
        self.row_e = torch.from_numpy(post[e]).to(dev)   # postsynaptic neuron of each edge
        self.col_e = torch.from_numpy(pre[e]).to(dev)    # presynaptic neuron of each edge
        w0 = brain.vals[self.e].clone()
        self.sign = torch.where(w0 < 0, -1.0, 1.0)
        self.w_max = w0.abs() * w_max_scale + 1e-6
        self.elig = torch.zeros(len(e), device=dev)
        self.pre_trace = torch.zeros(n, device=dev)

    @property
    def n_edges(self) -> int:
        return int(self.e.numel())

    def step(self, dopamine: float = 0.0) -> None:
        """Call once per brain step, after ``brain.step``."""
        b, dt = self.brain, self.brain.dt
        spikes = b.spikes.to(self.pre_trace.dtype)
        self.pre_trace.mul_(1.0 - dt / self.tau_pre).add_(spikes)
        self.elig.mul_(1.0 - dt / self.tau_elig)
        self.elig.add_(self.pre_trace[self.col_e] * spikes[self.row_e])
        if dopamine != 0.0:
            w = b.vals[self.e]
            mag = (w * self.sign) + self.lr * dopamine * self.elig
            mag = torch.minimum(mag.clamp(min=0.0), self.w_max)
            b.vals[self.e] = mag * self.sign
            b.mark_weights_changed()
