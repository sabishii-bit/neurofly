"""Command-line plumbing for experiments on a running brain: stimulate, silence, probe.

    parser = add_experiment_args(parser)
    apply_experiments(model, args)          # after the model exists
    log = ProbeLog(model, args.probe_out)   # log.record() after each step; log.save() at the end
"""
from __future__ import annotations

import numpy as np

from neurofly_core.selection import parse_spec


def add_experiment_args(p):
    g = p.add_argument_group("experiments on the brain (see neurofly_core.selection)")
    g.add_argument("--stimulate", action="append", default=[], metavar="SEL:MV",
                   help="extra drive on a selection every step, e.g. type_re=^PPL1:20")
    g.add_argument("--silence", action="append", default=[], metavar="SEL",
                   help="keep a selection from spiking, e.g. superclass=descending_neuron")
    g.add_argument("--probe", default=None, metavar="SEL",
                   help="record a selection's spikes and rates, e.g. name=readout")
    g.add_argument("--probe-out", default=None, help="write the probe to this .npz")
    return p


def apply_experiments(model, args) -> list[str]:
    """Apply --stimulate / --silence / --probe to a Model; returns what was done."""
    done = []
    for spec in args.stimulate:
        sel, mv = parse_spec(spec)
        if mv is None:
            raise SystemExit(f"--stimulate {spec!r} needs a drive: {spec}:20")
        done.append(f"stimulate {sel} with {mv:g} mV: {model.stimulate(sel, mv)} neurons")
    for spec in args.silence:
        sel, _ = parse_spec(spec)
        done.append(f"silence {sel}: {model.silence(sel)} neurons")
    if args.probe:
        sel, _ = parse_spec(args.probe)
        done.append(f"probe {sel}: {model.probe(sel)} neurons")
    return done


class ProbeLog:
    """Collects ``model.last_probe`` after every step and writes one .npz at the end."""

    def __init__(self, model, path: str | None):
        self.model, self.path = model, path
        self.spikes, self.rates, self.t = [], [], []

    def record(self) -> None:
        p = self.model.last_probe
        if p is None:
            return
        self.spikes.append(p["spikes"])
        self.rates.append(p["rates"])
        self.t.append(self.model.t)

    def save(self) -> str | None:
        if not self.path or not self.t:
            return None
        np.savez_compressed(self.path, t=np.asarray(self.t), spikes=np.stack(self.spikes),
                            rates=np.stack(self.rates), indices=self.model.probe_idx,
                            ids=(self.model.neuron_ids[self.model.probe_idx]
                                 if self.model.neuron_ids is not None else np.array([])))
        return self.path
