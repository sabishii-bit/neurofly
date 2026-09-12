"""Find the gains that give the brain sparse, not silent, not runaway, activity.

Sweeps one gain (``brain_gain``, ``retina_gain`` or ``audio_gain``) over a
geometric grid, runs the model on a few frames at each value, and reports the
fraction of readout neurons that fire, the fraction of all neurons that fire,
and the mean readout rate. The recommendation is the gain whose readout
activity is closest to ``target``.
"""
from __future__ import annotations

import numpy as np

from neurofly_core.model import Model


def activity(model: Model, frames, chunks=None, warm: int = 2) -> dict:
    """Run the model over ``frames`` (after ``warm`` warm-up observations) and summarise."""
    model.reset()
    rates_all, spikes = [], []
    for t, frame in enumerate(frames):
        chunk = chunks[t] if chunks is not None else None
        model.observe(frame, chunk)
        if t >= warm:
            rates_all.append(model.brain.rates())
            spikes.append(model.last_spikes)
    r = np.mean(rates_all, axis=0) if rates_all else np.zeros(model.brain.n)
    read = r[model.readout_idx]
    return {"readout_active": float(np.mean(read > 1.0)), "all_active": float(np.mean(r > 1.0)),
            "readout_rate": float(read.mean()),
            "spikes_per_step": float(np.mean(spikes) if spikes else 0)}


def sweep(build, param: str, frames, chunks=None, *, grid=None, target: float = 0.2) -> dict:
    """``build(**{param: value})`` must return a fresh Model. Returns the table and the pick."""
    grid = list(grid) if grid is not None else [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]
    rows = []
    for g in grid:
        model = build(**{param: g})
        row = {param: float(g), **activity(model, frames, chunks)}
        rows.append(row)
    best = min(rows, key=lambda r: abs(r["readout_active"] - target))
    return {"param": param, "target": target, "rows": rows, "pick": best}


def format_sweep(res: dict, base: float | None = None) -> str:
    p = res["param"]
    lines = [f"{p:>12s}  readout active   all active   readout rate   spikes/step"]
    for r in res["rows"]:
        mark = " <-" if r is res["pick"] else ""
        lines.append(f"{r[p]:12.3g}  {r['readout_active']:13.1%}  {r['all_active']:10.1%}  "
                     f"{r['readout_rate']:10.1f} Hz  {r['spikes_per_step']:11.0f}{mark}")
    pick = res["pick"][p]
    active = res["pick"]["readout_active"]
    lines.append(f"pick: --{p.replace('_', '-')} {pick:g} "
                 f"(readout active {active:.1%}, target {res['target']:.0%})")
    return "\n".join(lines)
