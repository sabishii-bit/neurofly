"""Measure how fast the spiking brain steps, to choose a subset and dt.

    python scripts/bench_brain.py --subset vnc --dt 0.5
    python scripts/bench_brain.py --subset full --device cuda
"""
import argparse
import time

import torch

from flybrain_body.data.connectome import SUBSETS
from flybrain_body.envs import DEFAULT_DATA_DIR, load_connectome
from flybrain_body.brain.lif import LIFBrain
from flybrain_body.interface.populations import Populations

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--brain", default="malecns", choices=["malecns", "synthetic"])
    p.add_argument("--subset", default="vnc", choices=sorted(SUBSETS))
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--drive-mv", type=float, default=12.0,
                   help="constant drive on all leg proprioceptive neurons")
    a = p.parse_args()

    cx = load_connectome(a.brain, subset=a.subset, data_dir=a.data_dir)
    pops = Populations(cx)
    brain = LIFBrain(cx.W, dt=a.dt, device=a.device)
    print(f"{cx.name}: {cx.n:,} neurons, {cx.n_edges:,} edges on {a.device}")

    seeds = [i for leg in pops.leg_proprio.values() for i in leg]
    drive = brain.drive(seeds, a.drive_mv)
    brain.run(20, drive)  # warm-up
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    counts = brain.run(a.steps, drive)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt_wall = time.time() - t0
    per_step = dt_wall / a.steps
    substeps = 2.0 / a.dt  # flybody's 2 ms control step
    print(f"{per_step * 1000:.2f} ms per brain step  ->  "
          f"{1 / (per_step * substeps):.1f} control steps/s "
          f"(x{1 / (per_step * substeps) / 500:.3f} real time)")
    active = int((counts > 0).sum())
    print(f"{active:,} neurons spiked in {a.steps * a.dt:.0f} ms of sim; "
          f"mean rate of active neurons {counts[counts > 0].mean() / (a.steps * a.dt) * 1000:.1f} Hz")
