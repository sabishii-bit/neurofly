"""Measure how fast the spiking brain steps, to choose a subset, dt and backend.

    neurofly bench --subset vnc --dt 0.5
    neurofly bench --subset central --backend torch     # compare with event
    neurofly bench --subset full --device cuda
"""
import argparse
import time

import torch

from neurofly_core.brain.lif import LIFBrain
from neurofly_training.data.connectome import SUBSETS
from neurofly_training.data.populations import Populations
from neurofly_training.envs import DEFAULT_DATA_DIR, load_connectome


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--brain", default="malecns", choices=["malecns", "synthetic"])
    p.add_argument("--subset", default="vnc", choices=sorted(SUBSETS))
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default="auto", choices=["auto", "event", "torch"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--drive-mv", type=float, default=12.0,
                   help="constant drive on all leg proprioceptive neurons (or, without a nerve "
                        "cord, on the visual projection neurons)")
    a = p.parse_args()

    cx = load_connectome(a.brain, subset=a.subset, data_dir=a.data_dir)
    pops = Populations(cx)
    brain = LIFBrain(cx.W, dt=a.dt, device=a.device, backend=a.backend)
    print(f"{cx.name}: {cx.n:,} neurons, {cx.n_edges:,} edges on {a.device}, "
          f"{brain.backend} backend")

    seeds = [i for leg in pops.leg_proprio.values() for i in leg] or list(pops.visual_projection)
    drive = brain.drive(seeds, a.drive_mv)
    brain.run(20, drive)  # warm-up (and compile, for the event backend)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    counts = brain.run(a.steps, drive)
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / a.steps
    substeps = 2.0 / a.dt  # the body's 2 ms control step
    print(f"{per_step * 1000:.2f} ms per brain step  ->  "
          f"{1 / (per_step * substeps):.1f} body control steps/s "
          f"(x{1 / (per_step * substeps) / 500:.3f} real time); "
          f"{1 / (per_step * 10 / a.dt):.1f} PC steps/s at 10 ms of brain per step")
    active = int((counts > 0).sum())
    rate = counts[counts > 0].mean() / (a.steps * a.dt) * 1000
    print(f"{active:,} neurons spiked in {a.steps * a.dt:.0f} ms of sim; "
          f"mean rate of active neurons {rate:.1f} Hz")


if __name__ == "__main__":
    main()
