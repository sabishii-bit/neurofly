"""Train an inverse dynamics model on your labelled recordings.

    neurofly idm data/recordings/run1 data/recordings/run2 --run-name idm1
    neurofly label footage/*.mp4 --idm runs/idm1 --out data/recordings/labelled
    neurofly imitate data/recordings/labelled/* --run-name from_footage

The model predicts the action at each frame from the frames around it (it sees the
future, which makes the problem easy), so a few minutes of your own play are enough to
label hours of footage that has no input log.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurofly_core.controls import ControlLayout
from neurofly_core.io.video import VideoFile
from neurofly_training.idm import InverseDynamics
from neurofly_training.pc.imitation import evaluate


def load_recording(rec: str, max_frames=None):
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    actions = np.load(os.path.join(rec, "actions.npy"))
    video = VideoFile(os.path.join(rec, "video.mp4"))
    frames = []
    f = video.reset()
    while f is not None and (max_frames is None or len(frames) < max_frames):
        frames.append(f)
        f = video.read()
    video.close()
    n = min(len(frames), len(actions))
    return ControlLayout.from_dict(meta["layout"]), frames[:n], actions[:n]


class _Scorer:
    """evaluate() wants something with .layout and __call__ per row."""

    def __init__(self, idm, preds):
        self.layout, self._preds, self._i = idm.layout, preds, 0

    def __call__(self, x):
        p = self._preds[self._i]
        self._i += 1
        return p


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+")
    p.add_argument("--context", type=int, default=2, help="frames before and after (k)")
    p.add_argument("--grid", default="12,16", help="luminance grid rows,cols")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    grid = tuple(int(v) for v in args.grid.split(","))
    run_dir = os.path.join("runs", args.run_name or f"idm_{time.strftime('%Y%m%d-%H%M%S')}")

    layout, tr_f, tr_a, te_f, te_a = None, [], [], [], []
    for rec in args.recordings:
        lay, frames, actions = load_recording(rec, args.max_frames)
        if layout is None:
            layout = lay
        elif lay.names != layout.names:
            raise SystemExit(f"{rec} has controls {lay.names}, expected {layout.names}")
        n_te = int(len(frames) * args.holdout)
        n_tr = len(frames) - n_te
        tr_f += frames[:n_tr]
        tr_a.append(actions[:n_tr])
        te_f += frames[n_tr:]
        te_a.append(actions[n_tr:])
    tr_a = np.concatenate(tr_a)
    idm = InverseDynamics(layout, grid=grid, k=args.context, hidden=args.hidden)
    print(f"training on {len(tr_f)} frames, context {args.context}, grid {grid} ...")
    idm.fit(tr_f, tr_a, epochs=args.epochs, lr=args.lr, seed=args.seed, verbose=True)
    idm.save(run_dir)

    def report(name, frames, actions):
        if not frames:
            return
        actions = np.concatenate(actions)
        preds = idm.predict(frames)
        print(f"{name}:")
        for k, m in evaluate(_Scorer(idm, preds), np.zeros((len(preds), 1)), actions).items():
            if "f1" in m:
                print(f"  {k:14s} held {m['rate']:5.1%}   precision {m['precision']:.2f}  "
                      f"recall {m['recall']:.2f}  f1 {m['f1']:.2f}")
            else:
                print(f"  {k:14s} mean |x| {m['rate']:.2f}   corr {m['corr']:+.2f}  "
                      f"mae {m['mae']:.2f}")

    report("train", tr_f, [tr_a])
    report("holdout", te_f, te_a)
    print(f"saved idm.pt and idm.json in {run_dir}; "
          f"label footage with `neurofly label ... --idm {run_dir}`")


if __name__ == "__main__":
    main()
