"""Fit the neuron-to-control map to your recorded use of the PC (video, sound, inputs).

    neurofly imitate data/recordings/run1 [data/recordings/run2] --run-name myapp
    neurofly imitate data/recordings/run1 --subset visual --audio none

Each recording directory comes from record_pc.py (video.mp4, actions.npy, meta.json and,
if it was recorded with sound, audio.wav). The recording is played through the brain, the
readout rates on every frame are the features, and a linear map from features to what you
did on that frame is the decoder. The last --holdout fraction of every recording is kept
for evaluation. The result loads into play_pc.py and watch.py like an ES run.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurofly_training.envs import ENV_ARGS, add_env_args, make_env, resolve_env_args
from neurofly_core.controls import ControlLayout
from neurofly_training.pc.imitation import collect_features, evaluate, fit_control_decoder

LAYOUT_ARGS = ("task", "keys", "buttons", "mouse", "scroll", "mouse_speed", "pad_buttons", "axes",
               "reward", "dry_run")


def load_recording(path: str):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    actions = np.load(os.path.join(path, "actions.npy"))
    layout = ControlLayout.from_dict(meta["layout"])
    return os.path.join(path, "video.mp4"), layout, actions, meta


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", help="directories written by record_pc.py")
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    p.add_argument("--holdout", type=float, default=0.2, help="fraction of each recording for eval")
    p.add_argument("--max-frames", type=int, default=None, help="per recording")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    args.task = os.path.join(args.recordings[0], "video.mp4")
    if args.audio is None:
        args.audio = "file"   # use the recording's sound if it has any
    env_kwargs = resolve_env_args(args)
    env_kwargs = {k: v for k, v in env_kwargs.items() if k not in LAYOUT_ARGS}

    run_dir = os.path.join("runs", args.run_name or f"imitate_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    layout = None
    Xtr, Ytr, Xte, Yte = [], [], [], []
    for rec in args.recordings:
        video, rec_layout, actions, meta = load_recording(rec)
        if layout is None:
            layout = rec_layout
        elif rec_layout.names != layout.names:
            raise SystemExit(f"{rec} was recorded with controls {rec_layout.names}, "
                             f"expected {layout.names}")
        env = make_env(video, seed=args.seed, device=args.device, layout=layout, **env_kwargs)
        t0 = time.time()
        print(f"{rec}: {len(actions)} frames at {env.fps:g} fps; {env.model.describe()}")
        X, Y = collect_features(env, actions, max_steps=args.max_frames, progress=True)
        env.close()
        n_te = int(len(X) * args.holdout)
        n_tr = len(X) - n_te
        Xtr.append(X[:n_tr])
        Ytr.append(Y[:n_tr])
        if n_te:
            Xte.append(X[n_tr:])
            Yte.append(Y[n_tr:])
        print(f"  {len(X)} frames through the brain in {time.time() - t0:.0f} s "
              f"({(time.time() - t0) / len(X) * 1000:.0f} ms per frame)")
    Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
    np.savez_compressed(os.path.join(run_dir, "features.npz"), X=Xtr, Y=Ytr)
    active = np.abs(Xtr).sum(0) > 0
    print(f"fitting on {len(Xtr)} frames; "
          f"{active.sum()} of {Xtr.shape[1]} features ever non-zero")

    dec = fit_control_decoder(Xtr, Ytr, layout, epochs=args.epochs, lr=args.lr, l2=args.l2,
                              seed=args.seed, verbose=True)
    dec.save(os.path.join(run_dir, "decoder.npz"))
    cfg = {k: getattr(args, k) for k in ENV_ARGS}
    cfg.update(keys=",".join(layout.keys), buttons=",".join(layout.buttons), mouse=layout.mouse,
               scroll=layout.scroll, mouse_speed=layout.mouse_speed,
               pad_buttons=",".join(layout.pad_buttons), axes=",".join(layout.axes),
               algo="imitation",
               seed=args.seed, recordings=args.recordings, n_frames=int(len(Xtr)))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    def report(name, X, Y):
        print(f"{name}:")
        for k, m in evaluate(dec, X, Y).items():
            if "f1" in m:
                print(f"  {k:14s} held {m['rate']:5.1%}   precision {m['precision']:.2f}  "
                      f"recall {m['recall']:.2f}  f1 {m['f1']:.2f}")
            else:
                print(f"  {k:14s} mean |x| {m['rate']:.2f}   corr {m['corr']:+.2f}  "
                      f"mae {m['mae']:.2f}")

    report("train", Xtr, Ytr)
    if Xte:
        report("holdout", np.concatenate(Xte), np.concatenate(Yte))
    print(f"saved decoder.npz in {run_dir}; use it with "
          f"`neurofly play --window <title> --run {run_dir}`")


if __name__ == "__main__":
    main()
