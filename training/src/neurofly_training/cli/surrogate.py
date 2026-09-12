"""Train through the brain with surrogate gradients, on your recordings.

    neurofly surrogate data/recordings/run1 --brain malecns --epochs 5 --run-name through_brain
    neurofly surrogate data/recordings/run1 --subset visual --bptt-window 16 --lr 5e-3

Unlike ``imitate``, which fits only the map from readout rates to controls, this moves
the retina's input map too, by backpropagating through the spiking dynamics (see
neurofly_training.surrogate). The result is written directly as an artifact in
runs/<name>/artifact and scored on the held-out tail of each recording.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neurofly_core.artifact import describe, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.io.audio import AudioFile, SilentAudio
from neurofly_core.io.video import VideoFile
from neurofly_training.envs import ENV_ARGS, add_env_args, make_pc_model, resolve_env_args
from neurofly_training.pc.imitation import evaluate
from neurofly_training.surrogate import train_surrogate

LAYOUT_ARGS = ("task", "keys", "buttons", "mouse", "scroll", "mouse_speed", "pad_buttons", "axes",
               "reward", "dry_run", "fps", "window", "region", "monitor", "max_steps")
# --reward is kept on args for the odour source; it plays no part in the fit


def load_frames(rec: str, with_audio: bool, max_frames: int | None):
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    actions = np.load(os.path.join(rec, "actions.npy"))
    video = VideoFile(os.path.join(rec, "video.mp4"))
    fps = video.fps or meta.get("fps", 10.0)
    audio = None
    if with_audio:
        wav = os.path.join(rec, "audio.wav")
        audio = AudioFile(wav, fps=fps) if os.path.exists(wav) else SilentAudio(fps=fps)
    frames, chunks = [], []
    frame = video.reset()
    chunk = audio.reset() if audio else None
    while frame is not None and (max_frames is None or len(frames) < max_frames):
        frames.append(frame)
        chunks.append(chunk)
        frame = video.read()
        chunk = audio.read() if audio else None
    video.close()
    if audio:
        audio.close()
    n = min(len(frames), len(actions))
    return ControlLayout.from_dict(meta["layout"]), frames[:n], chunks[:n], actions[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", help="directories written by neurofly record")
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--max-frames", type=int, default=None, help="per recording")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--bptt-window", type=int, default=8, help="frames per gradient step")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    args.task = os.path.join(args.recordings[0], "video.mp4")
    if args.audio is None:
        args.audio = "file"
    env_kwargs = {k: v for k, v in resolve_env_args(args).items() if k not in LAYOUT_ARGS}
    run_dir = os.path.join("runs", args.run_name or f"surrogate_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    layout = None
    train, hold = [], []
    for rec in args.recordings:
        lay, frames, chunks, actions = load_frames(rec, str(args.audio).lower() != "none",
                                                   args.max_frames)
        if layout is None:
            layout = lay
        elif lay.names != layout.names:
            raise SystemExit(f"{rec} was recorded with controls {lay.names}, "
                             f"expected {layout.names}")
        n_te = int(len(frames) * args.holdout)
        n_tr = len(frames) - n_te
        train.append((frames[:n_tr], chunks[:n_tr], actions[:n_tr]))
        if n_te:
            hold.append((frames[n_tr:], chunks[n_tr:], actions[n_tr:]))
    frames = [f for fr, _, _ in train for f in fr]
    chunks = [c for _, ch, _ in train for c in ch]
    actions = np.concatenate([a for _, _, a in train])
    model = make_pc_model(layout=layout, device=args.device, name=os.path.basename(run_dir),
                          **env_kwargs)
    print(model.describe())
    dets = None
    if model.detection is not None:
        from neurofly_training.pc.detect import cached, make_detector
        detector = make_detector(args.detect, device=args.device)
        dets, hold_dets = [], []
        for rec, (fr, _, _), h in zip(args.recordings, train, hold + [None] * len(train)):
            d = cached(detector, os.path.abspath(os.path.join(rec, "video.mp4")))
            d.reset()
            per = [d.detect(f) for f in (fr + (h[0] if h else []))]
            d.close()
            dets += per[:len(fr)]
            hold_dets.append(per[len(fr):])
        hold = [(fr, ch, ac, hd) for (fr, ch, ac), hd in zip(hold, hold_dets)]
        print(f"detections from {detector.name}: {sum(len(x) for x in dets)} boxes on "
              f"{len(dets)} training frames")
    print(f"training through the brain on {len(frames)} frames, {args.epochs} epochs ...")
    t0 = time.time()
    torch_seed(args.seed)
    senses = {}
    if model.senses():
        from neurofly_training.envs import sense_sources
        from neurofly_training.pc.task import load_task
        for kw, fn in sense_sources(model, load_task(args.reward)).items():
            senses[kw] = [fn(f, c, {"t": t}, dets[t] if dets else None)
                          for t, (f, c) in enumerate(zip(frames, chunks))]
    history = train_surrogate(model, frames, chunks, actions, epochs=args.epochs, lr=args.lr,
                              window=args.bptt_window, l2=args.l2, device=args.device,
                              verbose=True, detections=dets, **senses)
    print(f"done in {time.time() - t0:.0f} s; "
          f"loss {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}")

    out = save_model(model, os.path.join(run_dir, "artifact"),
                     extra={"algo": "surrogate", "recordings": args.recordings, "history": history})
    problems = validate(out)
    if problems:
        raise SystemExit("written, but validation failed:\n"
                         + "\n".join(f"  {x}" for x in problems))
    cfg = {k: getattr(args, k) for k in ENV_ARGS}
    cfg.update(algo="surrogate", seed=args.seed, layout=layout.to_dict(), artifact=out)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    def report(name, sets):
        if not sets:
            return
        X, Y = [], []
        for item in sets:
            fr, ch, ac = item[:3]
            hd = item[3] if len(item) > 3 else [None] * len(fr)
            model.reset()
            for f, c, d in zip(fr, ch, hd):
                X.append(model.observe(f, c, detections=d))
            Y.append(ac)
        X, Y = np.stack(X), np.concatenate(Y)
        print(f"{name}:")
        for k, m in evaluate(model.policy, X, Y).items():
            if "f1" in m:
                print(f"  {k:14s} held {m['rate']:5.1%}   precision {m['precision']:.2f}  "
                      f"recall {m['recall']:.2f}  f1 {m['f1']:.2f}")
            else:
                print(f"  {k:14s} mean |x| {m['rate']:.2f}   corr {m['corr']:+.2f}  "
                      f"mae {m['mae']:.2f}")

    report("train", train)
    report("holdout", hold)
    print(describe(out))
    print(f"ok -> {out}")


def torch_seed(seed: int) -> None:
    import torch
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
