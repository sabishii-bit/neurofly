"""Sweep a gain and pick the one that gives sparse readout activity.

    neurofly calibrate --brain malecns --window "My App"                  # frames from the screen
    neurofly calibrate --brain malecns --video data/recordings/run1/video.mp4 --param retina_gain
    neurofly calibrate --brain malecns --subset visual --target 0.3 --grid 0.5,1,2,4

Prints, for each value of --param (brain_gain by default), the fraction of readout neurons
that fire, the fraction of all neurons that fire, and the mean readout rate, then the pick
closest to --target. Use the picked value as the flag on train / imitate / build.
"""
from __future__ import annotations

import argparse

import numpy as np

from neurofly_core.io.video import ScreenCapture, VideoFile
from neurofly_training.calibrate import format_sweep, sweep
from neurofly_training.envs import add_env_args, make_pc_model, parse_region, resolve_env_args

LAYOUT_ARGS = ("task", "reward", "dry_run", "fps", "window", "region", "monitor", "max_steps",
               "keys", "buttons", "mouse", "scroll", "mouse_speed", "pad_buttons", "axes")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    p.add_argument("--video", default=None, help="frames from this video instead of the screen")
    p.add_argument("--frames", type=int, default=8, help="frames per gain value")
    p.add_argument("--param", default="brain_gain",
                   choices=["brain_gain", "retina_gain", "audio_gain"])
    p.add_argument("--grid", default=None, help="comma-separated gain values to try")
    p.add_argument("--target", type=float, default=0.2,
                   help="wanted fraction of active readout neurons")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.task = "pc"
    args.keys = args.keys or "w"
    opts = {k: v for k, v in resolve_env_args(args).items() if k not in LAYOUT_ARGS}
    opts.pop(args.param, None)
    if args.param == "audio_gain" and not args.audio:
        raise SystemExit("--param audio_gain needs --audio (the frames have no sound otherwise)")

    if args.video:
        src = VideoFile(args.video)
    else:
        src = ScreenCapture(window=args.window, region=parse_region(args.region),
                            monitor=args.monitor)
    frames = []
    f = src.reset()
    while f is not None and len(frames) < args.frames:
        frames.append(f)
        f = src.read()
    src.close()
    chunks = None
    if args.audio and str(args.audio).lower() != "none":
        rng = np.random.default_rng(0)
        chunks = [(0.05 * rng.standard_normal((1600, 1))).astype(np.float32) for _ in frames]
        print("audio: white noise at -26 dB stands in for sound during calibration")

    def build(**kw):
        return make_pc_model(device=args.device, keys="w", **opts, **kw)

    grid = [float(v) for v in args.grid.split(",")] if args.grid else None
    res = sweep(build, args.param, frames, chunks, grid=grid, target=args.target)
    print(format_sweep(res))


if __name__ == "__main__":
    main()
