"""Label footage that has no input log, with an inverse dynamics model.

    neurofly label footage/a.mp4 footage/b.mp4 --idm runs/idm1 --out data/recordings/labelled

Writes one recording directory per video under --out (video.mp4, actions.npy, meta.json),
ready for `neurofly imitate`, `neurofly surrogate` and `neurofly eval`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from neurofly_core.io.video import VideoFile
from neurofly_training.idm import InverseDynamics


def label_video(idm: InverseDynamics, video_path: str, out_dir: str, max_frames=None) -> int:
    video = VideoFile(video_path)
    fps = video.fps or 10.0
    frames = []
    f = video.reset()
    while f is not None and (max_frames is None or len(frames) < max_frames):
        frames.append(f)
        f = video.read()
    video.close()
    actions = idm.predict(frames)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "video.mp4")
    if os.path.abspath(dst) != os.path.abspath(video_path):
        shutil.copyfile(video_path, dst)
    np.save(os.path.join(out_dir, "actions.npy"), actions)
    h, w = frames[0].shape[:2]
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"layout": idm.layout.to_dict(), "fps": fps, "size": [w, h],
                   "n_frames": len(actions), "labelled_by": "idm", "source": video_path},
                  f, indent=2)
    return len(actions)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+")
    p.add_argument("--idm", required=True, help="run directory written by neurofly idm")
    p.add_argument("--out", required=True, help="directory to put one recording per video in")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    idm = InverseDynamics.load(args.idm)
    for v in args.videos:
        name = os.path.splitext(os.path.basename(v))[0]
        n = label_video(idm, v, os.path.join(args.out, name), args.max_frames)
        print(f"{v}: {n} frames -> {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
