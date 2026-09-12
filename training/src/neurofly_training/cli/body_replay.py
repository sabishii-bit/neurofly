"""Play joint trajectories through the fly body, no physics: the limbs as they should look.

    neurofly body-replay --gait --poses assets/gait.json --video videos/gait.mp4
    neurofly body-replay --trajectory data/flybody-data/walk.h5 --index 3 --poses assets/walk.json
    neurofly body-replay --download                # fetch flybody's walking dataset (large)

--gait is the open-loop tripod gait as joint angles around the standing pose; --trajectory
is a real fly's walking from flybody's walking imitation dataset. Both write poses.json
for examples/three_viewer.html and the workbench, and a video with --video.
"""
from __future__ import annotations

import argparse
import glob
import os

from neurofly_training.body.kinematics import gait_qpos, play, real_walking_qpos
from neurofly_training.envs import make_body_env

DATA_DIR = os.environ.get("FLYBODY_DATA", "data/flybody-data")


def find_walking_dataset(data_dir: str = DATA_DIR) -> str | None:
    hits = sorted(glob.glob(os.path.join(data_dir, "**", "*.h5"), recursive=True))
    hits = [h for h in hits if "walk" in os.path.basename(h).lower()] or hits
    return hits[0] if hits else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--gait", action="store_true", help="the tripod gait as joint angles")
    src.add_argument("--trajectory", default=None, metavar="FILE.h5",
                     help="a walking imitation dataset (default: the one under "
                          f"{DATA_DIR} when --index is given)")
    src.add_argument("--download", action="store_true",
                     help=f"download flybody's walking imitation dataset into {DATA_DIR}")
    p.add_argument("--index", type=int, default=None, help="trajectory index in the dataset")
    p.add_argument("--steps", type=int, default=600, help="gait: control steps (500 per second)")
    p.add_argument("--gait-hz", type=float, default=2.0)
    p.add_argument("--amplitude", type=float, default=1.0, help="gait: radians per unit of action")
    p.add_argument("--every", type=int, default=10, help="show every N-th step (10: 50 fps)")
    p.add_argument("--poses", default=None, help="write poses.json here")
    p.add_argument("--video", default=None, help="render a video here")
    p.add_argument("--camera", type=int, default=1)
    args = p.parse_args()
    if args.download:
        from flybody.download_data import figshare_download
        os.makedirs(DATA_DIR, exist_ok=True)
        figshare_download("walking-imitation-dataset", DATA_DIR)
        print(f"downloaded into {DATA_DIR}: {find_walking_dataset()}")
        return
    env = make_body_env("forward", seed=0, render_mode="rgb_array", camera_id=args.camera)
    env.reset()
    if args.gait or (args.trajectory is None and args.index is None):
        from neurofly_training.body.gait import TripodGait
        qpos = gait_qpos(env.physics, TripodGait(500.0, args.gait_hz), args.steps,
                         amplitude=args.amplitude)
        fps = 500.0
        what = f"tripod gait, {args.steps} steps"
    else:
        path = args.trajectory or find_walking_dataset()
        if not path:
            raise SystemExit(f"no walking dataset under {DATA_DIR}; run with --download first")
        qpos, fps = real_walking_qpos(path, args.index or 0)
        what = f"trajectory {args.index or 0} of {path}, {len(qpos)} steps"
    out = play(env, qpos, fps=fps, every=args.every, video=args.video, poses=args.poses)
    env.close()
    print(f"{what}: {out['frames']} frames at {out['fps']:g} fps"
          + (f"; poses -> {out['poses']}" if args.poses else "")
          + (f"; video -> {out['video']}" if args.video else ""))


if __name__ == "__main__":
    main()
