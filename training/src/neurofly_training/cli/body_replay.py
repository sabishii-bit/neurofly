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
import time
import zipfile

from neurofly_training.body.kinematics import gait_qpos, play, real_walking_qpos
from neurofly_training.envs import make_body_env

DATA_DIR = os.environ.get("FLYBODY_DATA", "data/flybody-data")
# flybody's walking imitation dataset (Vaxenburg et al., figshare 10.25378/janelia.25309105),
# 3.0 GB. The ndownloader host serves it directly; janelia.figshare.com answers 202 for it.
WALKING_URL = "https://ndownloader.figshare.com/files/51196868"
WALKING_ZIP = "datasets_walking-imitation.zip"


def find_walking_dataset(data_dir: str = DATA_DIR) -> str | None:
    """The walking dataset under ``data_dir``: the small one (100 snippets, quick to
    load) before the full one (16 252 snippets), .h5 or .hdf5."""
    hits = [h for ext in ("h5", "hdf5")
            for h in glob.glob(os.path.join(data_dir, "**", f"*.{ext}"), recursive=True)]
    hits = [h for h in hits if "walk" in os.path.basename(h).lower()] or hits
    hits.sort(key=lambda h: ("small" not in os.path.basename(h).lower(), h))
    return hits[0] if hits else None


def longest_trajectory(path: str) -> int:
    """Index of the longest snippet in a walking dataset (they run 0.5 to a few seconds)."""
    import h5py
    with h5py.File(path, "r") as f:
        return int(f["trajectory_lengths"][:].argmax())


def download(url: str, path: str, *, chunk: int = 1 << 20, log=print) -> str:
    """Stream ``url`` to ``path``, resuming a partial file with a Range request and
    printing progress; figshare's 202 (file being staged) is retried."""
    import requests
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    for attempt in range(30):
        headers = {"Range": f"bytes={have}-"} if have else {}
        r = requests.get(url, stream=True, headers=headers, timeout=60)
        if r.status_code == 202:                     # figshare is preparing the file
            r.close()
            log(f"figshare is staging the file, retrying ({attempt + 1})")
            time.sleep(min(60, 5 * (attempt + 1)))
            continue
        if r.status_code == 200 and have:            # server ignored the range: start over
            have = 0
        if r.status_code not in (200, 206):
            raise SystemExit(f"download failed: HTTP {r.status_code} for {url}")
        total = have + int(r.headers.get("Content-Length", 0))
        log(f"downloading {url} -> {path} ({total / 1e9:.2f} GB"
            + (f", resuming at {have / 1e9:.2f} GB" if have else "") + ")")
        t0 = time.time()
        with open(part, "ab" if have else "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
                have += len(block)
                if total and int(have / (64 << 20)) != int((have - len(block)) / (64 << 20)):
                    rate = (have / 1e6) / max(1e-3, time.time() - t0)
                    log(f"  {have / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.0f} MB/s)")
        if total and have < total:
            log("connection dropped, resuming")
            continue
        os.replace(part, path)
        return path
    raise SystemExit("gave up downloading after 30 attempts; run again to resume")


def download_walking_dataset(data_dir: str = DATA_DIR, log=print) -> str | None:
    """Fetch and unpack flybody's walking imitation dataset; returns the .h5 found."""
    zip_path = os.path.join(data_dir, WALKING_ZIP)
    if not os.path.exists(zip_path):
        download(WALKING_URL, zip_path, log=log)
    log(f"unpacking {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(os.path.join(data_dir, WALKING_ZIP[:-4]))
    os.remove(zip_path)
    return find_walking_dataset(data_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--gait", action="store_true", help="the tripod gait as joint angles")
    src.add_argument("--trajectory", default=None, metavar="FILE.h5",
                     help="a walking imitation dataset (default: the one under "
                          f"{DATA_DIR} when --index is given)")
    src.add_argument("--real", action="store_true",
                     help=f"the longest snippet of the dataset under {DATA_DIR}")
    src.add_argument("--download", action="store_true",
                     help=f"download flybody's walking imitation dataset (3 GB) into {DATA_DIR}; "
                          "resumes if interrupted")
    p.add_argument("--index", type=int, default=None,
                   help="trajectory index in the dataset (default: the longest snippet)")
    p.add_argument("--steps", type=int, default=600, help="gait: control steps (500 per second)")
    p.add_argument("--gait-hz", type=float, default=2.0)
    p.add_argument("--amplitude", type=float, default=1.0, help="gait: radians per unit of action")
    p.add_argument("--every", type=int, default=10, help="show every N-th step (10: 50 fps)")
    p.add_argument("--poses", default=None, help="write poses.json here")
    p.add_argument("--video", default=None, help="render a video here")
    p.add_argument("--camera", type=int, default=1)
    args = p.parse_args()
    if args.download:
        found = find_walking_dataset()
        if found:
            print(f"already there: {found}")
            return
        found = download_walking_dataset(DATA_DIR, log=lambda s: print(s, flush=True))
        if not found:
            raise SystemExit(f"downloaded and unpacked into {DATA_DIR}, but found no .h5 in it")
        print(f"downloaded into {DATA_DIR}: {found}")
        return
    env = make_body_env("forward", seed=0, render_mode="rgb_array", camera_id=args.camera)
    env.reset()
    if args.gait or (args.trajectory is None and args.index is None and not args.real):
        from neurofly_training.body.gait import TripodGait
        qpos = gait_qpos(env.physics, TripodGait(500.0, args.gait_hz), args.steps,
                         amplitude=args.amplitude)
        fps = 500.0
        what = f"tripod gait, {args.steps} steps"
    else:
        path = args.trajectory or find_walking_dataset()
        if not path:
            raise SystemExit(f"no walking dataset under {DATA_DIR}; run with --download first")
        index = longest_trajectory(path) if args.index is None else args.index
        qpos, fps = real_walking_qpos(path, index)
        what = f"trajectory {index} of {path}, {len(qpos)} steps"
    out = play(env, qpos, fps=fps, every=args.every, video=args.video, poses=args.poses)
    env.close()
    print(f"{what}: {out['frames']} frames at {out['fps']:g} fps"
          + (f"; poses -> {out['poses']}" if args.poses else "")
          + (f"; video -> {out['video']}" if args.video else ""))


if __name__ == "__main__":
    main()
