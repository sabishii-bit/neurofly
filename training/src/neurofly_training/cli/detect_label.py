"""Auto-label objects in footage with a detector, into a dataset you can fine-tune on.

    neurofly detect-label footage/*.mp4 --detect "owl:enemy,health pack" --out data/objects
    neurofly detect-label data/recordings/run1 --detect "owl:enemy" --out data/objects --every 5
    neurofly detect-label footage/a.mp4 --detect runs/det1 --preview check.mp4   # see what it finds

Runs the detector on every --every-th frame and writes the YOLO layout (images/, labels/
with `class cx cy w h`, classes.json, data.yaml). No hand labelling is needed to get
started: the open-vocabulary detector's boxes are the labels, and `neurofly detect-train`
distils them into a fast detector. Any YOLO-format labelling tool opens the folder if you
want to correct boxes by hand. --preview writes a video with the boxes drawn instead of
(or as well as) the dataset.
"""
from __future__ import annotations

import argparse
import os

from neurofly_core.io.video import VideoFile
from neurofly_training.pc.detect import draw, make_detector, write_yolo_dataset


def frames_of(path: str, every: int, max_frames: int | None):
    """Frames of a video file (or a recording directory), every ``every``-th one."""
    if os.path.isdir(path):
        path = os.path.join(path, "video.mp4")
    video = VideoFile(path)
    out, t = [], 0
    f = video.reset()
    while f is not None and (max_frames is None or len(out) < max_frames):
        if t % every == 0:
            out.append(f)
        f = video.read()
        t += 1
    fps = video.fps or 10.0
    video.close()
    return out, fps / every


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", help="video files or recording directories")
    p.add_argument("--detect", required=True, help="detector spec (see --help of train)")
    p.add_argument("--out", default=None, help="dataset directory to write or extend")
    p.add_argument("--every", type=int, default=1, help="use every N-th frame")
    p.add_argument("--max-frames", type=int, default=None, help="per video")
    p.add_argument("--threshold", type=float, default=None, help="detector confidence cut")
    p.add_argument("--preview", default=None, help="write a video with the boxes drawn")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.out and not args.preview:
        raise SystemExit("give --out (a dataset) and/or --preview (a video)")
    detector = make_detector(args.detect, device=args.device, threshold=args.threshold)
    print(f"detector {detector.name}: classes {detector.classes}")
    writer = None
    if args.preview:
        import imageio
        os.makedirs(os.path.dirname(args.preview) or ".", exist_ok=True)
    total_frames = total_boxes = 0
    for v in args.videos:
        frames, fps = frames_of(v, args.every, args.max_frames)
        detector.reset()
        dets = [detector.detect(f) for f in frames]
        n_boxes = sum(len(d) for d in dets)
        total_frames += len(frames)
        total_boxes += n_boxes
        print(f"{v}: {len(frames)} frames, {n_boxes} boxes")
        if args.out:
            existing = len(os.listdir(os.path.join(args.out, "images"))) \
                if os.path.isdir(os.path.join(args.out, "images")) else 0
            prefix = os.path.splitext(os.path.basename(v.rstrip("/\\")))[0]
            write_yolo_dataset(args.out, frames, dets, detector.classes, prefix=prefix,
                               start=existing)
        if args.preview:
            if writer is None:
                writer = imageio.get_writer(args.preview, fps=max(1, int(round(fps))),
                                            macro_block_size=1)
            for f, d in zip(frames, dets):
                writer.append_data(draw(f, d, detector.classes))
    if writer is not None:
        writer.close()
        print(f"preview -> {args.preview}")
    detector.close()
    if args.out:
        print(f"dataset -> {args.out}: {total_frames} images, {total_boxes} boxes, "
              f"classes {detector.classes}")
        print(f"fine-tune with: neurofly detect-train {args.out} --out runs/det1")


if __name__ == "__main__":
    main()
