"""A replay video: frames with the brain's spikes and the controls drawn beside them.

    neurofly replay --video videos/live.mp4 --probe probe.npz --out videos/replay.mp4
    neurofly replay --recording data/recordings/run1 --probe probe.npz --out videos/replay.mp4

The probe comes from `--probe ... --probe-out probe.npz` on play, watch or neurofly-core run;
the frames from the same session (`--record` on play / run) or a recording. The right panel
is a raster of the probed neurons over the last --history steps and, below it, the controls
held on that frame (from a recording's actions.npy, when given).
"""
from __future__ import annotations

import argparse
import json
import os

import imageio
import numpy as np
from PIL import Image, ImageDraw

from neurofly_core.controls import ControlLayout
from neurofly_core.io.video import VideoFile


def raster_image(spikes: np.ndarray, width: int, height: int) -> Image.Image:
    """spikes: (steps, neurons) counts -> a height x width image, neurons down, time across."""
    steps, n = spikes.shape
    img = np.zeros((height, width), np.uint8)
    if steps and n:
        rows = np.linspace(0, height, n + 1).astype(int)
        cols = np.linspace(0, width, steps + 1).astype(int)
        peak = max(1.0, float(spikes.max()))
        for t in range(steps):
            for i in range(n):
                v = spikes[t, i]
                if v > 0:
                    r0, r1 = rows[i], max(rows[i] + 1, rows[i + 1])
                    c0, c1 = cols[t], max(cols[t] + 1, cols[t + 1])
                    img[r0:r1, c0:c1] = int(80 + 175 * min(1.0, v / peak))
    return Image.fromarray(img).convert("RGB")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=None, help="frames of the session")
    p.add_argument("--recording", default=None, help="a recording directory (video + actions)")
    p.add_argument("--probe", default=None, help="probe .npz written by --probe-out")
    p.add_argument("--out", required=True)
    p.add_argument("--history", type=int, default=50, help="steps of raster shown")
    p.add_argument("--panel", type=int, default=320, help="width of the side panel")
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()
    if not args.video and not args.recording:
        raise SystemExit("give --video or --recording")
    video_path = args.video or os.path.join(args.recording, "video.mp4")
    actions = layout = None
    if args.recording and os.path.exists(os.path.join(args.recording, "actions.npy")):
        actions = np.load(os.path.join(args.recording, "actions.npy"))
        with open(os.path.join(args.recording, "meta.json")) as f:
            layout = ControlLayout.from_dict(json.load(f)["layout"])
    spikes = rates = None
    if args.probe:
        z = np.load(args.probe)
        spikes, rates = z["spikes"], z["rates"]

    video = VideoFile(video_path)
    fps = video.fps or 10.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frame = video.reset()
    t = 0
    with imageio.get_writer(args.out, fps=fps, macro_block_size=1) as w:
        while frame is not None and (args.max_frames is None or t < args.max_frames):
            h, wd = frame.shape[:2]
            canvas = Image.new("RGB", (wd + args.panel, h), (16, 16, 16))
            canvas.paste(Image.fromarray(frame), (0, 0))
            draw = ImageDraw.Draw(canvas)
            y = 4
            if spikes is not None and t < len(spikes):
                lo = max(0, t + 1 - args.history)
                ras = raster_image(spikes[lo:t + 1], args.panel - 8, int(h * 0.6))
                canvas.paste(ras, (wd + 4, y))
                y += ras.height + 4
                draw.text((wd + 4, y), f"step {t}  probe: {spikes.shape[1]} neurons, "
                          f"{int(spikes[t].sum())} spikes, mean {rates[t].mean():.1f} Hz",
                          fill=(220, 220, 220))
                y += 14
            if actions is not None and t < len(actions):
                st = layout.decode(actions[t])
                held = ", ".join(st.held) or "-"
                draw.text((wd + 4, y), f"held: {held}", fill=(120, 220, 120))
                y += 14
                if layout.mouse or layout.axes:
                    extra = f"mouse {st.dx:+.0f},{st.dy:+.0f}"
                    if st.axes:
                        extra += f"  axes {st.axis}"
                    draw.text((wd + 4, y), extra, fill=(120, 220, 120))
                    y += 14
            w.append_data(np.asarray(canvas))
            frame = video.read()
            t += 1
    video.close()
    print(f"wrote {t} frames to {args.out}")


if __name__ == "__main__":
    main()
