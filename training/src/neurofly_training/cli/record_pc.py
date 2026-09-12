"""Record yourself using the PC: frames, sound, and what you press, for imitation.

    neurofly record --window "My App" --keys w,a,s,d --mouse --out data/recordings/run1
    neurofly record --region 0,0,800,600 --keys w,a,s,d --buttons left \
        --audio loopback --seconds 120 --out data/recordings/run2

Press Esc to stop. Writes video.mp4, audio.wav (if --audio), actions.npy (frames x controls,
the ControlLayout's action vector) and meta.json, which train_imitation.py reads.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import imageio
import numpy as np

from neurofly_training.envs import make_layout, parse_region
from neurofly_core.io.audio import make_audio
from neurofly_core.io.controls import InputRecorder
from neurofly_core.io.video import ScreenCapture


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--keys", default=None, help="keys to log, e.g. w,a,s,d,space")
    p.add_argument("--buttons", default=None, help="mouse buttons to log: left,right,middle")
    p.add_argument("--mouse", action="store_true", help="log mouse motion")
    p.add_argument("--mouse-speed", type=float, default=50.0, help="pixels per step = full tilt")
    p.add_argument("--scroll", action="store_true", help="log scrolling")
    p.add_argument("--pad-buttons", default=None, help="gamepad buttons to log (XInput)")
    p.add_argument("--axes", default=None, help="gamepad axes to log: lx,ly,rx,ry,lt,rt")
    p.add_argument("--window", default=None, help="capture the window whose title contains this")
    p.add_argument("--region", default=None, help="capture left,top,width,height instead")
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--audio", default=None, help="'loopback', a capture device, or none")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--size", default="320,240", help="store frames at this width,height")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--panic", default="esc", help="key that stops the recording")
    args = p.parse_args()

    layout = make_layout(args.keys, args.buttons, args.mouse, args.scroll, args.mouse_speed,
                         args.pad_buttons, args.axes)
    w, h = (int(v) for v in args.size.split(","))
    w, h = w - w % 2, h - h % 2  # mp4 needs even dimensions
    video = ScreenCapture(window=args.window, region=parse_region(args.region),
                          monitor=args.monitor, size=(w, h), fps=args.fps)
    audio = make_audio(args.audio, fps=args.fps)
    os.makedirs(args.out, exist_ok=True)
    print(f"recording {video.describe()} -> {args.out} at {args.fps:g} fps; controls {layout.names}"
          + (f"; sound from {audio.name}" if audio is not None else "") + f"; {args.panic} stops")
    for i in range(int(args.countdown), 0, -1):
        print(f"starting in {i} ...")
        time.sleep(1.0)

    recorder = InputRecorder(layout, panic=args.panic)
    actions, chunks = [], []
    t0 = time.time()
    next_tick = time.perf_counter()
    if audio is not None:
        audio.reset()
    recorder.sample()
    try:
        with imageio.get_writer(os.path.join(args.out, "video.mp4"), fps=args.fps,
                                macro_block_size=1) as writer:
            while not recorder.stopped:
                lag = next_tick - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
                next_tick = max(next_tick + 1.0 / args.fps, time.perf_counter())
                writer.append_data(video.read())
                if audio is not None:
                    chunks.append(audio.read())
                actions.append(layout.encode(recorder.sample()))
                if len(actions) % int(args.fps * 5) == 0:
                    print(f"  {len(actions)} frames, {time.time() - t0:.0f} s, "
                          f"holding {sorted(recorder.keys | recorder.buttons) or '-'}")
                if args.seconds and time.time() - t0 >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        video.close()
        if audio is not None:
            audio.close()
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, layout.n)
    np.save(os.path.join(args.out, "actions.npy"), actions)
    meta = {"layout": layout.to_dict(), "fps": args.fps, "size": [w, h], "n_frames": len(actions),
            "region": list(video.region), "window": video.window_name, "audio": None}
    if audio is not None:
        import soundfile as sf
        data = np.concatenate(chunks) if chunks else np.zeros((0, audio.channels), np.float32)
        sf.write(os.path.join(args.out, "audio.wav"), data, audio.sample_rate)
        meta["audio"] = {"name": audio.name, "sample_rate": audio.sample_rate,
                         "seconds": len(data) / audio.sample_rate}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    rates = ", ".join(f"{n} {np.mean(actions[:, i] > 0):.0%}" if i < layout.n_binary
                      else f"{n} |{np.abs(actions[:, i]).mean():.2f}|"
                      for i, n in enumerate(layout.names))
    print(f"saved {len(actions)} frames ({len(actions) / args.fps:.0f} s) to {args.out}; {rates}")


if __name__ == "__main__":
    main()
