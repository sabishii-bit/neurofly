"""neurofly-core: run a trained controller.

    neurofly-core info     artifacts/myapp                 # what is inside
    neurofly-core validate artifacts/myapp                 # check the files
    neurofly-core serve    artifacts/myapp                 # JSON lines over stdio
    neurofly-core serve    artifacts/myapp --ws 127.0.0.1:8765
    neurofly-core serve    artifacts/myapp --ws 0.0.0.0:8765 --per-client --token secret  # hosted
    neurofly-core serve    artifacts/myapp --grpc 127.0.0.1:50051
    neurofly-core run      artifacts/myapp --window "My App" --dry-run   # drive the PC itself
    neurofly-core run      artifacts/myapp --region 0,0,800,600 --audio loopback \
                           --stimulate type_re=^PPL1:20 --probe name=readout --probe-out probe.npz

``run`` needs the ``pc`` extra (screen and sound capture, keyboard, mouse and gamepad).
"""
from __future__ import annotations

import argparse
import sys
import time

from neurofly_core.experiments import (ProbeLog, activity_sinks, add_experiment_args,
                                       apply_experiments)


def cmd_info(args):
    from neurofly_core.artifact import describe
    print(describe(args.artifact))


def cmd_validate(args):
    from neurofly_core.artifact import validate
    problems = validate(args.artifact)
    if problems:
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("ok")


def cmd_serve(args):
    import os
    from neurofly_core.artifact import load_model
    from neurofly_core.server import serve_stdio, serve_ws

    def make():
        m = load_model(args.artifact, device=args.device)
        for line in apply_experiments(m, args):
            print(line, file=sys.stderr, flush=True)
        return m

    model = make()
    print(model.describe(), file=sys.stderr, flush=True)
    sinks = activity_sinks(model, args, fps=1000.0 / model.config.brain_ms)
    for line in sinks.describe():
        print(line, file=sys.stderr, flush=True)
    token = args.token if args.token is not None else os.environ.get("NEUROFLY_TOKEN") or None
    origins = [o.strip() for o in args.origins.split(",") if o.strip()] if args.origins else None
    try:
        if args.grpc:
            from neurofly_core.rpc.server import serve_grpc
            serve_grpc(model, args.grpc, after_step=[sinks.record])
        elif args.ws:
            host, _, port = args.ws.rpartition(":")
            serve_ws(model, host or "127.0.0.1", int(port), after_step=[sinks.record],
                     loader=make, per_client=args.per_client, token=token, origins=origins)
        else:
            serve_stdio(model, after_step=[sinks.record])
    finally:
        saved = sinks.close()
        if saved:
            print(f"activity written to {saved}", file=sys.stderr, flush=True)


def cmd_run(args):
    from neurofly_core.artifact import load_model
    from neurofly_core.io.audio import make_audio
    from neurofly_core.io.controls import PanicKey, make_controls
    from neurofly_core.io.guard import FocusGuard, Watchdog
    from neurofly_core.io.video import ScreenCapture

    model = load_model(args.artifact, device=args.device)
    if model.policy is None:
        raise SystemExit("this artifact has no policy; nothing to run")
    print(model.describe(), file=sys.stderr)
    for line in apply_experiments(model, args):
        print(line, file=sys.stderr)
    region = [int(v) for v in args.region.split(",")] if args.region else None
    video = ScreenCapture(window=args.window, region=region, monitor=args.monitor, fps=args.fps)
    sr = model.audition.sample_rate if model.audition else 16000
    audio = make_audio(args.audio, fps=args.fps, sample_rate=sr)
    controls = make_controls("log" if args.dry_run else "pc", model.layout)
    panic = PanicKey(args.panic)
    probe = ProbeLog(model, args.probe_out)
    sinks = activity_sinks(model, args, fps=args.fps)
    for line in sinks.describe():
        print(line, file=sys.stderr)
    print(f"capturing {video.describe()} at {args.fps:g} fps"
          + (f"; sound from {audio.name}" if audio else "")
          + ("; DRY RUN" if args.dry_run else "") + f"; {args.panic} stops", file=sys.stderr)
    for i in range(int(args.countdown), 0, -1):
        print(f"starting in {i} ...", file=sys.stderr)
        time.sleep(1.0)
    guard = FocusGuard(enabled=not args.no_focus_guard and not args.dry_run)
    guard.arm()
    print(guard.describe(), file=sys.stderr)
    watchdog = Watchdog(controls, timeout=max(1.0, 5.0 / args.fps))
    model.reset()
    if audio is not None:
        audio.reset()
    period = 1.0 / args.fps
    next_tick = time.perf_counter()
    steps, spikes, last = 0, 0, time.time()
    try:
        while not panic.stopped and (args.steps is None or steps < args.steps):
            if not guard.ok():
                print("keyboard focus left the target window: stopping", file=sys.stderr)
                break
            watchdog.heartbeat()
            lag = next_tick - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            next_tick = max(next_tick + period, time.perf_counter())
            frame = video.read()
            chunk = audio.read() if audio is not None else None
            state, info = model.step(frame, chunk)
            controls.apply(state)
            probe.record()
            sinks.record()
            steps += 1
            spikes += info["spikes"]
            if time.time() - last >= 1.0:
                print(f"step {steps:5d}  holding [{', '.join(state.held) or '-'}]  "
                      f"spikes/s {spikes / (time.time() - last):,.0f}", file=sys.stderr)
                last, spikes = time.time(), 0
    except KeyboardInterrupt:
        pass
    finally:
        watchdog.close()
        controls.close()
        video.close()
        if audio is not None:
            audio.close()
        panic.close()
    saved = probe.save()
    activity = sinks.close()
    print(f"stopped after {steps} steps; everything released"
          + (f"; probe written to {saved}" if saved else "")
          + (f"; activity written to {activity}" if activity else ""), file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(prog="neurofly-core", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="describe an artifact")
    s.add_argument("artifact")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("validate", help="check an artifact's files and shapes")
    s.add_argument("artifact")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("serve", help="serve the controller over stdio, WebSocket or gRPC")
    s.add_argument("artifact")
    s.add_argument("--ws", default=None, metavar="HOST:PORT", help="WebSocket instead of stdio")
    s.add_argument("--grpc", default=None, metavar="HOST:PORT", help="gRPC instead of stdio")
    s.add_argument("--per-client", action="store_true",
                   help="WebSocket: a fresh brain for every connection instead of one shared")
    s.add_argument("--token", default=None,
                   help="WebSocket: clients must present this (?token= or a hello message); "
                        "default: the NEUROFLY_TOKEN environment variable")
    s.add_argument("--origins", default=None,
                   help="WebSocket: comma-separated browser origins allowed (default: any)")
    s.add_argument("--device", default="cpu")
    add_experiment_args(s)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("run", help="capture the screen and drive keyboard, mouse and gamepad")
    s.add_argument("artifact")
    s.add_argument("--window", default=None, help="capture the window whose title contains this")
    s.add_argument("--region", default=None, help="capture left,top,width,height instead")
    s.add_argument("--monitor", type=int, default=1)
    s.add_argument("--audio", default=None, help="'loopback', a capture device, or none")
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--countdown", type=float, default=3.0)
    s.add_argument("--panic", default="esc")
    s.add_argument("--dry-run", action="store_true", help="print the controls instead")
    s.add_argument("--no-focus-guard", action="store_true",
                   help="keep going even when the keyboard focus leaves the target window")
    s.add_argument("--device", default="cpu")
    add_experiment_args(s)
    s.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
