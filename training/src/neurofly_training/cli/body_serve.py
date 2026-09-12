"""Host the MuJoCo fly body and drive it over JSON, from any language.

    neurofly body-serve                                     # ws://127.0.0.1:8767
    neurofly body-serve --artifact artifacts/walk           # a brain that can act ("brain": true)
    neurofly body-serve --stdio                             # JSON lines on stdin/stdout
    neurofly body-serve --ws 0.0.0.0:8767 --token secret    # on a server

Then examples/three_viewer.html?glb=../assets/fly.glb&ws=ws://127.0.0.1:8767 shows the body
live, with buttons, while a program (Node: NeuroFlyBody) sends step / gait / set_pose /
replay requests. The protocol is in neurofly_training.body.server and docs/runtime.md.
"""
from __future__ import annotations

import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="forward", help="body task (forward, ball)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifact", default=None, help="a body artifact whose brain may act")
    p.add_argument("--ws", default="127.0.0.1:8767", metavar="HOST:PORT")
    p.add_argument("--stdio", action="store_true", help="JSON lines instead of a WebSocket")
    p.add_argument("--token", default=os.environ.get("NEUROFLY_TOKEN"),
                   help="require this token from clients (or NEUROFLY_TOKEN)")
    p.add_argument("--origins", default=None,
                   help="comma-separated browser origins allowed (default: any)")
    p.add_argument("--every", type=int, default=10, help="control steps per pose sent (10: 50 fps)")
    p.add_argument("--no-realtime", action="store_true",
                   help="run requests as fast as the physics allows instead of pacing to real time")
    p.add_argument("--time-limit", type=float, default=0.0,
                   help="seconds per episode before it ends on its own (0: never; the "
                        "training tasks use 2)")
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    from neurofly_training.body.server import BodySession, serve_stdio, serve_ws
    brain = None
    if args.artifact:
        from neurofly_core.artifact import load_model
        brain = load_model(args.artifact)
    session = BodySession(args.task, seed=args.seed, brain=brain, camera_id=args.camera,
                          width=args.width, height=args.height, every=args.every,
                          realtime=not args.no_realtime,
                          time_limit=args.time_limit or float("inf"))
    if args.stdio:
        serve_stdio(session)
        return
    host, _, port = args.ws.rpartition(":")
    origins = [o.strip() for o in args.origins.split(",")] if args.origins else None
    try:
        serve_ws(session, host or "127.0.0.1", int(port), token=args.token, origins=origins)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


if __name__ == "__main__":
    main()
