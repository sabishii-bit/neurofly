"""Open the 3D demos in a browser, building what they need first.

    python scripts/demo.py body        the fly cycling a tripod gait, in the Three.js viewer
    python scripts/demo.py live        the fly hosted by body-serve, driven from viewer buttons
    python scripts/demo.py brain       the brain atlas lit by a run's activity
    python scripts/demo.py workbench   what the fly saw, the brain and the body on one timeline
    python scripts/demo.py web-fps     the Three.js game training a served brain in the page
    python scripts/demo.py all         build every asset without opening anything

Assets land in assets/demo/ (ignored by git) and are reused when present; --rebuild
makes them again. --brain malecns uses the downloaded connectome (the default when the
data is there), --brain toy the small test brain. --no-open prints the URL instead of
opening it. Run with the Python that has neurofly installed.
"""
from __future__ import annotations

import argparse
import http.server
import os
import socket
import socketserver
import subprocess
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets", "demo")


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call([sys.executable, "-m", "neurofly_training", *args], cwd=ROOT)


def has_data() -> bool:
    from neurofly_training.envs import DEFAULT_DATA_DIR
    return os.path.exists(os.path.join(DEFAULT_DATA_DIR, "connectome-weights-traced-only.feather"))


def bars_video(path: str, seconds: float = 6.0, fps: int = 10) -> str:
    """A stimulus video for brains without a recording: a bright bar alternating sides."""
    import imageio
    import numpy as np
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for t in range(int(seconds * fps)):
            f = np.zeros((240, 320, 3), np.uint8)
            left = (t // fps) % 2 == 0
            f[:, :160] = 255 if left else 0
            f[:, 160:] = 0 if left else 255
            w.append_data(f)
    return path


def ensure_body(args) -> dict:
    """The fly and a pose file: the tripod gait as joint angles (default, clean), the gait
    through the physics (wobbly, honest), or a real fly's walking from flybody's dataset."""
    glb = os.path.join(ASSETS, "fly.glb")
    walk = getattr(args, "walk", "gait")
    poses = os.path.join(ASSETS, f"{walk}_poses.json")
    video = os.path.join(ASSETS, f"{walk}.mp4")
    if args.rebuild or not os.path.exists(glb):
        run("export-body", "--out", glb)
    if args.rebuild or not os.path.exists(poses):
        if walk == "physics":
            run("watch", "--task", "forward", "--policy", "gait", "--episodes", "1",
                "--max-steps", str(args.steps), "--poses", poses, "--video", video, "--every", "10")
        elif walk == "real":
            from neurofly_training.cli.body_replay import find_walking_dataset
            if not find_walking_dataset():
                run("body-replay", "--download")
            index = getattr(args, "index", None)
            run("body-replay", *(["--index", str(index)] if index is not None else ["--real"]),
                "--poses", poses, "--video", video)
        else:
            run("body-replay", "--gait", "--steps", str(args.steps), "--poses", poses,
                "--video", video)
    return {"glb": glb, "poses": poses, "video": video}


def ensure_brain(args) -> dict:
    brain = args.brain or ("malecns" if has_data() else "toy")
    atlas = os.path.join(ASSETS, f"atlas-{brain}")
    activity = os.path.join(ASSETS, f"activity-{brain}.json")
    stimulus = os.path.join(ASSETS, "stimulus.mp4")
    if args.rebuild or not os.path.exists(stimulus):
        bars_video(stimulus)
    if args.rebuild or not os.path.exists(os.path.join(atlas, "manifest.json")):
        run("export-atlas", "--out", atlas, "--brain", brain, "--subset", "central",
            *(["--synthetic-n", "1500"] if brain != "malecns" else []))
    if args.rebuild or not os.path.exists(activity):
        run("watch", "--task", stimulus, "--brain", brain, "--keys", "a,d", "--policy", "random",
            "--episodes", "1", "--max-steps", "60", "--activity-out", activity,
            "--video", os.path.join(ASSETS, "stimulus_seen.mp4"))
    return {"atlas": atlas, "activity": activity, "video": stimulus}


def serve(open_url: str | None, no_open: bool, port: int = 0) -> None:
    if not port:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=ROOT, **k)  # noqa: E731
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/{open_url}"
    print(f"\nserving {ROOT} on http://127.0.0.1:{port}\nopen: {url}\nCtrl+C to stop",
          flush=True)
    if not no_open:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()


def rel(path: str) -> str:
    return "../" + os.path.relpath(path, ROOT).replace(os.sep, "/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demo", choices=["body", "live", "brain", "workbench", "web-fps", "all"])
    p.add_argument("--brain", default=None, choices=["malecns", "toy"])
    p.add_argument("--steps", type=int, default=600, help="body: control steps of gait (500/s)")
    p.add_argument("--walk", default="gait", choices=["gait", "physics", "real"],
                   help="body: the tripod gait as joint angles (default), the gait through "
                        "the physics, or a real fly's walking (downloads flybody's dataset)")
    p.add_argument("--index", type=int, default=None,
                   help="body --walk real: which trajectory (default: the longest)")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--port", type=int, default=0, help="static server port (default: any free)")
    args = p.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    if args.demo == "body":
        b = ensure_body(args)
        serve(f"examples/three_viewer.html?glb={rel(b['glb'])}&poses={rel(b['poses'])}",
              args.no_open, args.port)
    elif args.demo == "live":
        glb = os.path.join(ASSETS, "fly.glb")
        if args.rebuild or not os.path.exists(glb):
            run("export-body", "--out", glb)
        proc = subprocess.Popen([sys.executable, "-m", "neurofly_training", "body-serve",
                                 "--ws", "127.0.0.1:8767"], cwd=ROOT)
        try:
            serve(f"examples/three_viewer.html?glb={rel(glb)}&ws=ws://127.0.0.1:8767",
                  args.no_open, args.port)
        finally:
            proc.terminate()
    elif args.demo == "brain":
        br = ensure_brain(args)
        serve(f"examples/brain_viewer.html?activity={rel(br['activity'])}", args.no_open, args.port)
    elif args.demo == "workbench":
        b, br = ensure_body(args), ensure_brain(args)
        serve("examples/workbench.html?atlas=" + rel(br["atlas"])
              + "&activity=" + rel(br["activity"])
              + "&video=" + rel(os.path.join(ASSETS, "stimulus_seen.mp4"))
              + "&glb=" + rel(b["glb"]) + "&poses=" + rel(b["poses"]), args.no_open, args.port)
    elif args.demo == "web-fps":
        brain = args.brain or ("malecns" if has_data() else "toy")
        art = os.path.join(ASSETS, f"fps-{brain}")
        if args.rebuild or not os.path.exists(os.path.join(art, "manifest.json")):
            run("build", art, "--brain", brain, "--keys", "w,a,d",
                *(["--synthetic-n", "1500"] if brain != "malecns" else []))
        proc = subprocess.Popen([sys.executable, "-m", "neurofly_core", "serve", art, "--ws",
                                 "127.0.0.1:8765", "--per-client"], cwd=ROOT)
        try:
            serve("examples/web_fps.html?ws=ws://127.0.0.1:8765", args.no_open, args.port)
        finally:
            proc.terminate()
    else:
        ensure_body(args)
        ensure_brain(args)
        print(f"assets in {ASSETS}", flush=True)


if __name__ == "__main__":
    main()
