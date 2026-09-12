"""Score an artifact offline against recordings.

    neurofly eval artifacts/myapp data/recordings/run1 data/recordings/run2
    neurofly eval artifacts/myapp data/recordings/run1 --reward my_project.py:Score --out eval.json

Prints agreement with what you did (per key, button and axis) and, with --reward, the
reward the policy would have earned on the footage; writes the full report as JSON next
to the artifact (eval.json) unless --out says otherwise.
"""
from __future__ import annotations

import argparse
import json
import os

from neurofly_training.evaluation import evaluate_artifact, format_report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("artifact")
    p.add_argument("recordings", nargs="+", help="directories written by neurofly record")
    p.add_argument("--reward", default=None, help="Task for reward: module:Name or file.py:Name")
    p.add_argument("--max-frames", type=int, default=None, help="per recording")
    p.add_argument("--out", default=None, help="JSON report path (default <artifact>/eval.json)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    report = evaluate_artifact(args.artifact, args.recordings, reward=args.reward,
                               max_frames=args.max_frames, device=args.device, progress=True)
    print(format_report(report))
    out = args.out or os.path.join(args.artifact, "eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
