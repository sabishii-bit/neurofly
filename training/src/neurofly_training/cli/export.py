"""Export a run as an artifact that neurofly-core runs from any language.

    neurofly export runs/imitate_myapp artifacts/myapp
    neurofly export runs/pc_malecns_20260911-1200 artifacts/myapp --name myapp
    neurofly-core info artifacts/myapp              # then: serve, run, or a binding
"""
from __future__ import annotations

import argparse

from neurofly_core.artifact import describe, validate
from neurofly_training.export import export_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory (PPO, ES or imitation, on a PC task)")
    p.add_argument("out", help="artifact directory to write")
    p.add_argument("--name", default=None, help="artifact name (default: the run's)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    path = export_run(args.run, args.out, name=args.name, device=args.device)
    problems = validate(path)
    if problems:
        raise SystemExit("exported, but validation failed:\n"
                         + "\n".join(f"  {x}" for x in problems))
    print(describe(path))
    print(f"ok -> {path}")


if __name__ == "__main__":
    main()
