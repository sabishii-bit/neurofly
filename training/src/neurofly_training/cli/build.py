"""Build a base artifact: the brain and its encoders, no policy yet.

    neurofly build artifacts/base --brain malecns --subset central --keys w,a,s,d --mouse
    neurofly build artifacts/base --brain malecns --keys w,a,s,d --audio loopback --include-frame

A base artifact is what a trainer in another language starts from: serve it, read features
with ``observe``, train a policy however you like, install it with ``set_policy`` and
``save`` (see artifact/SPEC.md and docs/runtime.md). ``neurofly-core run`` refuses an
artifact without a policy; everything else works.
"""
from __future__ import annotations

import argparse

from neurofly_core.artifact import describe, save_model, validate
from neurofly_training.envs import add_env_args, make_pc_model, resolve_env_args


def build_artifact(out: str, name: str | None = None, device: str = "cpu", **env_kwargs) -> str:
    model = make_pc_model(device=device, name=name, **env_kwargs)
    return save_model(model, out, extra={"base": True, "options": env_kwargs})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="artifact directory to write")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="cpu")
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    args = p.parse_args()
    args.task = "pc"
    env_kwargs = resolve_env_args(args)
    path = build_artifact(args.out, name=args.name, device=args.device, **env_kwargs)
    problems = validate(path)
    if problems:
        raise SystemExit("written, but validation failed:\n"
                         + "\n".join(f"  {x}" for x in problems))
    print(describe(path))
    print(f"ok -> {path}")


if __name__ == "__main__":
    main()
