"""Experiments on the body's brain-in-the-loop env, by connectome annotation."""
from __future__ import annotations

import numpy as np

from neurofly_core.experiments import wants_activity
from neurofly_core.selection import parse_spec


def select_cx(cx, sel: dict) -> np.ndarray:
    if "indices" in sel:
        return np.asarray(sel["indices"], dtype=np.int64)
    if "ids" in sel:
        return cx.select(bodyId=list(sel["ids"]))
    if "type_re" in sel:
        return cx.select(type_re=sel["type_re"])
    if "superclass" in sel:
        return cx.select(superclass=sel["superclass"])
    raise ValueError(f"the body env cannot resolve the selection {sel}")


def apply_to_body_env(env, args) -> list[str]:
    """--stimulate / --silence on a BrainInLoopEnv (has .brain and .cx); no probe."""
    done = []
    brain, cx = getattr(env, "brain", None), getattr(env, "cx", None)
    if brain is None:
        if args.stimulate or args.silence or args.probe or wants_activity(args):
            done.append("no brain in this environment: experiments ignored")
        return done
    for spec in args.stimulate:
        sel, mv = parse_spec(spec)
        if mv is None:
            raise SystemExit(f"--stimulate {spec!r} needs a drive: {spec}:20")
        idx = select_cx(cx, sel)
        brain.stimulate(idx, mv)
        done.append(f"stimulate {sel} with {mv:g} mV: {len(idx)} neurons")
    for spec in args.silence:
        sel, _ = parse_spec(spec)
        idx = select_cx(cx, sel)
        brain.silence(idx)
        done.append(f"silence {sel}: {len(idx)} neurons")
    if args.probe:
        done.append("probes are recorded on PC tasks only")
    if wants_activity(args):
        n = env.watch_activity(True, substeps=bool(getattr(args, "activity_substeps", False)))
        done.append(f"activity: recording every spike of {n} neurons")
    return done
