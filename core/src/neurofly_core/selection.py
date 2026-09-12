"""Choosing neurons at runtime, for stimulation, silencing and probes.

A selection is a dict with one of:

    {"indices": [...]}          positions in the model's neuron array
    {"ids": [...]}              the source connectome's ids (``neurons.ids`` in the artifact)
    {"type_re": "^PPL1"}        a regex on the neuron type (needs annotations in the artifact)
    {"superclass": "descending_neuron"}
    {"name": "readout" | "retina" | "audition" | "punish"}   a population the model already knows

On the command line the same thing is a string: ``type_re=^PPL1``, ``ids=123,456``,
``name=readout``; a stimulation adds the drive after a colon: ``type_re=^PPL1:20``.
"""
from __future__ import annotations

import re

import numpy as np

KEYS = ("indices", "ids", "type_re", "superclass", "name")


def parse_spec(spec: str) -> tuple[dict, float | None]:
    """``'type_re=^PPL1:20'`` -> ``({"type_re": "^PPL1"}, 20.0)``; without ``:mV`` the
    second value is None."""
    spec = spec.strip()
    mv = None
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        try:
            mv = float(tail)
            spec = head
        except ValueError:
            pass
    if "=" not in spec:
        raise ValueError(f"selection {spec!r} must look like key=value with key in {KEYS}")
    key, value = spec.split("=", 1)
    key = key.strip()
    if key not in KEYS:
        raise ValueError(f"unknown selection key {key!r}; choose from {KEYS}")
    if key in ("indices", "ids"):
        return {key: [int(v) for v in value.split(",") if v.strip()]}, mv
    return {key: value.strip()}, mv


def resolve(sel: dict, *, n: int, ids=None, types=None, superclass=None,
            named: dict | None = None) -> np.ndarray:
    """Neuron indices for a selection dict, given what the model knows about its neurons."""
    if "indices" in sel:
        idx = np.asarray(sel["indices"], dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f"indices outside [0, {n})")
        return idx
    if "ids" in sel:
        if ids is None:
            raise ValueError("this artifact has no neuron ids")
        pos = {int(v): i for i, v in enumerate(ids)}
        missing = [v for v in sel["ids"] if int(v) not in pos]
        if missing:
            raise ValueError(f"ids not in this model: {missing[:5]}")
        return np.asarray([pos[int(v)] for v in sel["ids"]], dtype=np.int64)
    if "type_re" in sel:
        if types is None:
            raise ValueError("this artifact has no neuron type annotations")
        pat = re.compile(sel["type_re"])
        return np.flatnonzero([bool(t) and pat.search(str(t)) is not None for t in types])
    if "superclass" in sel:
        if superclass is None:
            raise ValueError("this artifact has no superclass annotations")
        return np.flatnonzero(np.asarray(superclass, dtype=object) == sel["superclass"])
    if "name" in sel:
        if not named or sel["name"] not in named:
            raise ValueError(f"unknown population {sel['name']!r}; "
                             f"choose from {sorted(named or {})}")
        return np.asarray(named[sel["name"]], dtype=np.int64)
    raise ValueError(f"a selection needs one of {KEYS}")
