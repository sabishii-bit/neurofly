"""A brain atlas as plain files: every neuron's soma position, id and region, for viewers
and for other people's models.

    neurofly export-atlas --out assets/brain-atlas          # from the downloaded connectome
    neurofly export-atlas --out assets/brain-atlas --check  # verify the files' hashes

The atlas is the anatomy alone, in the format any renderer can read:

    positions.bin   float32 (n, 3), little-endian, micrometres, in the volume's own frame
    ids.bin         int64 (n,), the MaleCNS bodyId of each neuron
    groups.bin      uint8 (n,): 0 optic, 1 central, 2 descending, 3 vnc, 4 other
    superclass.json the superclass name of each neuron
    manifest.json   counts, bounds, the groups, a SHA-256 of every file, the data licence
    NOTICE.md       attribution

An activity replay (``neurofly_core.replay``) indexes neurons by ``bodyId``, so a model
that is not neurofly's at all can be shown on the same atlas.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from neurofly_training.data.connectome import Connectome, VNC_SUPERCLASSES

GROUPS = ["optic", "central", "descending", "vnc", "other"]
GROUP_COLOURS = ["#4fa3ff", "#ffc857", "#ff5da2", "#7ee787", "#9aa0a6"]
_OPTIC = {"ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal",
          "visual_projection_tbc"}
_DESCENDING = {"descending_neuron", "descending_neuron_tbc", "sensory_descending",
               "efferent_descending"}
DATA_LICENCE = ("MaleCNS v1.0 connectome, CC BY 4.0: FlyEM Project Team (HHMI Janelia), "
                "MRC LMB Cambridge, Google Research; see NOTICE.md")


def group_of(superclass) -> int:
    s = "" if superclass is None else str(superclass)
    if s in _OPTIC:
        return 0
    if s.startswith("cb_"):
        return 1
    if s in _DESCENDING:
        return 2
    if s in VNC_SUPERCLASSES:
        return 3
    return 4


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_atlas(cx: Connectome, out: str, known_only: bool = True) -> dict:
    """Write the atlas of ``cx`` to ``out`` and return its manifest. With ``known_only``
    neurons without a soma in the volume are left out (the default; a viewer wants real
    positions); otherwise they take their placed positions and are flagged."""
    positions, known = cx.positions(fill=not known_only)
    keep = np.flatnonzero(known) if known_only else np.arange(cx.n)
    positions = positions[keep].astype("<f4")
    ids = cx.neurons["bodyId"].values[keep].astype("<i8")
    superclass = ["" if s is None else str(s) for s in cx.neurons["superclass"].values[keep]]
    groups = np.asarray([group_of(s) for s in superclass], dtype=np.uint8)
    os.makedirs(out, exist_ok=True)
    positions.tofile(os.path.join(out, "positions.bin"))
    ids.tofile(os.path.join(out, "ids.bin"))
    groups.tofile(os.path.join(out, "groups.bin"))
    with open(os.path.join(out, "superclass.json"), "w") as f:
        json.dump(superclass, f)
    files = ["positions.bin", "ids.bin", "groups.bin", "superclass.json"]
    manifest = {
        "format": "neurofly-atlas", "version": 1, "dataset": "male-cns:v1.0",
        "connectome": cx.name, "n": int(len(keep)), "unit": "micrometre",
        "frame": "the MaleCNS volume: x right, y down, z front to back (as somaLocation, "
                 "times 0.008 um per voxel)",
        "positions": {"file": "positions.bin", "dtype": "float32", "shape": [int(len(keep)), 3]},
        "ids": {"file": "ids.bin", "dtype": "int64", "shape": [int(len(keep))]},
        "groups": {"file": "groups.bin", "dtype": "uint8", "shape": [int(len(keep))],
                   "names": GROUPS, "colours": GROUP_COLOURS,
                   "counts": {g: int((groups == i).sum()) for i, g in enumerate(GROUPS)}},
        "superclass": "superclass.json",
        "bounds": {"min": positions.min(axis=0).tolist() if len(keep) else None,
                   "max": positions.max(axis=0).tolist() if len(keep) else None},
        "known_only": bool(known_only),
        "known": None if known_only else known.tolist(),
        "sha256": {name: sha256(os.path.join(out, name)) for name in files},
        "licence": DATA_LICENCE,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(out, "NOTICE.md"), "w") as f:
        f.write("# Brain atlas notice\n\n"
                "Soma positions, ids and superclasses of the neurons of the MaleCNS v1.0\n"
                "connectome (FlyEM Project Team at HHMI Janelia Research Campus, with the MRC\n"
                "Laboratory of Molecular Biology, Cambridge, and Google Research), licensed\n"
                "CC BY 4.0. Exported by neurofly (`neurofly export-atlas`); the manifest\n"
                f"carries a SHA-256 of every file. Connectome subset: {cx.name}.\n")
    return manifest


def check_atlas(path: str) -> list[str]:
    """Problems with an atlas directory: missing files, hash mismatches, wrong sizes."""
    problems = []
    try:
        with open(os.path.join(path, "manifest.json")) as f:
            m = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [f"manifest: {e}"]
    if m.get("format") != "neurofly-atlas":
        return [f"not a neurofly-atlas manifest ({m.get('format')!r})"]
    n = int(m["n"])
    sizes = {"positions.bin": n * 12, "ids.bin": n * 8, "groups.bin": n}
    for name, digest in m.get("sha256", {}).items():
        p = os.path.join(path, name)
        if not os.path.exists(p):
            problems.append(f"{name}: missing")
            continue
        if name in sizes and os.path.getsize(p) != sizes[name]:
            problems.append(f"{name}: {os.path.getsize(p)} bytes, expected {sizes[name]}")
        if sha256(p) != digest:
            problems.append(f"{name}: SHA-256 differs from the manifest")
    return problems


def load_atlas(path: str) -> dict:
    """The atlas as arrays: ``positions`` (n, 3), ``ids`` (n,), ``groups`` (n,),
    ``superclass`` (list), plus the manifest."""
    with open(os.path.join(path, "manifest.json")) as f:
        m = json.load(f)
    n = int(m["n"])
    out = {"manifest": m,
           "positions": np.fromfile(os.path.join(path, "positions.bin"), "<f4").reshape(n, 3),
           "ids": np.fromfile(os.path.join(path, "ids.bin"), "<i8"),
           "groups": np.fromfile(os.path.join(path, "groups.bin"), np.uint8)}
    with open(os.path.join(path, m["superclass"])) as f:
        out["superclass"] = json.load(f)
    return out
