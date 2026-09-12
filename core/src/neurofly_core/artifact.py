"""The artifact: a trained controller on disk, readable from any language.

An artifact is a directory:

    manifest.json        everything that is not an array: sizes, parameters, layout,
                         provenance, and one entry per array giving its file, dtype and shape
    brain/*.bin          the synapses (CSC by presynaptic neuron: indptr, indices, values in mV)
    readout/ policy/ punish/ neurons/        shared by both kinds
    retina/ audition/                        kind "pc":   frames and sound in, controls out
    proprio/                                 kind "body": a body observation in, actuators out

Arrays are raw little-endian binaries with no header; the manifest says
``{"file": ..., "dtype": "float32" | "int64" | "bool", "shape": [...]}``. That
is the whole format: see artifact/SPEC.md at the repository root.

``save_model`` / ``load_model`` are the Python side for either kind; ``validate``
checks a directory without loading the brain.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
import scipy.sparse as sp

from neurofly_core.body import ActuatorDecoder, ActuatorLayout, BodyModel, ProprioMap
from neurofly_core.brain.lif import LIFBrain
from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.decode.mlp import MLPPolicy
from neurofly_core.encode.audition import AuditionEncoder
from neurofly_core.encode.vision import RetinaEncoder
from neurofly_core.model import BrainModel, Model, ModelConfig

FORMAT = "neurofly-artifact"
VERSION = 1
MANIFEST = "manifest.json"
_DTYPES = {"float32": np.float32, "int64": np.int64, "bool": np.bool_, "int32": np.int32,
           "float64": np.float64}


class _Writer:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def array(self, group: str, name: str, arr: np.ndarray) -> dict:
        arr = np.ascontiguousarray(arr)
        dtype = {np.dtype(np.float32): "float32", np.dtype(np.int64): "int64",
                 np.dtype(np.bool_): "bool", np.dtype(np.int32): "int32",
                 np.dtype(np.float64): "float64"}[arr.dtype]
        rel = f"{group}/{name}.bin"
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.astype(arr.dtype.newbyteorder("<")).tofile(path)
        return {"file": rel, "dtype": dtype, "shape": list(arr.shape)}

    def arrays(self, group: str, tables: dict) -> dict:
        return {k: self.array(group, k, v) for k, v in tables.items()}


class _Reader:
    def __init__(self, root: str):
        self.root = root

    def array(self, ref: dict) -> np.ndarray:
        path = os.path.join(self.root, ref["file"])
        arr = np.fromfile(path, dtype=np.dtype(_DTYPES[ref["dtype"]]).newbyteorder("<"))
        return arr.reshape(ref["shape"]).astype(_DTYPES[ref["dtype"]])

    def arrays(self, refs: dict) -> dict:
        return {k: self.array(v) for k, v in refs.items()}


def _common_manifest(w: _Writer, model: BrainModel, extra: dict | None) -> dict:
    b = model.brain
    c = model.config
    manifest = {
        "format": FORMAT, "version": VERSION, "kind": model.kind, "name": c.name,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config": c.to_dict(),
        "brain": {
            "n_neurons": b.n, "n_synapses": b.n_edges, "backend_hint": b.backend,
            "dt": b.dt, "tau_m": b.tau_m, "tau_syn": b.tau_syn, "v_rest": b.v_rest,
            "v_reset": b.v_reset, "v_th": b.v_th, "t_ref": b.t_ref, "rate_tau": b.rate_tau,
            "weights_unit": "mV per presynaptic spike (w_syn and gain already applied)",
            "weights_layout": "csc" if b.backend == "event" else "csr",
            "indptr": w.array("brain", "indptr", b.indptr.cpu().numpy()),
            "indices": w.array("brain", "indices", b.indices.cpu().numpy()),
            "values": w.array("brain", "values", b.vals.cpu().numpy().astype(np.float32)),
        },
        "readout": {"indices": w.array("readout", "indices", model.readout_idx)},
        "policy": None, "punish": None, "neurons": None,
        "features": {"n": model.n_features}, "actions": {"n": model.n_actions},
        "extra": extra or {},
    }
    if model.policy is not None:
        manifest["policy"] = {"params": model.policy.params(),
                              "tables": w.arrays("policy", model.policy.tables())}
    if model.punish_idx is not None and len(model.punish_idx):
        manifest["punish"] = {"indices": w.array("punish", "indices", model.punish_idx)}
    if model.neuron_ids is not None:
        manifest["neurons"] = {"ids": w.array("neurons", "ids", model.neuron_ids),
                               "note": "bodyId of each neuron in the source connectome"}
        if model.neuron_types is not None:
            ann = {"type": ["" if t is None else str(t) for t in model.neuron_types],
                   "superclass": ["" if t is None else str(t) for t in model.neuron_superclass]}
            os.makedirs(os.path.join(w.root, "neurons"), exist_ok=True)
            with open(os.path.join(w.root, "neurons", "annotations.json"), "w") as f:
                json.dump(ann, f)
            manifest["neurons"]["annotations"] = "neurons/annotations.json"
    return manifest


def save_model(model: BrainModel, path: str, extra: dict | None = None) -> str:
    """Write a PC or body controller as an artifact directory at ``path``."""
    w = _Writer(path)
    manifest = _common_manifest(w, model, extra)
    if model.kind == "pc":
        manifest["layout"] = model.layout.to_dict()
        manifest["retina"] = {"params": model.retina.params(),
                              "tables": w.arrays("retina", model.retina.tables())}
        manifest["audition"] = None
        if model.audition is not None:
            manifest["audition"] = {"params": model.audition.params(),
                                    "tables": w.arrays("audition", model.audition.tables())}
    elif model.kind == "body":
        manifest["body"] = {
            "actuators": list(model.layout.names),
            "obs_dim": model.n_obs, "obs_keys": list(model.obs_keys),
            "obs_slices": {k: list(v) for k, v in model.obs_slices.items()},
            "control_ms": model.config.brain_ms,
            "proprio": {"params": model.proprio.params(),
                        "tables": w.arrays("proprio", model.proprio.tables())},
        }
    else:
        raise ValueError(f"unknown model kind {model.kind!r}")
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(path: str) -> dict:
    with open(os.path.join(path, MANIFEST)) as f:
        m = json.load(f)
    if m.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} (format {m.get('format')!r})")
    if int(m.get("version", 0)) > VERSION:
        raise ValueError(f"artifact version {m['version']} is newer than this runtime ({VERSION})")
    m.setdefault("kind", "pc")
    return m


def _load_common(m: dict, r: _Reader, device: str, backend: str):
    bm = m["brain"]
    n = int(bm["n_neurons"])
    indptr, indices, values = r.array(bm["indptr"]), r.array(bm["indices"]), r.array(bm["values"])
    if bm["weights_layout"] == "csc":
        W = sp.csc_matrix((values, indices, indptr), shape=(n, n))
    else:
        W = sp.csr_matrix((values, indices, indptr), shape=(n, n))
    brain = LIFBrain(W, dt=bm["dt"], tau_m=bm["tau_m"], tau_syn=bm["tau_syn"],
                     v_rest=bm["v_rest"], v_reset=bm["v_reset"], v_th=bm["v_th"],
                     t_ref=bm["t_ref"], w_syn=1.0, gain=1.0, rate_tau=bm["rate_tau"],
                     device=device, backend=backend)
    punish = r.array(m["punish"]["indices"]) if m.get("punish") else None
    ids = r.array(m["neurons"]["ids"]) if m.get("neurons") else None
    types = superclass = None
    if m.get("neurons") and m["neurons"].get("annotations"):
        with open(os.path.join(r.root, m["neurons"]["annotations"])) as f:
            ann = json.load(f)
        types = np.asarray([t or None for t in ann["type"]], dtype=object)
        superclass = np.asarray([t or None for t in ann["superclass"]], dtype=object)
    return brain, n, dict(readout_idx=r.array(m["readout"]["indices"]),
                          config=ModelConfig.from_dict(m["config"]), punish_idx=punish,
                          neuron_ids=ids, neuron_types=types, neuron_superclass=superclass)


def _load_policy(m: dict, r: _Reader, layout, kind: str):
    if not m.get("policy"):
        return None
    p = m["policy"]
    tables = r.arrays(p["tables"])
    ptype = p["params"]["type"]
    if ptype == "linear":
        cls = ActuatorDecoder if kind == "body" else ControlDecoder
        return cls.from_tables(layout, p["params"], tables)
    if ptype == "mlp":
        return MLPPolicy.from_tables(layout, p["params"], tables)
    raise ValueError(f"unknown policy type {ptype!r}")


def load_model(path: str, device: str = "cpu", backend: str = "auto") -> BrainModel:
    """Rebuild a ``Model`` (kind pc) or ``BodyModel`` (kind body) from an artifact."""
    m = read_manifest(path)
    r = _Reader(path)
    brain, n, common = _load_common(m, r, device, backend)
    if m["kind"] == "pc":
        layout = ControlLayout.from_dict(m["layout"])
        retina = RetinaEncoder.from_tables(n, m["retina"]["params"],
                                           r.arrays(m["retina"]["tables"]), device=device)
        audition = None
        if m.get("audition"):
            audition = AuditionEncoder.from_tables(n, m["audition"]["params"],
                                                   r.arrays(m["audition"]["tables"]),
                                                   device=device)
        return Model(brain, layout=layout, retina=retina, audition=audition,
                     policy=_load_policy(m, r, layout, "pc"), **common)
    if m["kind"] == "body":
        b = m["body"]
        layout = ActuatorLayout(b["actuators"])
        proprio = ProprioMap.from_tables(n, b["proprio"]["params"],
                                         r.arrays(b["proprio"]["tables"]), device=device)
        return BodyModel(brain, proprio=proprio, layout=layout, obs_keys=b["obs_keys"],
                         obs_slices=b["obs_slices"], policy=_load_policy(m, r, layout, "body"),
                         **common)
    raise ValueError(f"unknown artifact kind {m['kind']!r}")


def validate(path: str) -> list[str]:
    """Problems with an artifact directory, without building the brain. Empty = fine."""
    problems: list[str] = []
    try:
        m = read_manifest(path)
    except Exception as e:
        return [str(e)]
    n = int(m["brain"]["n_neurons"])

    def check(ref: dict, what: str, dtype=None, ndim=None, max_index=None):
        p = os.path.join(path, ref["file"])
        if not os.path.exists(p):
            problems.append(f"{what}: missing file {ref['file']}")
            return None
        size = os.path.getsize(p)
        expect = int(np.prod(ref["shape"])) * np.dtype(_DTYPES[ref["dtype"]]).itemsize
        if size != expect:
            problems.append(f"{what}: {ref['file']} is {size} bytes, manifest says {expect}")
            return None
        if dtype and ref["dtype"] != dtype:
            problems.append(f"{what}: dtype {ref['dtype']}, expected {dtype}")
        if ndim and len(ref["shape"]) != ndim:
            problems.append(f"{what}: shape {ref['shape']}, expected {ndim}-d")
        if max_index is not None:
            arr = _Reader(path).array(ref)
            if arr.size and (arr.min() < 0 or arr.max() >= max_index):
                problems.append(f"{what}: indices outside [0, {max_index})")
        return ref

    b = m["brain"]
    check(b["indptr"], "brain.indptr", "int64", 1)
    if b["indptr"]["shape"] != [n + 1]:
        problems.append(f"brain.indptr has shape {b['indptr']['shape']}, expected [{n + 1}]")
    check(b["indices"], "brain.indices", "int64", 1, max_index=n)
    check(b["values"], "brain.values", "float32", 1)
    if b["indices"]["shape"] != b["values"]["shape"]:
        problems.append("brain.indices and brain.values differ in length")
    check(m["readout"]["indices"], "readout", "int64", 1, max_index=n)
    if m["kind"] == "pc":
        rp, rt = m["retina"]["params"], m["retina"]["tables"]
        if rp["mode"] == "hex":
            for k in ("indices", "pixels", "on"):
                if k not in rt:
                    problems.append(f"retina: hex mode needs table {k}")
            if "indices" in rt:
                check(rt["indices"], "retina.indices", "int64", 1, max_index=n)
            if "pixels" in rt:
                check(rt["pixels"], "retina.pixels", "int64", 1,
                      max_index=int(np.prod(rp["grid"])))
        elif rp["mode"] == "projection":
            for k in ("matrix_indptr", "matrix_indices", "matrix_values", "targets"):
                if k not in rt:
                    problems.append(f"retina: projection mode needs table {k}")
            if "targets" in rt:
                check(rt["targets"], "retina.targets", "int64", 1, max_index=n)
        else:
            problems.append(f"retina: unknown mode {rp['mode']!r}")
        if m.get("audition"):
            for k, v in m["audition"]["tables"].items():
                check(v, f"audition.{k}")
        n_out = ControlLayout.from_dict(m["layout"]).n
    elif m["kind"] == "body":
        body = m["body"]
        for k, v in body["proprio"]["tables"].items():
            check(v, f"proprio.{k}")
        if body["proprio"]["tables"]["matrix_indptr"]["shape"] != [n + 1]:
            problems.append("proprio.matrix_indptr does not match the neuron count")
        n_out = len(body["actuators"])
    else:
        return [f"unknown kind {m['kind']!r}"]
    if m.get("policy"):
        p = m["policy"]
        for k, v in p["tables"].items():
            check(v, f"policy.{k}")
        t = p["params"]["type"]
        if t == "linear" and p["tables"]["W"]["shape"][0] != n_out:
            problems.append("policy.W rows do not match the outputs")
        if t == "mlp":
            last = p["tables"][f"W{int(p['params']['n_layers']) - 1}"]["shape"][0]
            if last != n_out:
                problems.append("policy last layer does not match the outputs")
    return problems


def describe(path: str) -> str:
    m = read_manifest(path)
    b = m["brain"]
    pol = m["policy"]["params"]["type"] if m.get("policy") else "none"
    lines = [f"{m['name']} ({FORMAT} v{m['version']}, kind {m['kind']}, created {m['created']})",
             f"  brain: {b['n_neurons']:,} neurons, {b['n_synapses']:,} synapses, dt {b['dt']} ms, "
             f"{m['config']['brain_ms']:g} ms per observation"]
    if m["kind"] == "pc":
        rp = m["retina"]["params"]
        lines += [f"  retina: {rp['mode']}, grid {rp['grid']}",
                  f"  audition: {'yes' if m.get('audition') else 'no'}",
                  f"  features: {m['features']['n']}; actions: {m['actions']['n']} "
                  f"{ControlLayout.from_dict(m['layout']).names}"]
    else:
        body = m["body"]
        lines += [f"  observation: {body['obs_dim']} entries ({', '.join(body['obs_keys'])})",
                  f"  features: {m['features']['n']}; actuators: {len(body['actuators'])}"]
    lines.append(f"  policy: {pol}")
    if m.get("extra"):
        lines.append(f"  extra: {json.dumps(m['extra'])[:300]}")
    return "\n".join(lines)
