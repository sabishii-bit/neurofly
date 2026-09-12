"""A trained controller as a service: JSON lines over stdio, WebSocket, or gRPC.

Any language that can start a process and write lines can use it. One request
per line, one response per line:

    {"op": "info"}
    {"op": "reset"}
    {"op": "step", "frame": "<base64>", "width": 320, "height": 240,
     "audio": "<base64 of float32 little-endian samples>", "sample_rate": 16000,
     "channels": 1, "reward": 0.0}
    {"op": "observe", ...the same fields...}          -> features instead of controls
    {"op": "set_policy", "type": "linear", "W": [[...]], "b": [...]}
    {"op": "set_policy", "type": "mlp", "layers": [{"W": [[...]], "b": [...]}, ...],
     "activation": "tanh", "obs_mean": [...], "obs_var": [...]}
    {"op": "save", "path": "artifacts/name", "name": "name"}
    {"op": "body_step", "obs": [...], "reward": 0.0}       body artifacts: observation in,
    {"op": "body_observe", "obs": [...]}                   actuators (or features) out
    {"op": "stimulate", "type_re": "^PPL1", "mv": 20}     experiments on the brain:
    {"op": "silence", "superclass": "descending_neuron"}   see neurofly_core.selection
    {"op": "probe", "name": "readout"}                     -> "probe" in later responses
    {"op": "clear"}                                        undo stimulation and silencing
    {"op": "close"}

``frame`` is raw RGB bytes (row-major, 3 bytes per pixel) unless ``"format"``
is ``"png"`` or ``"jpeg"``, in which case it is the encoded image. ``audio`` and
``reward`` are optional. The response to ``step``:

    {"ok": true, "t": 12, "keys": ["w"], "buttons": [], "dx": 3.1, "dy": -0.4,
     "scroll": 0.0, "held": ["w"], "action": [...], "spikes": 812}

Errors come back as ``{"ok": false, "error": "..."}``. Logs go to stderr;
stdout carries only the protocol. The gRPC flavour is ``neurofly_core.rpc``.
See artifact/SPEC.md for the full contract.
"""
from __future__ import annotations

import base64
import io
import json
import sys

import numpy as np

from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.decode.mlp import MLPPolicy
from neurofly_core.model import Model
from neurofly_core.selection import KEYS as SELECTION_KEYS


def decode_frame(req: dict) -> np.ndarray:
    raw = base64.b64decode(req["frame"])
    return frame_from_bytes(raw, req.get("format", "rgb"), req.get("width"), req.get("height"))


def frame_from_bytes(raw: bytes, fmt: str = "rgb", width=None, height=None) -> np.ndarray:
    if fmt in ("rgb", "", None):
        w, h = int(width), int(height)
        if len(raw) != w * h * 3:
            raise ValueError(f"frame has {len(raw)} bytes, expected {w * h * 3} for {w}x{h} RGB")
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def decode_audio(req: dict) -> np.ndarray | None:
    if not req.get("audio"):
        return None
    return audio_from_bytes(base64.b64decode(req["audio"]), int(req.get("channels", 1)))


def audio_from_bytes(raw: bytes, channels: int = 1) -> np.ndarray | None:
    if not raw:
        return None
    return np.frombuffer(raw, dtype="<f4").reshape(-1, max(1, channels))


def selection_from(req: dict) -> dict:
    sel = {k: req[k] for k in SELECTION_KEYS if k in req}
    if len(sel) != 1:
        raise ValueError(f"give exactly one of {SELECTION_KEYS}")
    return sel


class Session:
    """One model, driven by request dicts. ``handle`` never raises; the array-level
    methods (``step_arrays`` and friends) are what the gRPC service calls."""

    def __init__(self, model: Model):
        self.model = model
        self.model.reset()

    # --- array-level API --------------------------------------------------------------

    def info(self) -> dict:
        m = self.model
        out = {"ok": True, "kind": m.kind, "name": m.config.name, "n_neurons": m.brain.n,
               "n_features": m.n_features, "n_actions": m.n_actions,
               "controls": m.layout.names, "layout": m.layout.to_dict(),
               "brain_ms": m.config.brain_ms, "has_policy": m.policy is not None,
               "has_annotations": m.neuron_types is not None,
               "populations": {k: int(len(v)) for k, v in m.populations().items()}}
        if m.kind == "pc":
            out.update({"has_audition": m.audition is not None,
                        "sample_rate": m.audition.sample_rate if m.audition else None,
                        "retina_grid": list(m.retina.grid)})
        else:
            out.update({"has_audition": False, "sample_rate": None, "retina_grid": [],
                        "n_obs": m.n_obs,
                        "obs": {"keys": m.obs_keys,
                                "slices": {k: list(v) for k, v in m.obs_slices.items()}}})
        return out

    def body_step_arrays(self, obs: np.ndarray, reward: float = 0.0,
                         observe_only: bool = False) -> dict:
        m = self.model
        if m.kind != "body":
            raise ValueError("body_step needs a body artifact")
        if observe_only or m.policy is None:
            feats = m.observe(obs, reward)
            out = {"ok": True, "t": m.t, "features": feats.tolist(), "spikes": m.last_spikes}
        else:
            action, info = m.step(obs, reward)
            out = {"ok": True, "t": info["t"], "spikes": info["spikes"], "action": info["action"]}
        if m.last_probe is not None:
            out["probe"] = {"spikes": m.last_probe["spikes"].tolist(),
                            "rates": m.last_probe["rates"].tolist()}
        return out

    def step_arrays(self, frame: np.ndarray, audio: np.ndarray | None = None,
                    reward: float = 0.0, observe_only: bool = False) -> dict:
        m = self.model
        if m.kind != "pc":
            raise ValueError("step / observe need a PC artifact; use body_step for a body one")
        if observe_only or m.policy is None:
            feats = m.observe(frame, audio, reward)
            out = {"ok": True, "t": m.t, "features": feats.tolist(), "spikes": m.last_spikes}
        else:
            state, info = m.step(frame, audio, reward)
            out = {"ok": True, "t": info["t"], "spikes": info["spikes"],
                   "action": info["action"], "held": info["held"]}
            out.update(state.to_dict())
        if m.last_probe is not None:
            out["probe"] = {"spikes": m.last_probe["spikes"].tolist(),
                            "rates": m.last_probe["rates"].tolist()}
        return out

    def policy_from(self, req: dict):
        """A policy from JSON arrays, checked against this model's shapes."""
        m = self.model
        kind = req.get("type", "linear")
        if kind == "linear":
            W, b = np.asarray(req["W"], np.float64), np.asarray(req["b"], np.float64)
            if W.shape != (m.n_actions, m.n_features) or b.shape != (m.n_actions,):
                raise ValueError(f"W must be {m.n_actions}x{m.n_features} and b {m.n_actions}")
            if m.kind == "body":
                from neurofly_core.body import ActuatorDecoder
                return ActuatorDecoder(m.n_features, m.layout, W=W, b=b)
            return ControlDecoder(m.n_features, m.layout, W=W, b=b)
        if kind == "mlp":
            layers = [(np.asarray(L["W"], np.float64), np.asarray(L["b"], np.float64))
                      for L in req["layers"]]
            if layers[0][0].shape[1] != m.n_features:
                raise ValueError(f"the first layer must take {m.n_features} features")
            return MLPPolicy(m.layout, layers, activation=req.get("activation", "tanh"),
                             obs_mean=req.get("obs_mean"), obs_var=req.get("obs_var"),
                             obs_clip=float(req.get("obs_clip", 10.0)),
                             obs_eps=float(req.get("obs_eps", 1e-8)))
        raise ValueError(f"unknown policy type {kind!r}")

    def save(self, path: str, name: str | None = None, extra: dict | None = None) -> str:
        from neurofly_core.artifact import save_model
        if name:
            self.model.config.name = str(name)
        return save_model(self.model, path, extra=extra)

    # --- JSON API -----------------------------------------------------------------------

    def handle(self, req: dict) -> dict:
        try:
            op = req.get("op")
            m = self.model
            if op == "info":
                return self.info()
            if op == "reset":
                m.reset()
                return {"ok": True}
            if op in ("body_step", "body_observe"):
                obs = np.asarray(req["obs"], dtype=np.float32)
                return self.body_step_arrays(obs, float(req.get("reward", 0.0)),
                                             observe_only=(op == "body_observe")
                                             or bool(req.get("observe_only")))
            if op in ("step", "observe"):
                observe = (op == "observe") or bool(req.get("observe_only"))
                return self.step_arrays(decode_frame(req), decode_audio(req),
                                        float(req.get("reward", 0.0)), observe_only=observe)
            if op == "set_policy":
                m.policy = self.policy_from(req)
                return {"ok": True, "type": m.policy.kind}
            if op == "save":
                path = self.save(req["path"], req.get("name"), req.get("extra"))
                return {"ok": True, "path": path}
            if op == "stimulate":
                n = m.stimulate(selection_from(req), float(req["mv"]))
                return {"ok": True, "n": n}
            if op == "silence":
                return {"ok": True, "n": m.silence(selection_from(req))}
            if op == "probe":
                if req.get("off"):
                    m.probe(None)
                    return {"ok": True, "n": 0}
                return {"ok": True, "n": m.probe(selection_from(req))}
            if op == "clear":
                m.clear()
                return {"ok": True}
            if op == "select":
                idx = m.select(selection_from(req))
                return {"ok": True, "n": int(len(idx)), "indices": idx.tolist()}
            if op == "close":
                return {"ok": True, "bye": True}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # the protocol must survive bad input
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def serve_stdio(model: Model) -> None:
    """Read JSON lines on stdin, write JSON lines on stdout, until close or EOF."""
    session = Session(model)
    out = sys.stdout
    print(json.dumps({"ok": True, "ready": True, **session.info()}), file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), file=out, flush=True)
            continue
        resp = session.handle(req)
        print(json.dumps(resp), file=out, flush=True)
        if resp.get("bye"):
            break


def serve_ws(model: Model, host: str = "127.0.0.1", port: int = 8765) -> None:
    """The same protocol over a WebSocket (needs the ``websockets`` package)."""
    import asyncio

    import websockets

    session = Session(model)

    async def handler(ws):
        await ws.send(json.dumps({"ok": True, "ready": True, **session.info()}))
        async for msg in ws:
            try:
                req = json.loads(msg)
            except json.JSONDecodeError as e:
                await ws.send(json.dumps({"ok": False, "error": f"bad JSON: {e}"}))
                continue
            resp = session.handle(req)
            await ws.send(json.dumps(resp))
            if resp.get("bye"):
                break

    async def main():
        async with websockets.serve(handler, host, port, max_size=64 * 1024 * 1024):
            print(f"neurofly-core listening on ws://{host}:{port}", file=sys.stderr, flush=True)
            await asyncio.Future()

    asyncio.run(main())
