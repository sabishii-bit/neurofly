"""Watching the whole brain: which neurons fired on each step, and where they are.

A ``SpikeAccumulator`` collects every neuron's spikes over one observation (all the
brain substeps that make it up). A model that has one produces ``last_activity``:

    {"indices": [neurons that fired], "counts": [how many times each],
     "steps": [[...], ...]}          # per substep, only when asked for

Two sinks turn that into a picture. ``ActivityLog`` writes one JSON file with the
neuron positions and every step's spikes, and ``ActivityBroadcaster`` pushes the same
over a WebSocket as it happens; ``examples/brain_viewer.html`` plays either. Both take
any *source* with ``activity_map()`` and ``last_activity`` (a ``BrainModel`` or a
brain-in-the-loop body environment).

    model.watch_activity(True)
    log = ActivityLog(model, "activity.json", fps=10)
    ... model.observe(frame); log.record() ...
    log.save()
"""
from __future__ import annotations

import json
import threading

import numpy as np
import torch


class SpikeAccumulator:
    """Per-observation spike record for every neuron in the brain."""

    def __init__(self, n: int, substeps: bool = False, device="cpu"):
        self.n, self.substeps = int(n), bool(substeps)
        self._counts = torch.zeros(self.n, dtype=torch.int32, device=device)
        self._steps: list[np.ndarray] = []

    def begin(self) -> None:
        self._counts.zero_()
        self._steps = []

    def add(self, spikes: torch.Tensor) -> None:
        self._counts += spikes.to(torch.int32)
        if self.substeps:
            self._steps.append(torch.nonzero(spikes).flatten().cpu().numpy().astype(np.int64))

    def finish(self) -> dict:
        counts = self._counts.cpu().numpy()
        idx = np.flatnonzero(counts).astype(np.int64)
        out = {"indices": idx, "counts": counts[idx].astype(np.int64)}
        if self.substeps:
            out["steps"] = self._steps
        return out


def activity_payload(t: int, activity: dict) -> dict:
    """The JSON form of one step's activity."""
    out = {"t": int(t), "indices": activity["indices"].tolist(),
           "counts": activity["counts"].tolist()}
    if "steps" in activity:
        out["steps"] = [s.tolist() for s in activity["steps"]]
    return out


def map_payload(source, fps: float | None = None) -> dict:
    """The header a viewer needs: every neuron's position, whether it is a real soma,
    its superclass, and the named populations."""
    m = source.activity_map()
    ids = getattr(source, "neuron_ids", None)
    if ids is None and hasattr(source, "cx"):
        ids = source.cx.neurons["bodyId"].values
    out = {"n": int(m["n"]), "unit": m.get("unit", "micrometre"),
           "ids": None if ids is None else [int(i) for i in ids],
           "positions": None if m["positions"] is None else m["positions"].tolist(),
           "known": None if m.get("known") is None else m["known"].tolist(),
           "superclass": m.get("superclass"),
           "populations": {k: v.tolist() for k, v in m.get("populations", {}).items()}}
    if fps is not None:
        out["fps"] = float(fps)
    return out


class ActivityLog:
    """Every step's activity into one JSON file the viewer can play back."""

    def __init__(self, source, path: str | None, fps: float = 10.0):
        self.source, self.path, self.fps = source, path, fps
        self.steps: list[dict] = []

    def record(self) -> None:
        """Keep this step's activity (in memory; ``save`` writes it if there is a path)."""
        a = self.source.last_activity
        if a is None:
            return
        self.steps.append(activity_payload(len(self.steps), a))

    def save(self) -> str | None:
        if not self.path or not self.steps:
            return None
        with open(self.path, "w") as f:
            json.dump({**map_payload(self.source, self.fps), "steps": self.steps}, f)
        return self.path


class ActivityBroadcaster:
    """The same over a WebSocket: every client gets the map on connect, then one message
    per step. Clients may send anything; it is ignored."""

    def __init__(self, source, host: str = "127.0.0.1", port: int = 8767, fps: float = 10.0):
        from websockets.sync.server import serve
        self.source = source
        self.header = json.dumps(map_payload(source, fps))
        self._clients: set = set()
        self._lock = threading.Lock()
        self._server = serve(self._handler, host, port, max_size=64 * 1024 * 1024)
        self.host = host
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.t = 0

    def _handler(self, ws) -> None:
        ws.send(self.header)
        with self._lock:
            self._clients.add(ws)
        try:
            for _ in ws:
                pass
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(ws)

    @property
    def n_clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def record(self) -> None:
        a = self.source.last_activity
        if a is None:
            return
        msg = json.dumps(activity_payload(self.t, a))
        self.t += 1
        with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                ws.send(msg)
            except Exception:
                with self._lock:
                    self._clients.discard(ws)

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2.0)


def parse_address(spec: str, default_host: str = "127.0.0.1") -> tuple[str, int]:
    host, _, port = str(spec).rpartition(":")
    return host or default_host, int(port)
