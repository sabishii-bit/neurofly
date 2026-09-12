"""Live poses over a WebSocket, for a viewer watching the simulation as it runs.

    caster = PoseBroadcaster(lambda: env.physics, fps=50, host="127.0.0.1", port=8766)
    ...
    caster.broadcast()     # after every rendered step
    caster.close()

Every client gets one header ``{"bodies": [...], "fps": 50, "units": "cm", "up": "z"}``
on connect, then one ``{"t": n, "pose": [x, y, z, qw, qx, qy, qz, ...]}`` per frame,
in the same order as ``poses.json``. ``examples/three_viewer.html?ws=ws://host:port``
consumes it.
"""
from __future__ import annotations

import json
import threading

from neurofly_training.body.export import PoseRecorder


class PoseBroadcaster:
    def __init__(self, physics, fps: float, host: str = "127.0.0.1", port: int = 8766,
                 skip_bodies=("world",)):
        from websockets.sync.server import serve
        self.recorder = PoseRecorder(physics, fps, skip_bodies)
        self.header = json.dumps({"bodies": self.recorder.bodies, "fps": float(fps),
                                  "units": "cm", "up": "z"})
        self._clients: set = set()
        self._lock = threading.Lock()
        self._server = serve(self._handler, host, port)
        self.port = self._server.socket.getsockname()[1]
        self.host = host
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.t = 0

    def _handler(self, ws) -> None:
        ws.send(self.header)
        with self._lock:
            self._clients.add(ws)
        try:
            for _ in ws:          # ignore anything the client says; keep the socket open
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

    def broadcast(self) -> None:
        """Read the current pose from the simulation and send it to every client."""
        self.recorder.record()
        row = self.recorder.frames.pop()
        msg = json.dumps({"t": self.t, "pose": row})
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
