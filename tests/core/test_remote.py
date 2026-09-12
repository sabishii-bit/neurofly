"""Hosting a brain: recording from inside the runtime, ping, tokens, per-client brains."""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from neurofly_core.artifact import save_model
from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.recording import Recorder
from neurofly_core.server import Session, serve_ws
from neurofly_training.build import build_model
from neurofly_training.data.connectome import Connectome

LAYOUT = ControlLayout(keys=["w"], mouse=True)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    model = build_model(Connectome.toy(1000, seed=0).subset("central"), LAYOUT, brain_ms=5)
    model.policy = ControlDecoder(model.n_features, LAYOUT, seed=1)
    return save_model(model, str(tmp_path_factory.mktemp("art") / "remote"))


def _req(op="observe", w=8, h=6, **extra):
    frame = np.full((h, w, 3), 30, np.uint8)
    return {"op": op, "frame": base64.b64encode(frame.tobytes()).decode(), "width": w,
            "height": h, **extra}


def test_recorder_writes_a_recording(tmp_path):
    rec = Recorder(str(tmp_path / "r"), LAYOUT, fps=5)
    rec.add(np.zeros((6, 8, 3), np.uint8), np.zeros((320, 1), np.float32), [1.0, 0.0, 0.0])
    rec.add(np.zeros((7, 9, 3), np.uint8), None, None)          # odd size is fitted to the first
    meta = rec.close()
    assert rec.close() == meta                                    # idempotent
    assert meta["n_frames"] == 2 and meta["size"] == [8, 6] and meta["audio"]["seconds"] > 0
    from neurofly_training.cli.train_imitation import load_recording
    video, layout, actions, m2 = load_recording(str(tmp_path / "r"))
    assert os.path.exists(video) and layout.names == LAYOUT.names
    assert actions.shape == (2, LAYOUT.n) and actions[0, 0] == 1 and actions[1].sum() == 0
    assert m2["source"] == "server"


def test_session_record_ping_hello(artifact, tmp_path):
    from neurofly_core.artifact import load_model
    s = Session(load_model(artifact))
    assert s.handle({"op": "ping"})["pong"] and s.handle({"op": "hello"})["n_actions"] == LAYOUT.n
    assert s.handle({"op": "record", "off": True}) == {"ok": True, "recording": False,
                                                       "n_frames": 0}
    r = s.handle({"op": "record", "path": str(tmp_path / "rec"), "fps": 10})
    assert r["recording"]
    a = s.handle(_req("observe", action=[1.0, 0.5, 0.0]))       # a human's action as the label
    b = s.handle(_req("step"))                                  # the policy's own action
    assert a["recording"] == 1 and b["recording"] == 2
    done = s.handle({"op": "record", "off": True})
    assert done["n_frames"] == 2 and not done["recording"]
    actions = np.load(tmp_path / "rec" / "actions.npy")
    assert actions.shape == (2, LAYOUT.n) and actions[0, 0] == 1.0 and actions[0, 1] == 0.5
    assert np.allclose(actions[1], b["action"])
    assert json.load(open(tmp_path / "rec" / "meta.json"))["n_frames"] == 2
    assert "recording" not in s.handle(_req("observe"))
    # close stops a recording that is still on
    s.handle({"op": "record", "path": str(tmp_path / "rec2")})
    s.handle(_req("observe"))
    assert s.handle({"op": "close"})["bye"] and s.recorder is None
    assert np.load(tmp_path / "rec2" / "actions.npy").shape == (1, LAYOUT.n)


def _serve(artifact, **kw):
    from neurofly_core.artifact import load_model
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(target=serve_ws, args=(None, "127.0.0.1", port),
                     kwargs=dict(loader=lambda: load_model(artifact), **kw), daemon=True).start()
    from websockets.sync.client import connect
    for _ in range(50):
        try:
            connect(f"ws://127.0.0.1:{port}").close()
            break
        except OSError:
            time.sleep(0.1)
    return port


def test_ws_token_and_per_client(artifact):
    pytest.importorskip("websockets")
    from websockets.sync.client import connect
    port = _serve(artifact, per_client=True, token="s3cret", origins=None)
    with connect(f"ws://127.0.0.1:{port}") as ws:              # no token: must say hello
        ws.send(json.dumps({"op": "info"}))
        assert "unauthorised" in json.loads(ws.recv())["error"]
    with connect(f"ws://127.0.0.1:{port}") as ws:
        ws.send(json.dumps({"op": "hello", "token": "s3cret"}))
        hello = json.loads(ws.recv())
        assert hello["ready"] and hello["per_client"]
    a = connect(f"ws://127.0.0.1:{port}/?token=s3cret")
    b = connect(f"ws://127.0.0.1:{port}/?token=s3cret")
    with a, b:
        assert json.loads(a.recv())["ready"] and json.loads(b.recv())["ready"]
        for _ in range(3):
            a.send(json.dumps(_req("observe")))
            ta = json.loads(a.recv())["t"]
        b.send(json.dumps(_req("observe")))
        tb = json.loads(b.recv())["t"]
        assert ta == 3 and tb == 1                            # two brains, not one
        a.send(json.dumps({"op": "close"}))
        assert json.loads(a.recv())["bye"]


def test_ws_shared_needs_a_model():
    with pytest.raises(ValueError):
        serve_ws(None, per_client=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_node_ws_client(artifact):
    node_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "node")
    if not os.path.exists(os.path.join(node_dir, "dist", "ws.js")):
        pytest.skip("the Node package is not built (npm run build)")
    out = subprocess.run(["node", "test_ws.js", artifact, sys.executable], cwd=node_dir,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ok" in out.stdout and "per_client: true" in out.stdout
