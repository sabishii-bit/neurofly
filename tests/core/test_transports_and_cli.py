import base64
import io
import json
import socket
import sys
import threading
import time

import numpy as np
import pytest

from neurofly_core import cli as core_cli
from neurofly_core.artifact import save_model
from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.experiments import ProbeLog, add_experiment_args, apply_experiments
from neurofly_core.selection import parse_spec, resolve
from neurofly_core.server import Session, decode_frame, selection_from, serve_stdio, serve_ws
from neurofly_training.build import build_model
from tests.fakes import FakePanic, FakeScreen

LAYOUT = ControlLayout(keys=["w"], mouse=True)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    from neurofly_training.data.connectome import Connectome
    cx = Connectome.toy(1200, seed=0).subset("central")
    model = build_model(cx, LAYOUT, brain_ms=5, name="cli")
    model.policy = ControlDecoder(model.n_features, LAYOUT, seed=2)
    return save_model(model, str(tmp_path_factory.mktemp("art") / "cli"))


def _step_line(w=8, h=6):
    frame = np.zeros((h, w, 3), np.uint8)
    return json.dumps({"op": "step", "frame": base64.b64encode(frame.tobytes()).decode(),
                       "width": w, "height": h})


def test_decode_png_and_selection_errors():
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.full((5, 7, 3), 9, np.uint8)).save(buf, format="PNG")
    f = decode_frame({"frame": base64.b64encode(buf.getvalue()).decode(), "format": "png"})
    assert f.shape == (5, 7, 3) and f[0, 0, 0] == 9
    with pytest.raises(ValueError):
        selection_from({"op": "x"})
    with pytest.raises(ValueError):
        selection_from({"type_re": "a", "name": "b"})
    with pytest.raises(ValueError):
        resolve({"indices": [99]}, n=10)
    with pytest.raises(ValueError):
        resolve({"ids": [5]}, n=10)
    with pytest.raises(ValueError):
        resolve({"ids": [5]}, n=10, ids=np.array([1, 2]))
    with pytest.raises(ValueError):
        resolve({"type_re": "x"}, n=10)
    with pytest.raises(ValueError):
        resolve({"superclass": "x"}, n=10)
    with pytest.raises(ValueError):
        resolve({}, n=10)
    with pytest.raises(ValueError):
        parse_spec("nokey=1")
    assert parse_spec("indices=1,2:5") == ({"indices": [1, 2]}, 5.0)


def test_serve_stdio(monkeypatch, capsys, artifact):
    from neurofly_core.artifact import load_model
    model = load_model(artifact)
    lines = ['{"op": "info"}', "not json", "", _step_line(), '{"op": "close"}', '{"op": "info"}']
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    serve_stdio(model)
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert out[0]["ready"] and out[1]["ok"] and "bad JSON" in out[2]["error"]
    assert out[3]["keys"] is not None and out[4]["bye"] and len(out) == 5   # stopped at close


def test_serve_ws(artifact):
    pytest.importorskip("websockets")
    from websockets.sync.client import connect
    from neurofly_core.artifact import load_model
    model = load_model(artifact)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(target=serve_ws, args=(model, "127.0.0.1", port), daemon=True).start()
    for _ in range(50):
        try:
            ws = connect(f"ws://127.0.0.1:{port}")
            break
        except OSError:
            time.sleep(0.1)
    with ws:
        assert json.loads(ws.recv())["ready"]
        ws.send("garbage")
        assert "bad JSON" in json.loads(ws.recv())["error"]
        ws.send(_step_line())
        assert json.loads(ws.recv())["ok"]
        ws.send('{"op": "close"}')
        assert json.loads(ws.recv())["bye"]


def test_experiment_helpers(artifact):
    import argparse
    from neurofly_core.artifact import load_model
    model = load_model(artifact)
    p = add_experiment_args(argparse.ArgumentParser())
    args = p.parse_args(["--stimulate", "name=readout:5", "--silence", "type_re=^L1$",
                         "--probe", "name=readout", "--probe-out", "x.npz"])
    done = apply_experiments(model, args)
    assert len(done) == 3 and model.probe_idx is not None
    with pytest.raises(SystemExit):
        apply_experiments(model, p.parse_args(["--stimulate", "name=readout"]))
    log = ProbeLog(model, None)
    model.reset()
    model.observe(np.zeros((6, 8, 3), np.uint8))
    log.record()
    assert log.save() is None and len(log.t) == 1


def test_core_cli_info_validate_serve(artifact, monkeypatch, capsys, tmp_path):
    core_cli.main(["info", artifact])
    assert "kind pc" in capsys.readouterr().out
    core_cli.main(["validate", artifact])
    assert "ok" in capsys.readouterr().out
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text('{"format": "nope"}')
    with pytest.raises(SystemExit):
        core_cli.main(["validate", str(tmp_path / "bad")])
    assert "not a neurofly-artifact" in capsys.readouterr().out
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"op": "close"}\n'))
    core_cli.main(["serve", artifact, "--stimulate", "name=readout:1"])
    assert json.loads(capsys.readouterr().out.splitlines()[0])["ready"]


def test_core_cli_run_with_fake_devices(artifact, monkeypatch, capsys):
    import neurofly_core.io.controls as C
    import neurofly_core.io.video as V
    monkeypatch.setattr(V, "ScreenCapture", FakeScreen)
    monkeypatch.setattr(C, "PanicKey", FakePanic)
    core_cli.main(["run", artifact, "--region", "0,0,64,48", "--dry-run", "--steps", "3",
                   "--countdown", "0", "--fps", "50", "--probe", "name=readout"])
    err = capsys.readouterr().err
    assert "stopped after 3 steps" in err and "focus guard off" in err


def test_core_cli_run_refuses_without_policy(tmp_path, monkeypatch):
    from neurofly_training.data.connectome import Connectome
    model = build_model(Connectome.toy(1000).subset("central"), LAYOUT)
    path = save_model(model, str(tmp_path / "nopol"))
    with pytest.raises(SystemExit):
        core_cli.main(["run", path, "--countdown", "0"])


def test_session_edge_cases(artifact):
    from neurofly_core.artifact import load_model
    model = load_model(artifact)
    s = Session(model)
    assert s.handle({"op": "body_step", "obs": [0.0]})["ok"] is False   # pc artifact
    layers = [{"W": np.ones((2, model.n_features)).tolist(), "b": [0, 0]},
              {"W": np.ones((LAYOUT.n, 2)).tolist(), "b": [0] * LAYOUT.n}]
    req = {"op": "set_policy", "type": "mlp", "layers": layers,
           "obs_mean": [0] * model.n_features, "obs_var": [1] * model.n_features}
    assert s.handle(req)["type"] == "mlp"
    bad = {"op": "set_policy", "type": "mlp", "layers": [{"W": [[1.0]], "b": [0.0]}]}
    assert s.handle(bad)["ok"] is False
    assert s.handle({"op": "set_policy", "type": "other"})["ok"] is False
