"""The body server: the fly driven over JSON, with poses pushed to viewers."""
import base64
import json
import sys
import threading

import numpy as np
import pytest

from neurofly_training.body.actuators import ACTUATOR_NAMES, leg_actuator_indices
from neurofly_training.body.server import BodySession, MAX_STEPS


@pytest.fixture(scope="module")
def session():
    s = BodySession("forward", seed=0, every=10, realtime=False)
    yield s
    s.close()


def test_info_describes_the_body(session):
    info = session.handle({"op": "hello"})
    assert info["kind"] == "body-server" and info["task"] == "forward"
    assert info["actuators"] == ACTUATOR_NAMES and info["n_actions"] == 59
    assert info["actuators"][0] == "adhere_claw_T1_left" and len(info["rest"]) == 59
    assert "walker/thorax" in info["bodies"] and info["fps"] == 50.0
    assert info["legs"] == ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
    jp = info["obs_slices"]["walker/joints_pos"]
    assert jp[1] - jp[0] == 85
    assert info["brain"] is None and info["tasks"] == ["ball", "forward"]
    assert info["time_limit"] is None                              # a hosted fly lives on


def test_step_with_named_actuators_and_legs(session):
    session.handle({"op": "reset"})
    r = session.handle({"op": "step", "steps": 20, "actuators": {"coxa_T1_left": 0.4},
                        "legs": {"T3R": {"femur": -0.2, "claw": 1.0}}})
    assert r["ok"] and r["t"] == 20 and r["steps"] == 20 and not r["done"]
    assert len(r["obs"]) == 286 and len(r["pose"]) == 7 * len(session.bodies)
    a = np.asarray(r["action"])
    rest = np.asarray(session.rest)
    coxa, femur = ACTUATOR_NAMES.index("coxa_T1_left"), ACTUATOR_NAMES.index("femur_T3_right")
    assert a[coxa] == pytest.approx(rest[coxa] + 0.4)
    assert a[femur] == pytest.approx(rest[femur] - 0.2)
    assert a[leg_actuator_indices("T3", "R")[-1]] == 1.0
    idle = [i for i in range(59) if i not in (ACTUATOR_NAMES.index("coxa_T1_left"),
                                                 ACTUATOR_NAMES.index("femur_T3_right"),
                                                 leg_actuator_indices("T3", "R")[-1])]
    assert np.allclose(a[idle], rest[idle])
    full = session.handle({"op": "step", "action": [0.0] * 59})
    assert full["ok"] and full["t"] == 21 and full["action"] == [0.0] * 59
    assert "error" in session.handle({"op": "step", "action": [0.0] * 3})
    assert "unknown actuator" in session.handle({"op": "step", "actuators": {"wing": 1}})["error"]
    assert "unknown leg" in session.handle({"op": "step", "legs": {"T9L": {}}})["error"]
    assert "no brain" in session.handle({"op": "step", "brain": True})["error"]
    assert "steps must be" in session.handle({"op": "step", "steps": MAX_STEPS + 1})["error"]


def test_gait_pushes_poses_and_keeps_the_fly_up(session):
    session.handle({"op": "reset"})
    got = []
    session.listeners.append(lambda t, row: got.append((t, row)))
    r = session.handle({"op": "gait", "steps": 300, "stride_hz": 2.0})
    session.listeners.pop()
    assert r["ok"] and r["steps"] == 300 and r["gait"]["stride_hz"] == 2.0 and not r["done"]
    assert len(got) == 30 and got[0][0] + 29 == got[-1][0]         # one pose per 10 steps
    thorax = session.bodies.index("walker/thorax")
    assert r["pose"][7 * thorax + 2] > 0.1                          # still standing
    obs = session.handle({"op": "observe"})
    assert obs["t"] == 300 and obs["pose"] == r["pose"]
    r = session.handle({"op": "step", "steps": 800})                # past the training task's 2 s
    assert r["ok"] and r["t"] == 1100 and not r["done"]
    limited = BodySession("forward", seed=0, realtime=False, time_limit=0.1)
    try:
        assert limited.info()["time_limit"] == 0.1
        assert limited.handle({"op": "step", "steps": 100})["done"]
    finally:
        limited.close()


def test_set_pose_and_replay_are_kinematic(session):
    session.handle({"op": "reset"})
    r = session.handle({"op": "set_pose", "joints": {"coxa_T1_left": 0.9}})
    assert r["ok"] and r["t"] == 0
    m = session.env.physics.model
    j = m.name2id("walker/coxa_T1_left", "joint")
    assert r["qpos"][int(m.jnt_qposadr[j])] == pytest.approx(0.9)
    assert "error" in session.handle({"op": "set_pose", "qpos": [0.0] * 3})
    r = session.handle({"op": "replay", "source": "gait", "steps": 100})
    assert r["ok"] and r["frames"] == 10 and r["steps"] == 100 and r["fps"] == 50.0
    assert "source must be" in session.handle({"op": "replay", "source": "dance"})["error"]


def test_frame_reset_and_misc(session):
    r = session.handle({"op": "frame", "width": 64, "height": 48})
    png = base64.b64decode(r["png"])
    assert r["ok"] and png[:8] == b"\x89PNG\r\n\x1a\n" and r["width"] == 64
    assert session.handle({"op": "ping"})["pong"]
    assert session.handle({"op": "poses", "on": False}) == {"ok": True, "on": False}
    assert "unknown op" in session.handle({"op": "fly"})["error"]
    assert "unknown task" in session.handle({"op": "reset", "task": "swim"})["error"]
    r = session.handle({"op": "reset", "seed": 3})
    assert r["ok"] and r["t"] == 0 and len(r["obs"]) == 286
    assert session.handle({"op": "close"})["bye"]


def test_brain_acts_when_loaded():
    from neurofly_training.envs import make_env
    from neurofly_training.export import body_model_from_env
    env = make_env("forward", seed=0, brain="synthetic", synthetic_n=1200)
    model = body_model_from_env(env, "walk")
    from neurofly_training.body.decoder import LinearDecoder
    from neurofly_core.body import ActuatorDecoder
    dec = LinearDecoder(env.pops, env.readout_idx, n_actions=59, seed=0)
    W = np.zeros((59, model.n_features), np.float32)
    for acts, feats, w in dec.blocks:
        W[np.ix_(acts, feats)] = w
    model.policy = ActuatorDecoder(model.n_features, model.layout, W=W, b=np.zeros(59))
    env.close()
    s = BodySession("forward", seed=0, brain=model, realtime=False)
    try:
        info = s.info()
        assert info["brain"] and info["brain_has_policy"]
        r = s.handle({"op": "step", "brain": True, "steps": 5})
        assert r["ok"] and r["t"] == 5 and "spikes" in r or r["ok"]
        assert len(r["action"]) == 59
    finally:
        s.close()
    with pytest.raises(ValueError):
        BodySession("forward", brain=object())


def _serve(session):
    from neurofly_training.body.server import serve_ws
    port = {}
    ev = threading.Event()

    def ready(p):
        port["p"] = p
        ev.set()
    threading.Thread(target=serve_ws, args=(session, "127.0.0.1", 0), kwargs={"ready": ready},
                     daemon=True).start()
    assert ev.wait(30)
    return port["p"]


def test_ws_drives_and_pushes_poses():
    pytest.importorskip("websockets")
    from websockets.sync.client import connect
    session = BodySession("forward", seed=0, every=10, realtime=False)
    port = _serve(session)
    viewer = connect(f"ws://127.0.0.1:{port}")
    header = json.loads(viewer.recv())
    assert header["ready"] and header["bodies"] == session.bodies and header["fps"] == 50.0
    with connect(f"ws://127.0.0.1:{port}") as ctl:
        json.loads(ctl.recv())
        ctl.send(json.dumps({"op": "poses", "on": False}))       # a controller need not watch
        assert json.loads(ctl.recv()) == {"ok": True, "on": False}
        ctl.send(json.dumps({"op": "gait", "steps": 50}))
        r = json.loads(ctl.recv())
        assert r["ok"] and r["steps"] == 50                     # no pose pushed in between
    pushed = [json.loads(viewer.recv(timeout=5)) for _ in range(5)]
    assert all("pose" in m and len(m["pose"]) == 7 * len(session.bodies) for m in pushed)
    assert [m["t"] for m in pushed] == list(range(pushed[0]["t"], pushed[0]["t"] + 5))
    viewer.close()


def test_stdio_returns_poses_inline(monkeypatch, capsys):
    from neurofly_training.body.server import serve_stdio
    import io
    session = BodySession("forward", seed=0, every=10, realtime=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        json.dumps({"op": "step", "steps": 20}) + "\n" + "not json\n"
        + json.dumps({"op": "close"}) + "\n"))
    serve_stdio(session)
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["ready"] and lines[0]["kind"] == "body-server"
    assert lines[1]["ok"] and len(lines[1]["poses"]) == 2
    assert "bad JSON" in lines[2]["error"] and lines[3]["bye"]


def test_cli_stdio(monkeypatch, capsys):
    from neurofly_training.cli import body_serve
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"op": "close"}) + "\n"))
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "--stdio", "--no-realtime"])
    body_serve.main()
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[0])["ready"] and json.loads(out[-1])["bye"]


@pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node not installed")
def test_node_body_client():
    import os
    import subprocess
    node_dir = os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "node")
    if not os.path.exists(os.path.join(node_dir, "dist", "body.js")):
        pytest.skip("the Node package is not built (npm run build)")
    out = subprocess.run(["node", "test_body.js", sys.executable], cwd=node_dir,
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ok" in out.stdout and "pushed poses" in out.stdout
