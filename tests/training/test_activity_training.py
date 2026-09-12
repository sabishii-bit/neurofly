"""Activity on the body env, through the CLI flags, and over gRPC."""
import argparse
import json
import sys

import numpy as np
import pytest

from neurofly_training import envs
from neurofly_training.experiments import apply_to_body_env
from tests.conftest import needs_data
from tests.fakes import FakePanic, FakeScreen
from tests.training.test_tools import _bars_recording


def test_body_env_activity(synthetic_cx):
    env = envs.make_env("forward", brain="synthetic", synthetic_n=1000)
    ns = argparse.Namespace(stimulate=[], silence=[], probe=None, activity_out="x.json",
                            activity_ws=None, activity_substeps=True)
    done = apply_to_body_env(env, ns)
    assert any("activity" in d for d in done) and env.activity is not None
    env.reset()
    env.step(env.action_space.sample())
    a = env.last_activity
    assert a is not None and len(a["steps"]) == env.substeps
    m = env.activity_map()
    assert m["positions"].shape == (env.cx.n, 3) and "motor" in m["populations"]
    assert env.watch_activity(False) == 0
    env.close()
    body = envs.make_env("forward", brain="none")
    assert "ignored" in apply_to_body_env(body, ns)[0]
    body.close()


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["neurofly-test"] + [str(a) for a in argv])
    module.main()


def test_watch_and_play_write_activity(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import play_pc, watch
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=12)
    _run(watch, ["--task", "rec/video.mp4", "--brain", "toy", "--keys", "a,d", "--policy",
                 "random", "--episodes", "1", "--max-steps", "4", "--activity-out", "pc.json"],
         monkeypatch)
    out = capsys.readouterr().out
    assert "activity written to pc.json" in out
    pc = json.load(open(tmp_path / "pc.json"))
    assert len(pc["steps"]) == 4 and len(pc["positions"]) == pc["n"]
    _run(watch, ["--task", "forward", "--brain", "synthetic", "--policy", "zero", "--episodes",
                 "1", "--max-steps", "20", "--every", "10", "--activity-out", "body.json"],
         monkeypatch)
    body = json.load(open(tmp_path / "body.json"))
    assert len(body["steps"]) == 2 and body["fps"] == 50.0
    monkeypatch.setattr(envs, "ScreenCapture", FakeScreen)
    monkeypatch.setattr(play_pc, "PanicKey", FakePanic)
    _run(play_pc, ["--region", "0,0,64,48", "--keys", "a", "--brain", "toy", "--synthetic-n",
                   "1000", "--policy", "fixed", "--dry-run", "--steps", "3", "--countdown", "0",
                   "--fps", "50", "--activity-out", "play.json", "--activity-ws", "127.0.0.1:0"],
         monkeypatch)
    out = capsys.readouterr().out
    assert "activity streaming on ws://127.0.0.1:" in out and "activity written to play.json" in out
    assert len(json.load(open(tmp_path / "play.json"))["steps"]) == 3


def test_core_run_and_serve_activity(tmp_path, monkeypatch, capsys):
    import io
    from neurofly_core import cli as core_cli
    import neurofly_core.io.controls as C
    import neurofly_core.io.video as V
    from neurofly_core.artifact import save_model
    from neurofly_core.controls import ControlLayout
    from neurofly_core.decode.linear import ControlDecoder
    from neurofly_training.build import build_model
    from neurofly_training.data.connectome import Connectome
    layout = ControlLayout(keys=["w"])
    model = build_model(Connectome.toy(1000).subset("central"), layout, brain_ms=5)
    model.policy = ControlDecoder(model.n_features, layout, seed=1)
    art = save_model(model, str(tmp_path / "art"))
    monkeypatch.setattr(V, "ScreenCapture", FakeScreen)
    monkeypatch.setattr(C, "PanicKey", FakePanic)
    core_cli.main(["run", art, "--region", "0,0,64,48", "--dry-run", "--steps", "2",
                   "--countdown", "0", "--fps", "50", "--activity-out", str(tmp_path / "r.json")])
    assert "activity written to" in capsys.readouterr().err
    assert len(json.load(open(tmp_path / "r.json"))["steps"]) == 2
    import base64
    frame = base64.b64encode(np.zeros((6, 8, 3), np.uint8).tobytes()).decode()
    lines = [json.dumps({"op": "activity", "on": True}),
             json.dumps({"op": "step", "frame": frame, "width": 8, "height": 6}),
             json.dumps({"op": "close"})]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    core_cli.main(["serve", art, "--activity-out", str(tmp_path / "s.json")])
    out = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert "activity" in out[2] and len(json.load(open(tmp_path / "s.json"))["steps"]) == 1


def test_grpc_activity_and_positions(synthetic_cx):
    grpc = pytest.importorskip("grpc")
    from neurofly_core.controls import ControlLayout
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    from neurofly_training.build import build_model
    model = build_model(synthetic_cx.subset("brain"), ControlLayout(keys=["w"]), brain_ms=5)
    seen = []
    server = make_server(model, "127.0.0.1:0", after_step=[lambda: seen.append(model.t)])
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        pos = stub.Positions(pb.Empty())
        assert pos.n == model.brain.n and len(pos.positions) == 3 * pos.n and pos.known
        assert len(pos.superclass) == pos.n and "readout" in pos.populations
        assert stub.Activity(pb.ActivityRequest(on=True, substeps=True)).n == model.brain.n
        frame = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        r = stub.Observe(pb.StepRequest(frame=frame.tobytes(), width=64, height=48))
        assert r.ok and r.activity.t == r.t and len(r.activity.indices) > 0
        assert len(r.activity.steps) == model.substeps + model.warmup_steps
        assert seen == [1]
        assert stub.Activity(pb.ActivityRequest(on=False)).n == 0
    finally:
        server.stop(0)


@needs_data
def test_real_connectome_has_soma_positions():
    from neurofly_training.data.connectome import Connectome
    from neurofly_training.envs import DEFAULT_DATA_DIR
    cx = Connectome.load(DEFAULT_DATA_DIR, verbose=False).subset("central")
    xyz, known = cx.positions()
    assert known.mean() > 0.8 and np.isfinite(xyz).all()
    span = xyz[known].max(0) - xyz[known].min(0)
    assert 200 < span.max() < 2000                      # the brain is under two millimetres
