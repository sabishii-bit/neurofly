"""Touch channels, one-step pulses on named populations (the giant fibre, the clock), and
those through the artifact, the session and gRPC, on the PC and the body model."""
import base64

import numpy as np
import pytest

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.server import Session
from neurofly_training.build import build_model, build_touch
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations

LAYOUT = ControlLayout(keys=["w"])


@pytest.fixture(scope="module")
def cx():
    return Connectome.toy(1500, seed=0).subset("central")


def _frame():
    return np.full((6, 8, 3), 60, np.uint8)


def test_populations_touch_clock_giant_fibre(synthetic_cx, cx):
    full = Populations(synthetic_cx)
    assert len(full.giantfibre) == 2 and len(full.clock) == 4
    assert {"head:labellar bristle", "head:grooming"} <= set(full.touch_types)
    assert any(k.startswith("leg:") for k in full.touch_types)
    head = Populations(cx)
    assert len(head.giantfibre) == 0 and len(head.clock) == 4     # the giant fibre is in the cord
    assert all(k.startswith("head:") for k in head.touch_types)
    assert "giant fibre 2" in full.summary() and len(full.readout("clock")) == 4


def test_touch_channels_pick_groups(cx):
    pops = Populations(cx)
    enc = build_touch(pops, cx.n, ["head:grooming", "hit"], gain=10.0)
    grooming = set(pops.touch_types["head:grooming"].tolist())
    assert set(np.flatnonzero(enc([1, 0]).cpu().numpy())) == grooming
    other = set(np.flatnonzero(enc([0, 1]).cpu().numpy()))
    assert other and other.isdisjoint(grooming)
    bare = Populations(cx)
    bare.touch_types = {}
    with pytest.raises(ValueError):
        build_touch(bare, cx.n, ["x"])


def test_pulses_and_touch_through_model_artifact_session(cx, tmp_path):
    m = build_model(cx, LAYOUT, brain_ms=10, touch_channels=["hit"], include_touch=True,
                    retina_gain=0.0)
    named = m.populations()
    assert "clock" in named and "touch" in named and "giantfibre" not in named
    m.reset()
    m.probe({"name": "clock"})
    m.observe(_frame())
    quiet = int(m.last_probe["spikes"].sum())
    m.observe(_frame(), pulses={"clock": 40.0})      # a pulse lasts one observation (10 ms)
    assert int(m.last_probe["spikes"].sum()) > quiet
    with pytest.raises(ValueError):
        m.observe(_frame(), pulses={"nothing": 1.0})
    assert m.pulse_drive(None) is None and m.pulse_drive({}) is None
    f = m.observe(_frame(), touch={"hit": 1.0})
    assert f[-1] == 1.0
    path = save_model(m, str(tmp_path / "touch"))
    assert validate(path) == [] and "touch: hit" in describe(path)
    loaded = load_model(path)
    assert loaded.touch.channels == ["hit"]
    s = Session(loaded)
    assert s.info()["touch_channels"] == ["hit"]
    frame = base64.b64encode(np.zeros((6, 8, 3), np.uint8).tobytes()).decode()
    r = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6, "touch": [0.5],
                  "pulses": {"clock": 10}})
    assert r["ok"] and abs(r["features"][-1] - 0.5) < 1e-6
    bad = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6,
                    "pulses": {"nowhere": 1}})
    assert not bad["ok"] and "unknown population" in bad["error"]
    grpc = pytest.importorskip("grpc")
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    server = make_server(loaded, "127.0.0.1:0")
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        assert list(stub.Info(pb.Empty()).touch_channels) == ["hit"]
        raw = np.zeros((6, 8, 3), np.uint8).tobytes()
        r = stub.Observe(pb.StepRequest(frame=raw, width=8, height=6, touch=[0.7],
                                        pulses={"clock": 10.0}))
        assert r.ok and abs(r.features[-1] - 0.7) < 1e-6
    finally:
        server.stop(0)


def test_giant_fibre_startle_on_the_cord(synthetic_cx):
    from neurofly_training.envs import make_env
    from neurofly_training.export import body_model_from_env
    env = make_env("forward", brain="synthetic", synthetic_n=1500)
    model = body_model_from_env(env, "walk")
    env.reset()
    obs = np.asarray(env._last_body_obs, np.float32)
    env.close()
    assert "giantfibre" in model.populations() and len(model.populations()["giantfibre"]) == 2
    model.reset()
    model.probe({"name": "giantfibre"})
    model.observe(obs)
    quiet = int(model.last_probe["spikes"].sum())
    # a body observation is 2 ms of brain time, so a startle pulse has to be large
    model.observe(obs, pulses={"giantfibre": 300.0})
    assert int(model.last_probe["spikes"].sum()) > quiet
    s = Session(model)
    r = s.handle({"op": "body_observe", "obs": obs.tolist(), "pulses": {"giantfibre": 300}})
    assert r["ok"] and r["probe"]["spikes"] and sum(r["probe"]["spikes"]) > 0
