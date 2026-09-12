"""Neuron positions, whole-brain activity, and the sinks and protocol ops around them."""
import argparse
import base64
import json
import socket
import threading
import time

import numpy as np
import pytest
import torch

from neurofly_core.activity import (ActivityBroadcaster, ActivityLog, SpikeAccumulator,
                                    parse_address)
from neurofly_core.artifact import load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.experiments import (ActivitySinks, add_experiment_args, apply_experiments,
                                       wants_activity)
from neurofly_core.model import Model
from neurofly_core.server import Session, serve_ws
from neurofly_training.build import build_model
from neurofly_training.data.connectome import Connectome

LAYOUT = ControlLayout(keys=["w"])


@pytest.fixture(scope="module")
def cx():
    return Connectome.toy(1200, seed=0).subset("central")


@pytest.fixture(scope="module")
def model(cx):
    return build_model(cx, LAYOUT, brain_ms=5)


def _frame(v=200):
    return np.full((6, 8, 3), v, np.uint8)


def test_connectome_positions_fill_and_units(synthetic_cx):
    raw, known = synthetic_cx.positions(fill=False)
    assert raw.shape == (synthetic_cx.n, 3) and raw.dtype == np.float32
    sc = synthetic_cx.neurons["superclass"].values
    assert not known[sc == "vnc_sensory"].any() and known[sc == "vnc_motor"].all()
    assert np.isnan(raw[~known]).all()
    filled, known2 = synthetic_cx.positions()
    assert (known2 == known).all() and np.isfinite(filled).all()
    assert np.allclose(filled[known], raw[known])
    # placed sensory neurons land near their side's motor group, not at the origin
    sens = np.flatnonzero((sc == "vnc_sensory") & (synthetic_cx.neurons["rootSide"] == "L"))
    motor = np.flatnonzero((sc == "vnc_motor") & (synthetic_cx.neurons["somaSide"] == "L"))
    assert np.linalg.norm(filled[sens].mean(0) - raw[motor].mean(0)) < 60
    again, _ = synthetic_cx.positions()
    assert np.array_equal(filled, again)                      # deterministic
    sub = synthetic_cx.subset("vnc")
    assert sub.positions()[0].shape == (sub.n, 3)
    bare = Connectome.synthetic(n=300, seed=3)
    bare.neurons = bare.neurons.drop(columns=["soma_x", "soma_y", "soma_z"])
    xyz, k = bare.positions()
    assert np.isnan(xyz).all() and not k.any()


def test_spike_accumulator():
    acc = SpikeAccumulator(5, substeps=True)
    acc.begin()
    acc.add(torch.tensor([True, False, True, False, False]))
    acc.add(torch.tensor([True, False, False, False, False]))
    out = acc.finish()
    assert out["indices"].tolist() == [0, 2] and out["counts"].tolist() == [2, 1]
    assert [s.tolist() for s in out["steps"]] == [[0, 2], [0]]
    acc.begin()
    assert acc.finish()["indices"].size == 0 and acc.finish()["steps"] == []


def test_model_activity_and_map(model):
    assert model.neuron_positions.shape == (model.brain.n, 3)
    assert "positions yes" in model.describe()
    assert model.watch_activity(True, substeps=True) == model.brain.n
    model.reset()
    model.observe(_frame())
    a = model.last_activity
    assert a["indices"].size > 0 and (a["counts"] > 0).all()
    assert len(a["steps"]) == model.substeps + model.warmup_steps
    assert sum(len(s) for s in a["steps"]) == a["counts"].sum()
    m = model.activity_map()
    assert m["n"] == model.brain.n and m["positions"] is model.neuron_positions
    assert len(m["superclass"]) == model.brain.n and "readout" in m["populations"]
    model.reset()
    assert model.last_activity is None
    assert model.watch_activity(False) == 0 and model.activity is None
    model.observe(_frame())
    assert model.last_activity is None


def test_artifact_keeps_positions(model, tmp_path):
    path = save_model(model, str(tmp_path / "art"))
    assert validate(path) == []
    m = json.load(open(tmp_path / "art" / "manifest.json"))
    assert m["neurons"]["positions"]["shape"] == [model.brain.n, 3]
    assert m["neurons"]["positions_unit"] == "micrometre"
    loaded = load_model(path)
    assert np.allclose(loaded.neuron_positions, model.neuron_positions)
    assert (loaded.positions_known == model.positions_known).all()
    # a wrong shape is caught by validate
    bad = json.load(open(tmp_path / "art" / "manifest.json"))
    bad["neurons"]["positions"]["shape"] = [3, model.brain.n]
    json.dump(bad, open(tmp_path / "art" / "manifest.json", "w"))
    assert any("positions" in p for p in validate(path))
    with pytest.raises(ValueError):
        Model(model.brain, readout_idx=model.readout_idx, layout=LAYOUT, retina=model.retina,
              neuron_positions=np.zeros((2, 3)))


def test_session_ops_and_hooks(model):
    calls = []
    s = Session(model, after_step=[lambda: calls.append(1)])
    assert s.info()["has_positions"]
    pos = s.handle({"op": "positions"})
    assert pos["ok"] and len(pos["positions"]) == model.brain.n and pos["unit"] == "micrometre"
    assert len(pos["known"]) == model.brain.n and "readout" in pos["populations"]
    assert s.handle({"op": "activity", "on": True})["n"] == model.brain.n
    req = {"op": "observe", "frame": base64.b64encode(_frame().tobytes()).decode(),
           "width": 8, "height": 6}
    r = s.handle(req)
    assert r["ok"] and r["activity"]["t"] == r["t"] and r["activity"]["indices"]
    assert "steps" not in r["activity"] and calls == [1]
    assert s.handle({"op": "activity", "on": False})["n"] == 0
    assert "activity" not in s.handle(req) and calls == [1, 1]
    s.handle({"op": "nope"})
    assert not s.handle({"op": "step", "frame": "", "width": 1, "height": 1})["ok"]
    assert calls == [1, 1]                                   # hooks only after a good step


def test_graph_op_sends_the_strongest_synapses(model):
    """A viewer drawing the brain as a graph asks for the synapses, capped, strongest first."""
    s = Session(model)
    n = model.brain.n
    full = s.handle({"op": "graph", "limit": 10_000_000})
    assert full["ok"] and full["sampled"] == full["n_edges"] == model.brain.n_edges
    assert len(full["pre"]) == len(full["post"]) == len(full["weight"]) == full["sampled"]
    assert all(0 <= i < n for i in full["pre"]) and all(0 <= i < n for i in full["post"])

    capped = s.handle({"op": "graph", "limit": 25})
    assert capped["sampled"] == 25 and capped["n_edges"] == full["n_edges"]
    # what came back is the heavy end of the distribution, not a uniform sample
    smallest_kept = min(abs(w) for w in capped["weight"])
    assert smallest_kept >= np.median(np.abs(full["weight"]))
    assert s.handle({"op": "graph", "limit": 0})["sampled"] == 0


def test_sinks_log_and_broadcast(model, tmp_path):
    pytest.importorskip("websockets")
    from websockets.sync.client import connect
    p = add_experiment_args(argparse.ArgumentParser())
    args = p.parse_args(["--activity-out", str(tmp_path / "act.json"), "--activity-ws",
                         "127.0.0.1:0"])
    assert wants_activity(args) and not wants_activity(p.parse_args([]))
    done = apply_experiments(model, args)
    assert any("activity" in d for d in done) and model.activity is not None
    sinks = ActivitySinks(model, args, fps=10)
    assert len(sinks.describe()) == 2
    port = sinks.caster.port
    with connect(f"ws://127.0.0.1:{port}") as ws:
        header = json.loads(ws.recv())
        assert header["n"] == model.brain.n and header["fps"] == 10 and header["positions"]
        for _ in range(20):
            if sinks.caster.n_clients:
                break
            time.sleep(0.05)
        model.reset()
        model.observe(_frame())
        sinks.record()
        msg = json.loads(ws.recv())
        assert msg["t"] == 0 and msg["indices"] and len(msg["counts"]) == len(msg["indices"])
        model.observe(_frame())
        sinks.record()
        assert json.loads(ws.recv())["t"] == 1
    out = sinks.close()
    assert out == str(tmp_path / "act.json")
    saved = json.load(open(out))
    assert len(saved["steps"]) == 2 and saved["steps"][1]["t"] == 1
    assert len(saved["positions"]) == model.brain.n and saved["superclass"]
    # a log without a path and a sink set without options are quiet no-ops
    assert ActivityLog(model, None).save() is None
    empty = ActivitySinks(model, p.parse_args([]))
    empty.record()
    assert empty.describe() == [] and empty.close() is None
    model.watch_activity(False)
    assert parse_address(":8767") == ("127.0.0.1", 8767)
    assert parse_address("0.0.0.0:1") == ("0.0.0.0", 1)


def test_ws_subscription_pushes_to_other_clients(model):
    pytest.importorskip("websockets")
    from websockets.sync.client import connect
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    threading.Thread(target=serve_ws, args=(model, "127.0.0.1", port), daemon=True).start()
    for _ in range(50):
        try:
            viewer = connect(f"ws://127.0.0.1:{port}")
            break
        except OSError:
            time.sleep(0.1)
    driver = connect(f"ws://127.0.0.1:{port}")
    with viewer, driver:
        assert json.loads(viewer.recv())["ready"] and json.loads(driver.recv())["ready"]
        viewer.send(json.dumps({"op": "activity", "on": True}))
        assert json.loads(viewer.recv())["n"] == model.brain.n
        driver.send(json.dumps({"op": "observe", "frame": base64.b64encode(_frame().tobytes())
                                .decode(), "width": 8, "height": 6}))
        reply = json.loads(driver.recv())
        pushed = json.loads(viewer.recv())
        assert reply["ok"] and pushed["indices"] == reply["activity"]["indices"]
        viewer.send(json.dumps({"op": "activity", "on": False}))
        assert json.loads(viewer.recv())["n"] == 0
        driver.send(json.dumps({"op": "close"}))
        assert json.loads(driver.recv())["bye"]
    model.watch_activity(False)


def test_broadcaster_alone(model):
    pytest.importorskip("websockets")
    model.watch_activity(True)
    caster = ActivityBroadcaster(model, port=0, fps=5)
    caster.record()                    # nothing observed yet: nothing sent, no error
    assert caster.n_clients == 0
    caster.close()
    model.watch_activity(False)
