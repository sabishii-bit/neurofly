"""The olfaction encoder, odours through the model, artifact and protocols, and the
mushroom-body style learning they make possible."""
import base64
import json

import numpy as np
import pytest

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.encode.olfaction import OlfactionEncoder
from neurofly_core.server import Session
from neurofly_training.build import build_model, build_olfaction
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations

CHANNELS = ["health", "danger"]
LAYOUT = ControlLayout(keys=["w"])


@pytest.fixture(scope="module")
def cx():
    return Connectome.toy(1200, seed=0).subset("central")


@pytest.fixture(scope="module")
def model(cx):
    return build_model(cx, LAYOUT, brain_ms=5, odour_channels=CHANNELS, include_odours=True)


def _frame():
    return np.full((6, 8, 3), 60, np.uint8)


def test_populations_find_glomeruli(synthetic_cx):
    pops = Populations(synthetic_cx)
    assert len(pops.olfactory) > 0 and len(pops.glomeruli) >= 3
    assert all(len(v) > 0 for v in pops.glomeruli.values())
    assert sum(len(v) for v in pops.glomeruli.values()) == len(pops.olfactory)
    assert "glomeruli" in pops.summary()


def test_encoder_vectors_hold_and_adapt(cx):
    pops = Populations(cx)
    enc = build_olfaction(pops, cx.n, CHANNELS, gain=10.0)
    assert enc.n_channels == 2 and enc.targets.size > 0
    names = list(pops.glomeruli)
    # channel 0 drives exactly the first glomerulus, channel 1 the second
    d0 = enc([1.0, 0.0]).cpu().numpy()
    assert set(np.flatnonzero(d0)) == set(pops.glomeruli[names[0]].tolist())
    assert np.allclose(d0[np.flatnonzero(d0)], 10.0)
    d1 = enc({"danger": 0.5}).cpu().numpy()             # dict: health held at 1, danger 0.5
    assert np.allclose(enc.levels, [1.0, 0.5])
    assert set(np.flatnonzero(d1)) == set(pops.glomeruli[names[0]].tolist()
                                          + pops.glomeruli[names[1]].tolist())
    assert enc(None).sum() == d1.sum()                    # None keeps the last odours
    assert np.allclose(enc.vector({"nope": 1.0}), [1.0, 0.5])
    assert np.allclose(enc.vector([2.0]), [1.0, 0.0])     # clipped, missing channels zero
    enc.reset()
    assert enc.levels.sum() == 0 and enc(None).sum() == 0
    back = OlfactionEncoder.from_tables(cx.n, enc.params(), enc.tables())
    assert back.channels == CHANNELS and np.allclose(back([1.0, 0.0]).cpu(), d0)
    adapting = OlfactionEncoder(cx.n, channels=CHANNELS, matrix=enc.matrix, targets=enc.targets,
                                gain=10.0, adapt=1.0)
    first = adapting([1.0, 0.0]).sum().item()
    for _ in range(40):
        last = adapting([1.0, 0.0]).sum().item()
    assert last < first * 0.2                            # the drive fades on a constant odour
    with pytest.raises(ValueError):
        OlfactionEncoder(cx.n, channels=CHANNELS, matrix=enc.matrix[:, :1], targets=enc.targets)
    more = build_olfaction(pops, cx.n, [f"c{i}" for i in range(len(names) + 2)])
    assert more.n_channels == len(names) + 2             # wraps around the glomeruli
    with pytest.raises(ValueError):
        build_olfaction(Populations(Connectome.synthetic(300, seed=2).subset("vnc")), 300, ["x"])


def test_model_smells(model):
    assert model.n_features == len(model.readout_idx) + 2 and "olfaction" in model.populations()
    model.reset()
    f0 = model.observe(_frame())
    assert f0[-2:].sum() == 0
    f1 = model.observe(_frame(), odours={"danger": 1.0})
    assert np.allclose(f1[-2:], [0.0, 1.0])
    f2 = model.observe(_frame())                          # held
    assert np.allclose(f2[-2:], [0.0, 1.0])
    state, info = model.step(_frame(), odours=[0.2, 0.0]) if model.policy else (None, None)
    model.reset()
    assert model.olfaction.levels.sum() == 0
    assert "olfaction" not in build_model(model.brain and Connectome.toy(800).subset("central"),
                                          LAYOUT).populations()


def test_artifact_and_protocols(model, tmp_path):
    path = save_model(model, str(tmp_path / "smell"))
    assert validate(path) == [] and "olfaction: health, danger" in describe(path)
    loaded = load_model(path)
    assert loaded.olfaction.channels == CHANNELS and loaded.include_odours
    s = Session(loaded)
    assert s.info()["odour_channels"] == CHANNELS
    frame = base64.b64encode(np.zeros((6, 8, 3), np.uint8).tobytes()).decode()
    req = {"op": "observe", "frame": frame, "width": 8, "height": 6}
    assert sum(s.handle(req)["features"][-2:]) == 0
    r = s.handle({**req, "odours": {"health": 0.7}})
    assert abs(r["features"][-2] - 0.7) < 1e-6
    r = s.handle({**req, "odours": [0.0, 1.0]})
    assert r["features"][-2:] == [0.0, 1.0]
    assert s.handle(req)["features"][-2:] == [0.0, 1.0]   # held when absent
    m = json.load(open(tmp_path / "smell" / "manifest.json"))
    m["olfaction"]["params"]["channels"] = []
    json.dump(m, open(tmp_path / "smell" / "manifest.json", "w"))
    assert any("channels" in p for p in validate(path))
    grpc = pytest.importorskip("grpc")
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    server = make_server(loaded, "127.0.0.1:0")
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        assert list(stub.Info(pb.Empty()).odour_channels) == CHANNELS
        raw = np.zeros((6, 8, 3), np.uint8).tobytes()
        r = stub.Observe(pb.StepRequest(frame=raw, width=8, height=6, odours=[0.3, 0.9]))
        assert r.ok and abs(r.features[-1] - 0.9) < 1e-6
        r = stub.Observe(pb.StepRequest(frame=raw, width=8, height=6))
        assert abs(r.features[-1] - 0.9) < 1e-6
    finally:
        server.stop(0)


def test_punished_odour_changes_synapses(cx):
    """The point of smelling: with plasticity on, punishment during an odour changes the
    synapses the odour's activity reached, far more than punishment in silence."""
    def run(with_odour: bool) -> float:
        m = build_model(cx, LAYOUT, brain_ms=10, odour_channels=CHANNELS, odour_gain=30.0,
                        retina_gain=0.0, plasticity=True, dopamine_punish=20.0)
        before = m.brain.vals.clone()
        m.reset()
        for _ in range(30):
            m.observe(_frame(), reward=-1.0, odours={"danger": 1.0} if with_odour else None)
        return float((m.brain.vals - before).abs().sum())

    with_odour, silent = run(True), run(False)
    assert with_odour > 0 and with_odour > 1.5 * silent          # about double on the toy brain
