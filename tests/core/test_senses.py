"""Reward dopamine, learning in the mushroom body, taste and temperature channels, and
the compass as a readout."""
import base64

import numpy as np
import pytest

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.model import Model, ModelConfig
from neurofly_core.server import Session
from neurofly_training.build import build_gustation, build_model, build_thermo
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations

LAYOUT = ControlLayout(keys=["w"])


@pytest.fixture(scope="module")
def cx():
    return Connectome.toy(1500, seed=0).subset("central")


def _frame():
    return np.full((6, 8, 3), 60, np.uint8)


def test_populations_find_the_new_systems(synthetic_cx, cx):
    for c in (synthetic_cx, cx):
        pops = Populations(c)
        assert len(pops.pam) == 8 and len(pops.ppl1) == 8 and len(pops.kenyon) == 40
        assert len(pops.mbon) == 8 and len(pops.compass) == 8
        assert len(pops.taste_types) >= 3 and len(pops.thermo_types) == 2
        assert len(pops.readout("compass")) == 8 and len(pops.readout("mbon+descending")) > 8
        text = pops.summary()
        assert "PAM 8" in text and "Kenyon" in text and "compass 8" in text


def test_reward_dopamine_drives_pam(cx):
    m = build_model(cx, LAYOUT, brain_ms=5, dopamine_reward=25.0, dopamine_punish=25.0,
                    retina_gain=0.0)
    pops = Populations(cx)
    assert set(m.populations()["reward"]) == set(pops.pam.tolist())
    assert {"kenyon", "mbon", "compass"} <= set(m.populations())
    assert "reward 25 mV" in m.describe()

    def spikes_of(idx, reward):
        m.reset()
        m.probe({"indices": idx.tolist()})
        total = 0
        for _ in range(6):
            m.observe(_frame(), reward=reward)
            total += int(m.last_probe["spikes"].sum())
        return total

    assert spikes_of(pops.pam, +1.0) > 0 and spikes_of(pops.pam, 0.0) == 0
    assert spikes_of(pops.ppl1, -1.0) > 0 and spikes_of(pops.ppl1, +1.0) == 0


def test_mushroom_body_learns_both_ways(cx):
    """With plasticity on the Kenyon cell to MBON synapses, punishment during an odour
    weakens the synapses that odour's Kenyon cells used; reward strengthens them; a
    silent brain changes nothing."""
    pops = Populations(cx)

    def run(reward: float, odour: bool):
        m = build_model(cx, LAYOUT, brain_ms=10, odour_channels=["food"], odour_gain=30.0,
                        retina_gain=0.0, plasticity=True, plasticity_target="mbon",
                        dopamine_punish=20.0, dopamine_reward=20.0)
        e = m.plasticity
        before = m.brain.vals[e.e].clone()
        m.reset()
        for _ in range(20):
            m.observe(_frame(), reward=reward, odours=[1.0] if odour else [0.0])
        delta = (m.brain.vals[e.e].abs() - before.abs()).cpu().numpy()
        return delta, e

    punished, e = run(-1.0, True)
    assert e.n_edges > 0
    pre, post = e.col_e.cpu().numpy(), e.row_e.cpu().numpy()
    assert set(pre) <= set(pops.kenyon.tolist()) and set(post) <= set(pops.mbon.tolist())
    assert punished.sum() < 0 and (punished < 0).any()
    rewarded, _ = run(+1.0, True)
    assert rewarded.sum() > 0
    silent, _ = run(-1.0, False)
    assert abs(silent).sum() < 0.01 * abs(punished).sum()
    with pytest.raises(ValueError):
        Model(build_model(cx, LAYOUT).brain, readout_idx=[0], layout=LAYOUT,
              retina=build_model(cx, LAYOUT).retina, config=ModelConfig(plasticity=True,
                                                                        plasticity_target="mbon"))
    with pytest.raises(ValueError):
        build_model(cx, LAYOUT, plasticity=True, plasticity_target="nowhere")


def test_taste_and_thermo_channels(cx, tmp_path):
    pops = Populations(cx)
    g = build_gustation(pops, cx.n, ["sugar", "bitter"], gain=10.0)
    t = build_thermo(pops, cx.n, ["heat"], gain=10.0)
    assert set(np.flatnonzero(g([1, 0]).cpu().numpy())) == \
        set(pops.taste_types[list(pops.taste_types)[0]].tolist())
    assert set(np.flatnonzero(t([1]).cpu().numpy())) == \
        set(pops.thermo_types[list(pops.thermo_types)[0]].tolist())
    m = build_model(cx, LAYOUT, brain_ms=5, odour_channels=["food"], taste_channels=["sugar",
                    "bitter"], thermo_channels=["heat"], include_odours=True, include_tastes=True,
                    include_thermo=True)
    assert m.n_features == len(m.readout_idx) + 4
    assert {"olfaction", "gustation", "thermo"} <= set(m.populations())
    m.reset()
    f = m.observe(_frame(), odours=[0.5], tastes={"bitter": 1.0}, thermo=[0.25])
    assert np.allclose(f[-4:], [0.5, 0.0, 1.0, 0.25])
    f = m.observe(_frame(), tastes=[0.0, 0.0])                   # others held
    assert np.allclose(f[-4:], [0.5, 0.0, 0.0, 0.25])
    path = save_model(m, str(tmp_path / "senses"))
    assert validate(path) == []
    text = describe(path)
    assert "gustation: sugar, bitter" in text and "thermo: heat" in text
    loaded = load_model(path)
    assert loaded.gustation.channels == ["sugar", "bitter"] and loaded.thermo.channels == ["heat"]
    assert loaded.reward_idx is not None and len(loaded.reward_idx) == len(pops.pam)
    s = Session(loaded)
    info = s.info()
    assert info["taste_channels"] == ["sugar", "bitter"] and info["thermo_channels"] == ["heat"]
    frame = base64.b64encode(np.zeros((6, 8, 3), np.uint8).tobytes()).decode()
    r = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6,
                  "tastes": {"sugar": 0.9}, "thermo": [0.1]})
    assert abs(r["features"][-3] - 0.9) < 1e-6 and abs(r["features"][-1] - 0.1) < 1e-6
    grpc = pytest.importorskip("grpc")
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    server = make_server(loaded, "127.0.0.1:0")
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        i = stub.Info(pb.Empty())
        assert list(i.taste_channels) == ["sugar", "bitter"] and list(i.thermo_channels) == ["heat"]
        raw = np.zeros((6, 8, 3), np.uint8).tobytes()
        r = stub.Observe(pb.StepRequest(frame=raw, width=8, height=6, tastes=[0.2, 0.3],
                                        thermo=[0.4]))
        assert abs(r.features[-2] - 0.3) < 1e-6 and abs(r.features[-1] - 0.4) < 1e-6
    finally:
        server.stop(0)
    with pytest.raises(ValueError):
        build_thermo(Populations(Connectome.synthetic(300, seed=2).subset("vnc")), 300, ["x"])


def test_compass_readout(cx):
    m = build_model(cx, LAYOUT, readout="compass", brain_ms=5)
    assert m.n_features == 8 and set(m.readout_idx) == set(Populations(cx).compass.tolist())
    assert len(m.select({"name": "compass"})) == 8
