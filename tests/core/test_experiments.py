import numpy as np
import pytest

from neurofly_core.artifact import load_model, save_model
from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.io.controls import Controls
from neurofly_core.selection import parse_spec
from neurofly_core.server import Session
from neurofly_training.build import build_model

LAYOUT = ControlLayout(keys=["w"], mouse=True)


def _frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_parse_spec():
    assert parse_spec("type_re=^PPL1:20") == ({"type_re": "^PPL1"}, 20.0)
    assert parse_spec("ids=1,2, 3") == ({"ids": [1, 2, 3]}, None)
    assert parse_spec("name=readout") == ({"name": "readout"}, None)
    with pytest.raises(ValueError):
        parse_spec("nonsense")


def test_select_silence_stimulate_probe(synthetic_cx):
    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=10)
    assert model.neuron_types is not None
    l1 = model.select({"type_re": "^L1$"})
    assert len(l1) > 0 and all(model.neuron_types[i] == "L1" for i in l1)
    desc = model.select({"superclass": "descending_neuron"})
    assert np.array_equal(np.sort(desc), np.sort(model.readout_idx))
    assert np.array_equal(model.select({"name": "readout"}), model.readout_idx)
    ids = model.neuron_ids[desc[:3]]
    assert np.array_equal(model.select({"ids": ids.tolist()}), desc[:3])
    with pytest.raises(ValueError):
        model.select({"name": "nope"})

    # stimulated readout neurons fire; silenced ones never do
    model.reset()
    model.observe(_frame())
    base = model.brain.rates(model.readout_idx).mean()
    assert model.stimulate({"name": "readout"}, 30.0) == len(desc)
    model.reset()
    model.observe(_frame())
    assert model.brain.rates(model.readout_idx).mean() > base + 10
    model.clear()
    assert model.manipulations == []
    assert model.silence({"name": "readout"}) == len(desc)
    model.stimulate({"name": "readout"}, 30.0)
    model.reset()
    model.observe(_frame())
    assert model.brain.rates(model.readout_idx).sum() == 0
    model.clear()

    # a probe reports spikes and rates of its neurons after every observe
    assert model.probe({"name": "retina"}) == model.retina.n_driven
    model.reset()
    model.observe(_frame())
    p = model.last_probe
    assert p["spikes"].shape == (model.retina.n_driven,) and p["rates"].shape == p["spikes"].shape
    assert p["spikes"].sum() > 0
    model.probe(None)
    model.observe(_frame())
    assert model.last_probe is None


def test_annotations_survive_the_artifact(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("central"), LAYOUT)
    path = save_model(model, str(tmp_path / "ann"))
    again = load_model(path)
    assert again.neuron_types is not None
    assert len(again.select({"superclass": "descending_neuron"})) == len(model.readout_idx)
    assert (tmp_path / "ann" / "neurons" / "annotations.json").exists()


def test_server_experiment_ops(synthetic_cx):
    import base64
    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=5)
    model.policy = ControlDecoder(model.n_features, LAYOUT)
    s = Session(model)
    assert s.handle({"op": "select", "type_re": "^L1$"})["n"] > 0
    assert s.handle({"op": "silence", "name": "readout"})["n"] == len(model.readout_idx)
    assert s.handle({"op": "stimulate", "superclass": "descending_neuron", "mv": 20})["ok"]
    assert s.handle({"op": "stimulate", "superclass": "x", "type_re": "y", "mv": 1})["ok"] is False
    assert s.handle({"op": "probe", "name": "retina"})["n"] == model.retina.n_driven
    f = _frame()
    r = s.handle({"op": "step", "frame": base64.b64encode(f.tobytes()).decode(),
                  "width": 64, "height": 48})
    assert r["ok"] and "probe" in r and len(r["probe"]["spikes"]) == model.retina.n_driven
    assert s.handle({"op": "probe", "off": True})["ok"]
    assert s.handle({"op": "clear"})["ok"] and model.manipulations == []
    assert s.handle({"op": "info"})["populations"]["readout"] == len(model.readout_idx)


def test_gamepad_layout_and_controls():
    lay = ControlLayout(keys=["w"], pad_buttons=["a", "rb"], axes=["lx", "rt"])
    assert lay.names == ["key:w", "pad:a", "pad:rb", "axis:lx", "axis:rt"]
    assert lay.n == 5 and lay.n_binary == 3 and lay.has_pad
    st = ControlState(frozenset({"w"}), pad_buttons=frozenset({"rb"}),
                      axes=(("lx", -0.5), ("rt", 0.75)))
    v = lay.encode(st)
    assert v.tolist() == [1.0, -1.0, 1.0, -0.5, 0.5]        # trigger 0.75 -> 0.5 in [-1, 1]
    back = lay.decode(v)
    assert back.pad_buttons == {"rb"} and back.axis == {"lx": -0.5, "rt": 0.75}
    assert back.held == ["w", "pad:rb"]
    assert ControlLayout.from_dict(lay.to_dict()).names == lay.names
    with pytest.raises(ValueError):
        ControlLayout(pad_buttons=["home"])

    log = []

    class Rec(Controls):
        def _press_pad(self, n):
            log.append(("+p", n))

        def _release_pad(self, n):
            log.append(("-p", n))

        def _axes(self, values):
            log.append(("ax", tuple(sorted(values.items()))))

    c = Rec()
    c.apply(st)
    c.apply(ControlState(axes=(("lx", -0.5), ("rt", 0.75))))   # same axes: no axes call
    c.close()
    assert log == [("+p", "rb"), ("ax", (("lx", -0.5), ("rt", 0.75))), ("-p", "rb"),
                   ("ax", (("lx", 0.0), ("rt", 0.0)))]
