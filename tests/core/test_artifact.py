import base64
import json

import numpy as np

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.decode.mlp import MLPPolicy
from neurofly_core.server import Session
from neurofly_training.build import build_model

LAYOUT = ControlLayout(keys=["w", "a"], buttons=["left"], mouse=True)


def _frames(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8) for _ in range(n)]


def _run(model, frames, audio=None):
    model.reset()
    return np.stack([model.observe(f, audio) for f in frames])


def test_roundtrip_linear_policy(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=5, audio=True,
                        dopamine_punish=5.0, name="unit")
    model.policy = ControlDecoder(model.n_features, LAYOUT, seed=3)
    path = save_model(model, str(tmp_path / "art"), extra={"note": "test"})
    assert validate(path) == []
    assert "unit" in describe(path)
    manifest = json.load(open(tmp_path / "art" / "manifest.json"))
    assert manifest["brain"]["n_neurons"] == model.brain.n
    assert manifest["policy"]["params"]["type"] == "linear"
    again = load_model(path)
    assert again.n_features == model.n_features and again.layout.names == LAYOUT.names
    assert again.brain.n_edges == model.brain.n_edges and again.audition is not None
    frames = _frames(4)
    tone = 0.2 * np.sin(2 * np.pi * 300 * np.arange(1600) / 16000).astype(np.float32)[:, None]
    assert np.allclose(_run(model, frames, tone), _run(again, frames, tone))
    model.reset()
    again.reset()
    s1, i1 = model.step(frames[0], tone)
    s2, i2 = again.step(frames[0], tone)
    assert s1 == s2 and i1["action"] == i2["action"]


def test_roundtrip_mlp_policy(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("central"), LAYOUT, brain_ms=5)
    n = model.n_features
    rng = np.random.default_rng(0)
    layers = [(rng.normal(size=(8, n)), rng.normal(size=8)),
              (rng.normal(size=(LAYOUT.n, 8)), rng.normal(size=LAYOUT.n))]
    model.policy = MLPPolicy(LAYOUT, layers, activation="tanh", obs_mean=np.zeros(n),
                             obs_var=np.ones(n))
    path = save_model(model, str(tmp_path / "mlp"))
    assert validate(path) == []
    again = load_model(path)
    x = rng.normal(size=n)
    assert np.allclose(again.policy(x), model.policy(x), atol=1e-6)
    assert np.all(np.abs(again.policy(x)) <= 1)


def test_validate_reports_corruption(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("central"), LAYOUT)
    path = save_model(model, str(tmp_path / "bad"))
    (tmp_path / "bad" / "brain" / "values.bin").write_bytes(b"\x00" * 8)
    problems = validate(path)
    assert any("brain.values" in p for p in problems)


def test_server_session(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=5, audio=True)
    model.policy = ControlDecoder(model.n_features, LAYOUT, seed=1)
    s = Session(model)
    info = s.handle({"op": "info"})
    assert info["ok"] and info["n_actions"] == LAYOUT.n and info["controls"] == LAYOUT.names
    frame = _frames(1)[0]
    audio = (0.1 * np.sin(np.arange(800))).astype("<f4")
    req = {"op": "step", "frame": base64.b64encode(frame.tobytes()).decode(),
           "width": 64, "height": 48, "audio": base64.b64encode(audio.tobytes()).decode(),
           "channels": 1, "reward": -1.0}
    r = s.handle(req)
    assert r["ok"] and r["t"] == 1 and len(r["action"]) == LAYOUT.n
    assert set(r) >= {"keys", "buttons", "dx", "dy", "scroll", "held", "spikes"}
    r2 = s.handle(dict(req, observe_only=True))
    assert r2["ok"] and len(r2["features"]) == model.n_features
    assert s.handle({"op": "reset"})["ok"] and s.handle({"op": "step", "frame": "AAAA",
                                                           "width": 3, "height": 3})["ok"] is False
    assert s.handle({"op": "nope"})["ok"] is False
    # a trainer in another language: observe, set_policy, save
    model.policy = None
    r3 = s.handle(dict(req, op="observe"))
    assert r3["ok"] and len(r3["features"]) == model.n_features
    assert s.handle(dict(req))["features"]              # step without a policy = observe
    W = np.zeros((LAYOUT.n, model.n_features))
    b = [1.0] + [-1.0] * (LAYOUT.n - 1)          # hold "w", nothing else, whatever the rates
    assert s.handle({"op": "set_policy", "type": "linear", "W": W.tolist(),
                     "b": b})["type"] == "linear"
    assert s.handle({"op": "set_policy", "type": "linear", "W": [[1.0]], "b": [0.0]})["ok"] is False
    r4 = s.handle(dict(req))
    assert "keys" in r4 and r4["keys"] == ["w"]
    layers = [{"W": np.ones((4, model.n_features)).tolist(), "b": [0.0] * 4},
              {"W": np.ones((LAYOUT.n, 4)).tolist(), "b": [0.0] * LAYOUT.n}]
    assert s.handle({"op": "set_policy", "type": "mlp", "layers": layers})["type"] == "mlp"
    out = s.handle({"op": "save", "path": str(tmp_path / "from_client"), "name": "client"})
    assert out["ok"] and validate(out["path"]) == []
    assert load_model(out["path"]).policy.kind == "mlp"
    assert s.handle({"op": "close"})["bye"]
