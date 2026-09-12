import json

import numpy as np

from neurofly_core.artifact import load_model, save_model, validate
from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.decode.linear import ControlDecoder
from neurofly_training.build import build_model
from neurofly_training.data.populations import Populations
from neurofly_training.evaluation import evaluate_artifact, format_report
from neurofly_training.surrogate import train_surrogate


def _bars_recording(path, T=40):
    """A recording directory: video of alternating bars, actions holding a/d with the bar."""
    import imageio.v2 as imageio
    path.mkdir()
    layout = ControlLayout(keys=["a", "d"], mouse=True)
    frames, actions = [], np.zeros((T, layout.n), np.float32)
    with imageio.get_writer(str(path / "video.mp4"), fps=10, macro_block_size=1) as w:
        for t in range(T):
            left = (t // 10) % 2 == 0
            f = np.zeros((60, 80, 3), np.uint8)
            f[:, :40] = 255 if left else 0
            f[:, 40:] = 0 if left else 255
            actions[t] = layout.encode(ControlState(frozenset({"a" if left else "d"}),
                                                    dx=-30.0 if left else 30.0))
            frames.append(f)
            w.append_data(f)
    np.save(path / "actions.npy", actions)
    json.dump({"layout": layout.to_dict(), "fps": 10, "size": [80, 60], "n_frames": T},
              open(path / "meta.json", "w"))
    return layout, frames, actions


def test_evaluate_artifact(synthetic_cx, tmp_path):
    layout, frames, actions = _bars_recording(tmp_path / "rec")
    ret = Populations(synthetic_cx).retina()
    readout = np.concatenate([ret["L"]["idx"], ret["R"]["idx"]])
    model = build_model(synthetic_cx, layout, readout=readout, brain_ms=50)
    # a policy that is right by construction: fit on the recording's own features
    from neurofly_training.pc.imitation import fit_control_decoder
    model.reset()
    X = np.stack([model.observe(f) for f in frames])
    model.policy = fit_control_decoder(X, actions, layout, epochs=200)
    art = save_model(model, str(tmp_path / "art"))
    report = evaluate_artifact(art, [str(tmp_path / "rec")],
                               reward="neurofly_training.pc.task:PatchBrightness")
    assert report["recordings"][0]["frames"] == 40
    assert report["agreement"] > 0.9, report
    assert report["controls"]["key:a"]["f1"] > 0.9 and "reward_mean" in report
    assert 0.0 <= report["reward_mean"] <= 1.0
    text = format_report(report)
    assert "agreement" in text and "key:a" in text
    # without a policy there is nothing to score, and that is said
    model.policy = None
    art2 = save_model(model, str(tmp_path / "art2"))
    r2 = evaluate_artifact(art2, [str(tmp_path / "rec")], max_frames=5)
    assert not r2["has_policy"] and "nothing to score" in format_report(r2)


def test_surrogate_projection_mode_learns(synthetic_cx, tmp_path):
    layout, frames, actions = _bars_recording(tmp_path / "rec", T=20)
    cx = synthetic_cx.subset("central")
    pops = Populations(cx)
    model = build_model(cx, layout, readout=pops.visual_projection, brain_ms=10)
    assert model.retina.mode == "projection"
    hist = train_surrogate(model, frames, None, actions, epochs=4, lr=5e-2, window=4)
    assert len(hist["loss"]) == 4 and np.isfinite(hist["loss"]).all()
    assert hist["loss"][-1] < hist["loss"][0]
    assert isinstance(model.policy, ControlDecoder) and model.retina.mode == "projection"
    # the trained model still runs and exports
    model.reset()
    obs = model.observe(frames[0])
    assert obs.shape == (model.n_features,) and np.isfinite(model.act(obs)).all()
    path = save_model(model, str(tmp_path / "trained"))
    assert validate(path) == [] and load_model(path).policy.kind == "linear"


def test_surrogate_hex_mode_runs(synthetic_cx, tmp_path):
    layout, frames, actions = _bars_recording(tmp_path / "rec", T=12)
    model = build_model(synthetic_cx.subset("brain"), layout, brain_ms=5, include_frame=True,
                        frame_grid=(6, 8))
    assert model.retina.mode == "hex"
    n_driven = model.retina.n_driven
    hist = train_surrogate(model, frames, None, actions, epochs=2, lr=1e-2, window=6)
    assert np.isfinite(hist["loss"]).all()
    assert model.retina.mode == "hex" and 0 < model.retina.n_driven <= n_driven
    model.reset()
    assert model.observe(frames[0]).shape == (model.n_features,)
