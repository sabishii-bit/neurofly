import json
import time

import numpy as np
import pytest

from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.io.controls import Controls
from neurofly_core.io.guard import FocusGuard, Watchdog
from neurofly_training.build import build_model
from neurofly_training.calibrate import format_sweep, sweep
from neurofly_training.data.connectome import Connectome
from neurofly_training.idm import InverseDynamics
from neurofly_training.pc.dagger import human_active
from neurofly_training.pc.imitation import collect_features, evaluate, fit_control_decoder
from neurofly_training.pc.screen import TemplatePresence, match_template
from tests.training.test_tools import _bars_recording


@pytest.fixture(scope="module")
def toy_cx():
    return Connectome.toy(n=1500, seed=0)


def _bars(T=40, layout=None):
    layout = layout or ControlLayout(keys=["a", "d"], mouse=True)
    frames, actions = [], np.zeros((T, layout.n), np.float32)
    for t in range(T):
        left = (t // 10) % 2 == 0
        f = np.zeros((60, 80, 3), np.uint8)
        f[:, :40] = 255 if left else 0
        f[:, 40:] = 0 if left else 255
        actions[t] = layout.encode(ControlState(frozenset({"a" if left else "d"}),
                                                dx=-30.0 if left else 30.0))
        frames.append(f)
    return layout, frames, actions


def test_toy_brain_readout_tells_left_from_right(toy_cx):
    layout, frames, actions = _bars()
    model = build_model(toy_cx.subset("brain"), layout, brain_ms=30)
    assert model.retina.mode == "hex"
    model.reset()
    left = model.observe(frames[0])
    right = model.observe(frames[10])
    right = model.observe(frames[10])
    assert np.abs(left - right).sum() > 0.5, "descending rates should differ between eyes"
    # imitation on the descending readout works on the toy brain (it does not on the random one)
    from neurofly_core.io.video import SyntheticVideo
    from neurofly_training.pc.env import PCEnv
    env = PCEnv(model, video=SyntheticVideo(lambda t: frames[t], n_frames=len(frames)))
    X, Y = collect_features(env, actions)
    dec = fit_control_decoder(X, Y, layout, epochs=200)
    ev = evaluate(dec, X, Y)
    assert ev["key:a"]["f1"] > 0.85 and ev["key:d"]["f1"] > 0.85, ev
    env.close()


def test_idm_learns_and_labels(tmp_path):
    layout, frames, actions = _bars(T=80)
    idm = InverseDynamics(layout, grid=(6, 8), k=1, hidden=32)
    hist = idm.fit(frames, actions, epochs=150, lr=3e-3)
    assert hist[-1] < hist[0]
    pred = idm.predict(frames)
    assert pred.shape == actions.shape
    assert np.mean((pred[:, :2] > 0) == (actions[:, :2] > 0)) > 0.95
    idm.save(str(tmp_path / "idm"))
    again = InverseDynamics.load(str(tmp_path / "idm"))
    assert np.allclose(again.predict(frames), pred, atol=1e-5)
    # label a video and read it back as a recording
    from neurofly_training.cli.label import label_video
    _bars_recording(tmp_path / "rec", T=20)
    n = label_video(again, str(tmp_path / "rec" / "video.mp4"), str(tmp_path / "labelled"))
    assert n == 20 and (tmp_path / "labelled" / "actions.npy").exists()
    meta = json.load(open(tmp_path / "labelled" / "meta.json"))
    assert meta["labelled_by"] == "idm" and meta["layout"]["keys"] == ["a", "d"]


def test_template_matching_and_task():
    frame = np.zeros((60, 80, 3), np.uint8)
    ramp = np.linspace(40, 250, 12)[None, :] * np.linspace(0.5, 1.0, 10)[:, None]
    icon = ramp.astype(np.uint8)
    frame[20:30, 50:62] = icon[..., None]
    template = frame[20:30, 50:62].copy()
    score, where = match_template(frame, template)
    assert score > 0.99 and where == (20, 50)
    task = TemplatePresence(template, end_below=0.5)
    info = {}
    assert task.reward(frame, None, ControlState(), info) > 0.99
    assert not task.done(frame, None, info)
    gone = np.zeros_like(frame)
    info = {}
    task.reward(gone, None, ControlState(), info)
    assert task.done(gone, None, info)


def test_calibrate_sweep(toy_cx):
    layout, frames, _ = _bars(T=6)

    def build(**kw):
        return build_model(toy_cx.subset("central"), layout, brain_ms=5, **kw)

    res = sweep(build, "brain_gain", frames, grid=[0.5, 1.0, 2.0], target=0.3)
    assert [r["brain_gain"] for r in res["rows"]] == [0.5, 1.0, 2.0]
    assert all(0 <= r["readout_active"] <= 1 for r in res["rows"])
    assert "pick: --brain-gain" in format_sweep(res)


def test_human_active_and_watchdog():
    assert not human_active(ControlState())
    assert human_active(ControlState(keys=frozenset({"w"})))
    assert human_active(ControlState(dx=5.0)) and not human_active(ControlState(dx=1.0))
    assert human_active(ControlState(axes=(("lx", 0.5),)))

    released = []

    class Rec(Controls):
        def release_all(self):
            released.append(time.monotonic())

    dog = Watchdog(Rec(), timeout=0.2)
    dog.heartbeat()
    time.sleep(0.6)
    dog.close()
    assert released and dog.fired
    guard = FocusGuard(enabled=False)
    guard.arm()
    assert guard.ok() and "off" in guard.describe()


def test_replay_video(tmp_path):
    from neurofly_training.cli import replay
    _bars_recording(tmp_path / "rec", T=8)
    spikes = np.random.default_rng(0).integers(0, 3, (8, 5))
    np.savez(tmp_path / "probe.npz", t=np.arange(8), spikes=spikes, rates=np.ones((8, 5)),
             indices=np.arange(5), ids=np.arange(5))
    import sys
    argv = sys.argv
    sys.argv = ["replay", "--recording", str(tmp_path / "rec"),
                "--probe", str(tmp_path / "probe.npz"), "--out", str(tmp_path / "replay.mp4")]
    try:
        replay.main()
    finally:
        sys.argv = argv
    from neurofly_core.io.video import VideoFile
    v = VideoFile(str(tmp_path / "replay.mp4"))
    f = v.reset()
    assert f.shape[1] == 80 + 320 and f.shape[0] == 60
    v.close()
