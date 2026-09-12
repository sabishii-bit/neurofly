import argparse
import json
import sys
import types

import numpy as np
import pytest

from neurofly_core.controls import ControlLayout, ControlState
from neurofly_training import envs
from neurofly_training.data import download as dl
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations
from neurofly_training.pc import screen
from neurofly_training.pc.dagger import CorrectionRecorder
from tests.fakes import FakeInputRecorder


def test_envs_helpers_and_errors(tmp_path):
    p = envs.add_env_args(argparse.ArgumentParser())
    args = p.parse_args(["--task", "pc", "--brain", "toy", "--keys", "w"])
    cfg = envs.resolve_env_args(args)
    assert cfg["subset"] == "central" and cfg["readout"] == "descending"
    args = p.parse_args(["--task", "forward"])
    assert envs.resolve_env_args(args)["subset"] == "vnc"
    with pytest.raises(SystemExit):
        envs.resolve_env_args(p.parse_args(["--task", "pc", "--brain", "none"]))
    with pytest.raises(SystemExit):
        envs.resolve_env_args(p.parse_args(["--task", "nosuchtask"]))
    with pytest.raises(ValueError):
        envs.make_env("nosuchtask")
    with pytest.raises(ValueError):
        envs.load_connectome("nope")
    assert envs.parse_region("1,2,3,4") == (1, 2, 3, 4) and envs.parse_region("") is None
    assert envs.wants_audio("loopback") and not envs.wants_audio("none")
    assert "pc" in envs.list_tasks()
    # a pc env on a fake screen, dry run, with a live "none" audio spec
    from tests.fakes import FakeScreen
    env = envs.make_pc_env("pc", brain="toy", synthetic_n=1000, keys="w", dry_run=True,
                          video_source=FakeScreen(), audio="none", max_steps=2)
    assert env.realtime and env.controls.__class__.__name__ == "LoggingControls"
    assert env.render() is None
    env.reset()
    _, _, _, trunc, _ = env.step(env.action_space.sample())
    _, _, _, trunc, _ = env.step(env.action_space.sample())
    assert trunc
    env.close()
    with pytest.raises(ValueError):
        envs.make_pc_env("pc", brain="none", keys="w", video_source=FakeScreen())


def test_connectome_and_populations_extras(synthetic_cx):
    cx = synthetic_cx
    assert "neurons" in cx.summary() and cx.n_synapses > 0
    motor = cx.select(superclass=["vnc_motor", "descending_neuron"])
    assert len(motor) > len(cx.select(superclass="vnc_motor"))
    up = cx.neighborhood(motor[:3], hops=1, direction="up")
    down = cx.neighborhood(motor[:3], hops=1, direction="down")
    assert len(up) >= 3 and len(down) >= 3
    with pytest.raises(ValueError):
        cx.subset("nope")
    pops = Populations(cx)
    with pytest.raises(ValueError):
        pops.readout("motor+nope")
    assert len(pops.readout("cbmotor+visual")) >= len(pops.visual_projection)
    assert "retina columns" in pops.summary()
    # no hex columns at all -> empty retina, projection fallback in build
    bare = Connectome.synthetic(n=600, seed=1)
    bare.neurons = bare.neurons.drop(columns=["assignedOlHex1", "assignedOlHex2"])
    assert Populations(bare).n_retina == 0
    assert cx.subset("full") is cx


def test_download_with_fake_requests(tmp_path, monkeypatch):
    class Resp:
        headers = {"content-length": "6"}

        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield b"abc"
            yield b"def"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    calls = []
    monkeypatch.setattr(dl.requests, "get",
                        lambda url, stream=True, timeout=0: calls.append(url) or Resp())
    dl.download(str(tmp_path), overwrite=False)
    assert len(calls) == len(dl.FILES)
    for name in dl.FILES:
        assert (tmp_path / name).read_bytes() == b"abcdef"
    dl.download(str(tmp_path), overwrite=False)      # skips existing files
    assert len(calls) == len(dl.FILES)
    dl.download(str(tmp_path), overwrite=True)
    assert len(calls) == 2 * len(dl.FILES)


def test_correction_recorder(tmp_path, monkeypatch):
    from neurofly_training.pc import dagger
    monkeypatch.setattr(dagger, "InputRecorder", FakeInputRecorder)
    layout = ControlLayout(keys=["w"], mouse=True)
    rec = CorrectionRecorder(layout, str(tmp_path / "corr"), fps=10)
    frame = np.zeros((6, 8, 3), np.uint8)
    states = [rec.step(frame) for _ in range(4)]
    assert states[0] is None and states[1] is not None and rec.taken_over == 2
    out = rec.close()
    actions = np.load(tmp_path / "corr" / "actions.npy")
    assert actions.shape == (2, layout.n)
    assert json.load(open(tmp_path / "corr" / "meta.json"))["corrections"]
    assert out.endswith("corr")


def test_screen_ocr_with_fake_tesseract(monkeypatch, tmp_path):
    frame = np.zeros((10, 20, 3), np.uint8)
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    assert screen.read_number(frame) is None
    fake = types.ModuleType("pytesseract")
    fake.image_to_string = lambda img, config="": " 42 \n"
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert screen.read_number(frame, region=(0, 0, 0.5, 1.0)) == 42
    task = screen.NumberOnScreen(region=(0, 0, 1, 1), scale=2.0, end_below=10)
    info = {}
    assert task.reward(frame, None, ControlState(), info) == 0.0        # first reading
    fake.image_to_string = lambda img, config="": "45"
    assert task.reward(frame, None, ControlState(), info) == 6.0        # (45 - 42) * 2
    assert not task.done(frame, None, info)
    fake.image_to_string = lambda img, config="": "3"
    task.reward(frame, None, ControlState(), info)
    assert task.done(frame, None, info)
    task.reset(None, None, None)
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), 100, np.uint8)).save(tmp_path / "t.png")
    t = screen.TemplatePresence(str(tmp_path / "t.png"), region=(0, 0, 1, 1))
    big = np.zeros((2, 2, 3), np.uint8)
    assert screen.match_template(big, t.template)[0] == -1.0            # template larger than frame


def test_body_experiments_helper(synthetic_cx):
    from neurofly_training.experiments import apply_to_body_env, select_cx
    env = envs.make_env("forward", brain="synthetic", synthetic_n=1000)
    ns = argparse.Namespace(stimulate=["superclass=vnc_motor:5", "indices=1,2:3"],
                            silence=["type_re=synthetic"], probe="name=readout")
    done = apply_to_body_env(env, ns)
    assert len(done) == 4 and "PC tasks only" in done[-1]
    with pytest.raises(SystemExit):
        apply_to_body_env(env, argparse.Namespace(stimulate=["superclass=vnc_motor"],
                                                  silence=[], probe=None))
    ids = env.cx.neurons["bodyId"].values[:2].tolist()
    assert len(select_cx(env.cx, {"ids": ids})) == 2
    with pytest.raises(ValueError):
        select_cx(env.cx, {"name": "readout"})
    env.close()
    body = envs.make_env("forward", brain="none")
    assert "ignored" in apply_to_body_env(body, ns)[0]
    body.close()
