"""Odours from a Task or from detections, through the env, eval and surrogate."""
import json
import sys

import numpy as np
import pytest

from neurofly_training import envs
from neurofly_training.pc import detect as D
from neurofly_training.pc.task import Task, load_task
from tests.training.test_detectors import PerfectDetector
from tests.training.test_tools import _bars_recording


class Smelly(Task):
    """health rises with the frame index; danger when the bright bar is on the left."""

    def odours(self, frame, audio, info):
        left = frame[:, :40].mean() > frame[:, 40:].mean()
        return {"danger": 1.0 if left else 0.0, "health": min(1.0, info.get("t", 0) / 10)}


def test_odour_channels_and_sources():
    assert envs.odour_channels(None) == [] and envs.odour_channels("none") == []
    assert envs.odour_channels("health, danger") == ["health", "danger"]
    assert envs.odour_channels(["a"]) == ["a"]
    assert envs.odour_channels("detections", ["bar"]) == ["bar"]
    with pytest.raises(ValueError):
        envs.odour_channels("detections")
    assert load_task(None).odours(None, None, {}) is None


def test_env_smells_from_task_and_from_detections(tmp_path):
    _bars_recording(tmp_path / "rec", T=12)
    video = str(tmp_path / "rec" / "video.mp4")
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a,d",
                           odours="health,danger", include_odours=True, task_obj=Smelly())
    obs, _ = env.reset()
    assert env.model.olfaction.channels == ["health", "danger"]
    assert obs[-1] == 1.0 and obs[-2] == 0.0                     # left bar at t=0, no health yet
    for _ in range(11):
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert obs[-2] == 1.0 and info["odours"]["health"] == 1.0 and obs[-1] == 0.0   # right bar
    env.close()
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a,d", detect=["bar"],
                           detector=PerfectDetector(), odours="detections", include_odours=True)
    obs, _ = env.reset()
    assert env.model.olfaction.channels == ["bar"] and env.model.config.meta["odours"] == \
        "detections"
    assert abs(obs[-1] - 0.9) < 1e-6                             # the detector's score
    env.close()
    plain = envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a")
    assert plain.model.olfaction is None and plain.odours is None
    plain.close()


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["neurofly-test"] + [str(a) for a in argv])
    module.main()


def test_build_eval_surrogate_with_odours(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import build, eval as eval_cli, surrogate
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=16)
    task = "tests.training.test_odours:Smelly"
    _run(build, ["art", "--brain", "toy", "--synthetic-n", "1000", "--keys", "a,d",
                 "--odours", "health,danger", "--include-odours"], monkeypatch)
    out = capsys.readouterr().out
    assert "olfaction: health, danger" in out
    _run(eval_cli, ["art", "rec", "--max-frames", "6", "--reward", task], monkeypatch)
    assert (tmp_path / "art" / "eval.json").exists()
    _run(surrogate, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--epochs", "1",
                     "--bptt-window", "4", "--max-frames", "8", "--run-name", "sur",
                     "--odours", "health,danger", "--reward", task], monkeypatch)
    out = capsys.readouterr().out
    assert "olfaction: health, danger" in out and "ok ->" in out
    monkeypatch.setattr(D, "make_detector", lambda spec, device="cpu", threshold=None:
                        PerfectDetector())
    _run(surrogate, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--epochs", "1",
                     "--bptt-window", "4", "--max-frames", "8", "--run-name", "sur2",
                     "--detect", "owl:bar", "--odours", "detections"], monkeypatch)
    assert "olfaction: bar" in capsys.readouterr().out
    cfg = json.load(open(tmp_path / "runs" / "sur2" / "config.json"))
    assert cfg["odours"] == "detections"
