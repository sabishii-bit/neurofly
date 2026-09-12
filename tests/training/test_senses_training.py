"""Taste, temperature, reward dopamine and the mushroom-body plasticity target through
the env and the CLI."""
import json
import sys

import numpy as np
import pytest

from neurofly_training import envs
from neurofly_training.pc.task import Task
from tests.training.test_tools import _bars_recording


class Feeling(Task):
    def odours(self, frame, audio, info):
        return {"food": 1.0}

    def tastes(self, frame, audio, info):
        return {"sugar": 0.5 if info.get("t", 0) % 2 else 0.0}

    def thermo(self, frame, audio, info):
        return [min(1.0, info.get("t", 0) / 4)]

    def reward(self, frame, audio, state, info):
        return 1.0


def test_env_feeds_every_sense(tmp_path):
    _bars_recording(tmp_path / "rec", T=8)
    video = str(tmp_path / "rec" / "video.mp4")
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1500, keys="a", odours="food",
                           tastes="sugar,bitter", thermo="heat", include_odours=True,
                           include_tastes=True, include_thermo=True, task_obj=Feeling(),
                           dopamine_reward=20.0, plasticity=True, plasticity_target="mbon")
    assert set(env.senses) == {"odours", "tastes", "thermo", "pulses"}
    obs, _ = env.reset()
    assert np.allclose(obs[-4:], [1.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        obs, r, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert r == 1.0 and abs(obs[-1] - 1.0) < 1e-6 and obs[-3] == 0.0   # t=4: sugar off, heat 1
    assert info["tastes"] == {"sugar": 0.0} and info["thermo"] == [1.0]
    assert env.model.plasticity is not None and env.model._reward is not None
    env.close()
    with pytest.raises(ValueError):
        envs.make_pc_model("pc", brain="toy", synthetic_n=1500, keys="a", plasticity=True,
                           plasticity_target="nowhere")


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["neurofly-test"] + [str(a) for a in argv])
    module.main()


def test_build_and_surrogate_with_all_senses(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import build, surrogate
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=12)
    _run(build, ["art", "--brain", "toy", "--synthetic-n", "1500", "--keys", "a,d", "--odours",
                 "food", "--tastes", "sugar", "--thermo", "heat,wet", "--dopamine-reward", "10",
                 "--dopamine-punish", "10", "--plasticity", "--plasticity-target", "mbon"],
         monkeypatch)
    out = capsys.readouterr().out
    assert "gustation: sugar" in out and "thermo: heat, wet" in out
    m = json.load(open(tmp_path / "art" / "manifest.json"))
    assert m["config"]["plasticity_target"] == "mbon" and m["config"]["dopamine_reward"] == 10
    assert m["reward"]["indices"]["shape"] == [8]
    _run(surrogate, ["rec", "--brain", "toy", "--synthetic-n", "1500", "--epochs", "1",
                     "--bptt-window", "4", "--max-frames", "8", "--run-name", "sur",
                     "--tastes", "sugar", "--thermo", "heat", "--reward",
                     "tests.training.test_senses_training:Feeling"], monkeypatch)
    out = capsys.readouterr().out
    assert "gustation: sugar" in out and "ok ->" in out
