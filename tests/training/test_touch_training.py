"""Touch and pulses from a Task through the env and the CLI."""
import sys

import numpy as np

from neurofly_training import envs
from neurofly_training.pc.task import Task
from tests.training.test_tools import _bars_recording


class Hitting(Task):
    def touch(self, frame, audio, info):
        return {"hit": 1.0 if info.get("t", 0) == 2 else 0.0}

    def pulses(self, frame, audio, info):
        return {"clock": 20.0} if info.get("t", 0) == 3 else None


def test_env_touch_and_pulses(tmp_path):
    _bars_recording(tmp_path / "rec", T=8)
    video = str(tmp_path / "rec" / "video.mp4")
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1500, keys="a", touch="hit",
                           include_touch=True, task_obj=Hitting())
    assert set(env.senses) == {"touch", "pulses"}
    obs, _ = env.reset()
    assert obs[-1] == 0.0
    seen = []
    for _ in range(4):
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        seen.append((info["t"], float(obs[-1]), info.get("pulses")))
    by_t = {t: (touch, pulses) for t, touch, pulses in seen}
    assert by_t[2][0] == 1.0 and by_t[3][1] == {"clock": 20.0} and by_t[1][1] is None
    env.close()


def test_build_with_touch(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import build
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "art", "--brain", "toy", "--synthetic-n",
                                      "1500", "--keys", "a", "--touch", "head:grooming,hit",
                                      "--include-touch"])
    build.main()
    assert "touch: head:grooming, hit" in capsys.readouterr().out
