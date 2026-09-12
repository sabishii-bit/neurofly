"""Joint trajectories played through the body without physics."""
import json
import os
import sys

import numpy as np
import pytest

from neurofly_training.body.actuators import leg_actuator_indices
from neurofly_training.body.gait import TripodGait
from neurofly_training.body.kinematics import actuator_joints, gait_qpos, play
from neurofly_training.envs import make_body_env


def test_gait_qpos_moves_leg_joints_within_range():
    env = make_body_env("forward", seed=0)
    env.reset()
    m = env.physics.model
    joints = actuator_joints(m)
    assert len(joints) == 45 and all(a < m.nu for a in joints)         # 45 joint actuators
    assert 0 not in joints and joints[6] == int(m.jnt_qposadr[m.actuator_trnid[0, 0]])
    q = gait_qpos(env.physics, TripodGait(500.0, 2.0), 250, amplitude=1.0)
    assert q.shape == (250, m.nq)
    assert np.allclose(q[:, :7], q[0, :7])                              # the root stays put
    for t, side in (("T1", "L"), ("T2", "R"), ("T3", "L"), ("T3", "R")):
        coxa = joints[leg_actuator_indices(t, side)[2]]
        assert np.ptp(q[:, coxa]) > 0.3
        j = int(np.flatnonzero(m.jnt_qposadr == coxa)[0])
        assert q[:, coxa].min() >= m.jnt_range[j][0] and q[:, coxa].max() <= m.jnt_range[j][1]
    head = joints[6]
    assert np.ptp(q[:, head]) == 0                                       # the head is left alone
    env.close()


def test_play_records_poses_and_video(tmp_path):
    env = make_body_env("forward", seed=0, render_mode="rgb_array")
    env.reset()
    q = gait_qpos(env.physics, TripodGait(500.0, 2.0), 100)
    out = play(env, q, fps=500.0, every=10, video=str(tmp_path / "g.mp4"),
               poses=str(tmp_path / "g.json"))
    env.close()
    assert out["frames"] == 10 and out["fps"] == 50.0 and os.path.exists(out["video"])
    poses = json.load(open(tmp_path / "g.json"))
    frames = np.asarray(poses["frames"])
    assert frames.shape == (10, 7 * len(poses["bodies"]))
    i = poses["bodies"].index("walker/tarsus_T1_left")
    tarsus = frames[:, 7 * i: 7 * i + 3]
    j = poses["bodies"].index("walker/thorax")
    thorax = frames[:, 7 * j: 7 * j + 3]
    assert np.ptp(tarsus, axis=0).max() > 0.02                          # the foot moves (cm)
    assert np.ptp(thorax, axis=0).max() < 1e-6                          # the body does not


def test_body_replay_cli(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import body_replay
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "--gait", "--steps", "60", "--poses",
                                      str(tmp_path / "p.json")])
    body_replay.main()
    assert "tripod gait, 60 steps: 6 frames" in capsys.readouterr().out
    monkeypatch.setattr(body_replay, "find_walking_dataset", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "--index", "0"])
    with pytest.raises(SystemExit):
        body_replay.main()


@pytest.mark.skipif(not os.environ.get("FLYBODY_WALKING_H5"),
                    reason="set FLYBODY_WALKING_H5 to a walking imitation dataset")
def test_real_walking_dataset():
    from neurofly_training.body.kinematics import real_walking_qpos
    q, fps = real_walking_qpos(os.environ["FLYBODY_WALKING_H5"], 0)
    assert q.ndim == 2 and fps > 0
