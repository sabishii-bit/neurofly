import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from neurofly_training.body.actuators import (ACTION_TO_MODEL, ACTUATOR_NAMES,
                                              MODEL_ACTUATOR_NAMES, action_order)
from neurofly_training.body.tasks import TASKS
from neurofly_training.envs import make_body_env


@pytest.mark.parametrize("task", sorted(TASKS))
def test_body_env_roundtrip(task):
    env = make_body_env(task, seed=0, render_mode="rgb_array")
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    for _ in range(20):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(r)
    assert env.render().shape == (480, 640, 3)
    env.close()


def test_layout_matches_flybody():
    env = make_body_env("forward", seed=0)
    walker = env.dm_env.task._walker
    assert [a.name for a in walker.actuators] == MODEL_ACTUATOR_NAMES
    assert action_order(walker) == ACTUATOR_NAMES            # the layout the action vector uses
    assert ACTUATOR_NAMES[:6] == [f"adhere_claw_{t}_{s}" for t in ("T1", "T2", "T3")
                                  for s in ("left", "right")]
    assert [MODEL_ACTUATOR_NAMES[i] for i in ACTION_TO_MODEL] == ACTUATOR_NAMES
    # the env's action bounds are the model's control ranges, in action order
    rng = np.asarray(env.physics.model.actuator_ctrlrange)[ACTION_TO_MODEL]
    assert np.allclose(env.unscale_action(-np.ones(59, np.float32)), rng[:, 0])
    assert np.allclose(env.unscale_action(np.ones(59, np.float32)), rng[:, 1])
    assert env.joint_names is not None and len(env.joint_names) == 85
    jp = env.obs_slices["walker/joints_pos"]
    assert jp.stop - jp.start == 85
    assert env.obs_slices["walker/touch"].stop - env.obs_slices["walker/touch"].start == 6
    lo = env.unscale_action(-np.ones(59, np.float32))
    hi = env.unscale_action(np.ones(59, np.float32))
    spec = env.dm_env.action_spec()
    assert np.allclose(lo, spec.minimum) and np.allclose(hi, spec.maximum)
    env.close()


def test_forward_task_reward_and_termination():
    env = make_body_env("forward", seed=0)
    env.reset()
    task = env.dm_env.task
    factors = task.get_reward_factors(env.physics)
    assert len(factors) == 3 and all(0 <= f <= 1 for f in factors)
    # standing still: upright and at height, speed factor at its floor
    assert factors[1] > 0.9 and factors[2] > 0.9
    assert not task.check_termination(env.physics)
    env.close()


def test_sb3_check_env():
    check_env(make_body_env("forward", seed=0), warn=False)


def test_tripod_gait_moves_every_leg():
    """The open-loop gait swings all six legs in two alternating tripods, and the fly
    neither falls nor stays frozen."""
    from neurofly_training.body.gait import TRIPOD_A, TRIPOD_B, TripodGait
    from neurofly_training.body.actuators import leg_actuator_indices
    gait = TripodGait(control_hz=500, stride_hz=2.0)
    acts = gait.actions(250)                                  # one full cycle
    assert acts.shape == (250, 59) and np.abs(acts).max() <= 1.0
    for t, side in TRIPOD_A + TRIPOD_B:
        coxa = leg_actuator_indices(t, side)[2]
        assert acts[:, coxa].std() > 0.2                      # every leg strides
    a_coxa = leg_actuator_indices(*TRIPOD_A[0])[2]
    b_coxa = leg_actuator_indices(*TRIPOD_B[0])[2]
    assert np.corrcoef(acts[:, a_coxa], acts[:, b_coxa])[0, 1] > 0.9   # the tripods alternate
    claw = leg_actuator_indices("T1", "L")[-1]
    assert (acts[:, claw] == -1).all()                                 # no grip by default
    gripping = TripodGait(control_hz=500, adhesion=0.5).actions(250)[:, claw]
    assert gripping.min() == pytest.approx(-1.0, abs=1e-3)
    assert gripping.max() == pytest.approx(0.0, abs=1e-3)
    env = make_body_env("forward", seed=0)
    obs, _ = env.reset()
    # through the physics the sinusoids sit on the standing pose, not mid-range
    from neurofly_training.body.gait import rest_action
    rest = rest_action(env.physics.model)
    ctrl = env.unscale_action(rest)
    pos = (env.physics.model.actuator_trntype == 0)[ACTION_TO_MODEL]
    assert np.abs(ctrl[pos]).max() < 1e-5 and (ctrl[:6] == 0).all()       # no grip
    gait = TripodGait(control_hz=500, stride_hz=2.0, rest=rest)
    assert np.allclose(gait(0)[~pos][gait(0)[~pos] != -1], rest[~pos][gait(0)[~pos] != -1])
    jp = env.obs_slices["walker/joints_pos"]
    joints = []
    for k in range(200):
        obs, r, term, trunc, _ = env.step(gait(k))
        joints.append(obs[jp.start:jp.stop])
        assert not term
    joints = np.stack(joints)
    assert (joints.std(axis=0) > 0.05).sum() >= 12                    # many joints moved
    env.close()
