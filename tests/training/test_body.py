import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from neurofly_training.body.actuators import ACTUATOR_NAMES
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
    assert [a.name for a in walker.actuators] == ACTUATOR_NAMES
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
