"""Body tasks. Each is a dm_control ``composer.Task``; ``make_env`` builds one.

Add a task by subclassing ``flybody.tasks.base.Walking`` and registering the
factory in ``TASKS``.
"""
from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.utils import rewards

from flybody.fly_envs import walk_on_ball
from flybody.fruitfly import fruitfly
from flybody.tasks.base import Walking
from flybody.tasks.constants import _TERMINAL_ANGVEL, _TERMINAL_LINVEL

_REST_HEIGHT = 0.128  # thorax height (cm) of the fly standing in its default pose


class WalkForward(Walking):
    """Free fly on a floor, rewarded for walking in +x while staying upright.

    Per-step reward = speed * upright * height, each in [0, 1]. The episode
    ends early if the fly falls over. Zero action holds the default standing
    pose, so learning starts from a stable posture.
    """

    def __init__(self, target_speed: float = 1.5, claw_friction: float | None = 1.0, **kwargs):
        self._target_speed = target_speed
        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)
        if claw_friction is not None:
            self._walker.mjcf_model.find(
                "default", "adhesion-collision").geom.friction = (claw_friction,)

    def world_velocity(self, physics) -> np.ndarray:
        xmat = np.reshape(physics.bind(self._walker.root_body).xmat, (3, 3))
        return xmat @ self._walker.observables.velocimeter(physics)

    def thorax_up(self, physics) -> float:
        return float(self._walker.observables.world_zaxis(physics)[2])

    def thorax_height(self, physics) -> float:
        return float(self._walker.observables.thorax_height(physics))

    def get_reward_factors(self, physics):
        vx = self.world_velocity(physics)[0]
        speed = rewards.tolerance(vx, bounds=(self._target_speed, float("inf")),
                                  margin=self._target_speed, sigmoid="linear",
                                  value_at_margin=0.1)
        upright = rewards.tolerance(self.thorax_up(physics), bounds=(0.9, 1.0),
                                    margin=0.6, sigmoid="linear", value_at_margin=0.0)
        height = rewards.tolerance(self.thorax_height(physics),
                                   bounds=(0.8 * _REST_HEIGHT, 1.3 * _REST_HEIGHT),
                                   margin=0.5 * _REST_HEIGHT, sigmoid="linear",
                                   value_at_margin=0.0)
        return (float(speed), float(upright), float(height))

    def check_termination(self, physics) -> bool:
        fallen = (self.thorax_up(physics) < 0.5
                  or self.thorax_height(physics) < 0.45 * _REST_HEIGHT)
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))
        exploded = linvel > _TERMINAL_LINVEL or angvel > _TERMINAL_ANGVEL
        return bool(fallen or exploded or super().check_termination(physics))


def make_walk_forward(random_state=None, time_limit: float = 2.0, **task_kwargs):
    task = WalkForward(walker=fruitfly.FruitFly, arena=floors.Floor(),
                       force_actuators=False, disable_wings=True,
                       joint_filter=0.01, adhesion_filter=0.007,
                       time_limit=time_limit, **task_kwargs)
    return composer.Environment(time_limit=time_limit, task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def make_walk_on_ball(random_state=None, **task_kwargs):
    return walk_on_ball(random_state=random_state, **task_kwargs)


TASKS = {
    "forward": make_walk_forward,   # free walking, must balance
    "ball": make_walk_on_ball,      # tethered on a ball, cannot fall
}


def make_env(task: str = "forward", seed: int | None = None, **task_kwargs):
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    return TASKS[task](random_state=np.random.RandomState(seed), **task_kwargs)
