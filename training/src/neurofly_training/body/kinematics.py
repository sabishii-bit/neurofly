"""Move the fly's joints directly, without physics: a clean way to see the limbs.

Two sources of joint trajectories:

* ``gait_qpos``: the open-loop tripod gait as joint angles around the standing pose,
  the body held still. No dynamics, so nothing wobbles or falls; the legs cycle the way
  the pattern says. This is for checking the body model and the viewers.
* ``real_walking_qpos``: a real fly's walking, from flybody's walking imitation dataset
  (motion capture of Drosophila, resampled to the model's joints). Download it once with
  ``neurofly body-replay --download`` (3 GB, resumable); then any trajectory in it plays
  through the model.

``play`` sets the joints frame by frame, records the world pose of every body and,
optionally, renders a video:

    neurofly body-replay --gait --poses assets/gait.json --video videos/gait.mp4
    neurofly body-replay --trajectory data/flybody-data/walking.h5 --index 3 --poses ...
"""
from __future__ import annotations

import numpy as np

from neurofly_training.body.actuators import ACTUATOR_NAMES, MODEL_TO_ACTION
from neurofly_training.body.export import PoseRecorder
from neurofly_training.body.gait import JOINT_ACTUATOR


def actuator_joints(model) -> dict[int, int]:
    """Action index -> qpos address of the joint that actuator drives (joint actuators
    only; the action order differs from the model's, see ``actuators``)."""
    out = {}
    for mi in range(model.nu):
        if int(model.actuator_trntype[mi]) == JOINT_ACTUATOR:
            adr = int(model.jnt_qposadr[int(model.actuator_trnid[mi, 0])])
            out[int(MODEL_TO_ACTION[mi])] = adr
    return out


def _joint_range(model, action_index: int) -> tuple[float, float]:
    mi = int(np.flatnonzero(MODEL_TO_ACTION == action_index)[0])
    lo, hi = model.jnt_range[int(model.actuator_trnid[mi, 0])]
    return float(lo), float(hi)


def gait_qpos(physics, gait, n_steps: int, amplitude: float = 1.0) -> np.ndarray:
    """Joint positions (n_steps, nq) of the gait pattern around the standing pose: each
    joint actuator's target moves its joint by ``amplitude`` radians per unit of action,
    clipped to the joint's range; the root stays where it is."""
    m = physics.model
    base = np.array(physics.data.qpos, dtype=np.float64)
    joints = actuator_joints(m)
    ranges = {a: _joint_range(m, a) for a in joints}
    lo = {a: r[0] for a, r in ranges.items()}
    hi = {a: r[1] for a, r in ranges.items()}
    out = np.repeat(base[None], n_steps, axis=0)
    rest = getattr(gait, "rest", 0.0)
    for k in range(n_steps):
        action = gait(k) - rest                            # the swing, not the offset
        for a, adr in joints.items():
            if ACTUATOR_NAMES[a].startswith(("head", "abdomen")):
                continue
            out[k, adr] = float(np.clip(base[adr] + amplitude * action[a], lo[a], hi[a]))
    return out


def real_walking_qpos(path: str, index: int = 0, start: int = 0,
                      end: int | None = None) -> tuple[np.ndarray, float]:
    """A trajectory of flybody's walking imitation dataset as (qpos (T, nq), fps)."""
    from flybody.tasks.trajectory_loaders import HDF5WalkingTrajectoryLoader
    loader = HDF5WalkingTrajectoryLoader(path)
    traj = loader.get_trajectory(index, start_step=start, end_step=end)
    return np.asarray(traj["qpos"], dtype=np.float64), 1.0 / float(loader.timestep)


def play(env, qpos: np.ndarray, *, fps: float, every: int = 1, video: str | None = None,
         poses: str | None = None) -> dict:
    """Set the joints to every ``every``-th row of ``qpos``, record the poses and,
    with ``video``, render each shown frame. Returns what was written."""
    physics = env.physics
    rec = PoseRecorder(lambda: env.physics, fps=fps / every)
    writer = None
    if video:
        import imageio
        import os
        os.makedirs(os.path.dirname(video) or ".", exist_ok=True)
        writer = imageio.get_writer(video, fps=max(1, int(round(fps / every))), macro_block_size=1)
    shown = 0
    with physics.reset_context():
        pass
    for k in range(0, len(qpos), every):
        physics.data.qpos[:] = qpos[k][: physics.model.nq]
        physics.forward()
        rec.record()
        if writer is not None:
            writer.append_data(env.render())
        shown += 1
    if writer is not None:
        writer.close()
    out = {"frames": shown, "fps": fps / every, "bodies": len(rec.bodies)}
    if poses:
        out["poses"] = rec.save(poses)
    if video:
        out["video"] = video
    return out
