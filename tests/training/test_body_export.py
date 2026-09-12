import json

import pytest

import numpy as np

from neurofly_training.body.export import PoseRecorder, _srgb_to_linear, export_body
from neurofly_training.envs import make_body_env

trimesh = pytest.importorskip("trimesh")


def test_export_body_and_poses(tmp_path):
    env = make_body_env("forward", seed=0)
    env.reset()
    info = export_body(env.physics, str(tmp_path / "fly.glb"))
    assert 50 < info["geoms"] < 100 and "walker/thorax" in info["bodies"]   # no collision shapes
    scene = trimesh.load(str(tmp_path / "fly.glb"))
    # the model's material colours, not one grey: a red (eyes), a translucent (wings), a brown
    colours = {tuple(int(v) for v in g.visual.material.baseColorFactor)
               for g in scene.geometry.values()}
    assert len(colours) >= 3
    assert any(r > 150 and g < 40 and b < 40 for r, g, b, _ in colours)
    assert any(a < 255 for *_, a in colours)
    assert not any(r == g == b == 127 for r, g, b, _ in colours)
    names = set(scene.graph.nodes)
    assert "walker/thorax" in names and "walker/head" in names
    rec = PoseRecorder(env.physics, fps=50)
    for _ in range(3):
        env.step(env.action_space.sample())
        rec.record()
    path = rec.save(str(tmp_path / "poses.json"))
    poses = json.load(open(path))
    assert poses["bodies"] == rec.bodies and len(poses["frames"]) == 3
    assert len(poses["frames"][0]) == 7 * len(rec.bodies) and poses["up"] == "z"
    # Three.js's GLTFLoader strips "/" and other punctuation from node names; every pose
    # body must still map to exactly one node afterwards, or the viewer animates nothing
    import re
    sanitise = lambda n: re.sub(r"[^\w-]", "", re.sub(r"\s", "_", n))  # noqa: E731
    node_keys = [sanitise(n) for n in scene.graph.nodes]
    assert len(set(node_keys)) == len(node_keys)
    assert all(sanitise(b) in set(node_keys) for b in poses["bodies"])
    env.close()


def test_srgb_to_linear_is_darker_and_bounded():
    c = np.array([0.0, 0.04, 0.5, 1.0])
    lin = _srgb_to_linear(c)
    assert lin[0] == 0.0 and lin[3] == pytest.approx(1.0)
    assert 0 < lin[2] < 0.25 and lin[1] == pytest.approx(0.04 / 12.92)
