import json

import numpy as np
import pytest

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.body import ActuatorDecoder, BodyModel
from neurofly_core.server import Session
from neurofly_training.body.decoder import LinearDecoder
from neurofly_training.envs import make_env
from neurofly_training.export import body_model_from_env, export_run


def _body_env():
    return make_env("forward", seed=0, brain="synthetic", synthetic_n=1200)


def test_body_model_matches_env_and_roundtrips(tmp_path):
    env = _body_env()
    model = body_model_from_env(env, "walk")
    assert isinstance(model, BodyModel) and model.n_obs == env.body.observation_space.shape[0]
    assert model.n_actions == 59 and model.layout.names[0] == "head_abduct"
    # same brain, same sensory map: the features match the training env step for step
    obs0, _ = env.reset()
    body_obs = env._last_body_obs.copy()
    model.reset()
    f0 = model.observe(body_obs)
    assert np.allclose(f0[:len(model.readout_idx)], obs0[:len(model.readout_idx)], atol=1e-6)
    seq = []
    for _ in range(3):
        o, _, _, _, _ = env.step(np.zeros(59, np.float32))
        seq.append((env._last_body_obs.copy(), o))
    for body_obs, o in seq:
        f = model.observe(body_obs)
        assert np.allclose(f, o, atol=1e-6)
    # through the artifact
    W = np.random.default_rng(0).normal(scale=0.1, size=(59, model.n_features))
    model.policy = ActuatorDecoder(model.n_features, model.layout, W=W)
    path = save_model(model, str(tmp_path / "body"))
    assert validate(path) == []
    assert "kind body" in describe(path) and "actuators: 59" in describe(path)
    again = load_model(path)
    assert isinstance(again, BodyModel) and again.obs_keys == model.obs_keys
    model.reset()
    again.reset()
    a1, i1 = model.step(seq[0][0])
    a2, i2 = again.step(seq[0][0])
    assert np.allclose(a1, a2) and a1.shape == (59,) and i1["spikes"] == i2["spikes"]
    assert "sensory" in again.populations() and len(again.select({"name": "sensory"})) > 0
    env.close()


def test_export_es_body_run(tmp_path):
    env = _body_env()
    dec = LinearDecoder(env.pops, env.readout_idx, seed=0)
    dec.set_params(np.random.default_rng(1).normal(scale=0.1, size=dec.n_params))
    run = tmp_path / "run"
    run.mkdir()
    dec.save(run / "decoder.npz")
    cfg = {"task": "forward", "brain": "synthetic", "subset": "vnc", "dt": 0.5,
           "readout": "motor+descending", "include_proprio": False, "plasticity": False,
           "brain_gain": 1.0, "encoder_gain": 12.0, "synthetic_n": 1200, "algo": "es", "seed": 0}
    json.dump(cfg, open(run / "config.json", "w"))
    out = export_run(str(run), str(tmp_path / "art"))
    assert validate(out) == []
    model = load_model(out)
    obs, _ = env.reset()
    body_obs = env._last_body_obs
    model.reset()
    feats = model.observe(body_obs)
    assert np.allclose(model.act(feats), dec(feats), atol=1e-6)
    env.close()


def test_body_session_ops(tmp_path):
    env = _body_env()
    model = body_model_from_env(env, "walk")
    env.close()
    s = Session(model)
    info = s.handle({"op": "info"})
    assert info["kind"] == "body" and info["n_obs"] == model.n_obs and "slices" in info["obs"]
    obs = np.zeros(model.n_obs).tolist()
    r = s.handle({"op": "body_step", "obs": obs})
    assert r["ok"] and len(r["features"]) == model.n_features        # no policy: features
    assert s.handle({"op": "step", "frame": "AA==", "width": 1, "height": 1})["ok"] is False
    W = np.zeros((59, model.n_features)).tolist()
    r = s.handle({"op": "set_policy", "type": "linear", "W": W, "b": [0.5] * 59})
    assert r["type"] == "linear"
    r = s.handle({"op": "body_step", "obs": obs, "reward": -1.0})
    assert r["ok"] and len(r["action"]) == 59 and abs(r["action"][0] - np.tanh(0.5)) < 1e-5
    assert s.handle({"op": "probe", "name": "sensory"})["n"] > 0
    assert "probe" in s.handle({"op": "body_observe", "obs": obs})
    out = s.handle({"op": "save", "path": str(tmp_path / "saved")})
    assert out["ok"] and load_model(out["path"]).kind == "body"


def test_body_grpc(tmp_path):
    grpc = pytest.importorskip("grpc")
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    env = _body_env()
    model = body_model_from_env(env, "walk")
    env.close()
    model.policy = ActuatorDecoder(model.n_features, model.layout)
    server = make_server(model, "127.0.0.1:0")
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        info = stub.Info(pb.Empty())
        assert info.kind == "body" and info.n_obs == model.n_obs and len(info.controls) == 59
        assert json.loads(info.obs_json)["keys"] == model.obs_keys
        r = stub.BodyStep(pb.BodyRequest(obs=[0.0] * model.n_obs))
        assert r.ok and len(r.action) == 59 and r.t == 1
        replies = list(stub.BodyStream(iter([pb.BodyRequest(obs=[0.0] * model.n_obs)] * 3)))
        assert [x.t for x in replies] == [2, 3, 4]
        bad = stub.Step(pb.StepRequest(frame=b"\x00" * 3, width=1, height=1))
        assert not bad.ok and "body" in bad.error
    finally:
        server.stop(0)


def test_pose_broadcaster():
    from websockets.sync.client import connect
    from neurofly_training.body.stream import PoseBroadcaster
    from neurofly_training.envs import make_body_env
    env = make_body_env("forward", seed=0)
    env.reset()
    caster = PoseBroadcaster(lambda: env.physics, fps=50, port=0)
    try:
        with connect(f"ws://127.0.0.1:{caster.port}") as ws:
            header = json.loads(ws.recv())
            assert header["bodies"] == caster.recorder.bodies and header["fps"] == 50
            for _ in range(3):
                env.step(env.action_space.sample())
                caster.broadcast()
            msg = json.loads(ws.recv())
            assert msg["t"] == 0 and len(msg["pose"]) == 7 * len(header["bodies"])
    finally:
        caster.close()
        env.close()


def test_export_mjcf(tmp_path):
    from dm_control import mjcf
    import mujoco
    from neurofly_training.envs import make_body_env
    env = make_body_env("forward", seed=0)
    env.reset()
    mjcf.export_with_assets(env.dm_env.task.root_entity.mjcf_model, str(tmp_path / "mj"),
                            out_file_name="fly.xml")
    env.close()
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "mj" / "fly.xml"))
    assert m.nbody > 60 and m.nmesh > 80
