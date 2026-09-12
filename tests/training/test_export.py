import json
import os

import numpy as np

from neurofly_core.artifact import load_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_training.envs import ENV_ARGS, make_env, resolve_env_args
from neurofly_training.export import export_run, mlp_from_sb3
from tests.training.test_pc import _write_bars_video


class _Args:
    pass


def _config(video, **over):
    a = _Args()
    for k in ENV_ARGS:
        setattr(a, k, None)
    a.task, a.brain, a.synthetic_n = video, "synthetic", 1200
    a.dt, a.brain_ms, a.brain_gain = 0.5, 5.0, 1.0
    a.include_proprio = a.plasticity = a.mouse = a.scroll = a.dry_run = False
    a.include_frame = a.include_audio = False
    a.encoder_gain, a.mouse_speed, a.fps, a.monitor = 12.0, 50.0, 10.0, 1
    a.retina_mode, a.retina_gain, a.retina_temporal = "auto", 15.0, 0.0
    a.audio_gain, a.dopamine_punish, a.keys = 15.0, 0.0, "a,d"
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_export_imitation_run(synthetic_cx, tmp_path):
    video = tmp_path / "video.mp4"
    layout, _ = _write_bars_video(video, T=6)
    args = _config(str(video), mouse=True, audio="file")
    cfg = resolve_env_args(args)
    env = make_env(seed=0, **cfg)
    dec = ControlDecoder(env.observation_space.shape[0], env.layout, seed=5)
    run = tmp_path / "run"
    run.mkdir()
    dec.save(run / "decoder.npz")
    json.dump(dict(cfg, algo="imitation", seed=0), open(run / "config.json", "w"))
    out = export_run(str(run), str(tmp_path / "art"))
    assert validate(out) == []
    model = load_model(out)
    assert model.layout.names == env.layout.names and model.policy.kind == "linear"
    obs, _ = env.reset()
    model.reset()
    frame = env.render()
    feats = model.observe(frame, env.audio.read())
    assert np.allclose(model.policy(feats), dec(feats))
    env.close()


def test_export_ppo_policy_matches_sb3(synthetic_cx, tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    video = tmp_path / "video.mp4"
    layout, _ = _write_bars_video(video, T=12)
    args = _config(str(video), synthetic_n=800)
    cfg = resolve_env_args(args)
    venv = VecNormalize(DummyVecEnv([lambda: make_env(seed=0, **cfg)]), norm_obs=True,
                        norm_reward=True, clip_obs=10.0)
    ppo = PPO("MlpPolicy", venv, n_steps=16, batch_size=8, n_epochs=1, seed=0, device="cpu",
              policy_kwargs=dict(net_arch=dict(pi=[16, 16], vf=[16, 16])))
    ppo.learn(32)
    run = tmp_path / "run"
    run.mkdir()
    ppo.save(run / "model.zip")
    venv.save(str(run / "vecnormalize.pkl"))
    json.dump(dict(cfg, algo="ppo", seed=0), open(run / "config.json", "w"))
    venv.close()

    layout = ControlLayout(keys=["a", "d"])
    pol = mlp_from_sb3(ppo, layout, str(run / "vecnormalize.pkl"))
    rng = np.random.default_rng(0)
    for _ in range(5):
        raw = rng.normal(size=pol.n_features).astype(np.float32)
        venv2 = VecNormalize.load(str(run / "vecnormalize.pkl"),
                                  DummyVecEnv([lambda: make_env(seed=0, **cfg)]))
        venv2.training = False
        a_sb3, _ = ppo.predict(venv2.normalize_obs(raw[None]), deterministic=True)
        venv2.close()
        assert np.allclose(pol(raw), a_sb3[0], atol=1e-5)
    out = export_run(str(run), str(tmp_path / "art"))
    assert validate(out) == [] and os.path.exists(os.path.join(out, "policy", "W0.bin"))
    assert load_model(out).policy.kind == "mlp"


def test_build_base_artifact(tmp_path):
    from neurofly_training.cli.build import build_artifact
    out = build_artifact(str(tmp_path / "base"), brain="synthetic", synthetic_n=1000,
                         keys="w,a", mouse=True, include_frame=True)
    assert validate(out) == []
    model = load_model(out)
    assert model.policy is None
    assert model.layout.names == ["key:w", "key:a", "mouse:dx", "mouse:dy"]
    assert json.load(open(os.path.join(out, "manifest.json")))["extra"]["base"] is True
