import numpy as np

from flybrain_body.envs import BrainInLoopEnv, make_body_env, make_env
from flybrain_body.interface.decoder import LinearDecoder


def test_brain_in_loop_env(synthetic_cx):
    body = make_body_env("forward", seed=0)
    env = BrainInLoopEnv(body, synthetic_cx, dt=0.5, seed=0)
    assert env.substeps == 4
    assert env.encoder.n_driven > 0
    feats, _ = env.reset()
    assert feats.shape == env.observation_space.shape and feats.dtype == np.float32
    total = 0
    for _ in range(25):
        feats, r, term, trunc, info = env.step(np.zeros(59, np.float32))
        total += info["brain_spikes"]
        assert np.isfinite(feats).all()
    assert total > 0, "sensory drive from the standing fly should make the brain spike"
    env.close()


def test_make_env_synthetic_with_proprio_and_plasticity():
    env = make_env("forward", seed=1, brain="synthetic", synthetic_n=1200,
                   include_proprio=True, plasticity=True)
    assert env.plasticity is not None and env.plasticity.n_edges > 0
    feats, _ = env.reset()
    assert feats.shape[0] == len(env.readout_idx) + env.body.observation_space.shape[0]
    env.step(env.action_space.sample())
    env.close()


def test_linear_decoder_roundtrip(synthetic_cx):
    body = make_body_env("forward", seed=0)
    env = BrainInLoopEnv(body, synthetic_cx, seed=0)
    dec = LinearDecoder(env.pops, env.readout_idx, seed=0)
    feats, _ = env.reset()
    a = dec(feats)
    assert a.shape == (59,) and np.all(np.abs(a) <= 1)
    theta = dec.get_params()
    assert theta.shape == (dec.n_params,)
    dec.set_params(theta * 2)
    assert np.allclose(dec.get_params(), theta * 2)
    env.close()
