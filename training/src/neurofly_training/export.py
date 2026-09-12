"""A run directory -> an artifact directory for neurofly-core.

The model is rebuilt from the run's config (same connectome subset, same
encoders, same readout, same layout), the trained policy is converted to plain
arrays (a linear table for ES and imitation runs, an MLP with its observation
normalisation for PPO runs), and everything is written as an artifact.

Body runs are not exportable: the body lives in MuJoCo, which the runtime
does not carry.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from neurofly_core.artifact import save_model
from neurofly_core.body import ActuatorDecoder, ActuatorLayout, BodyModel, ProprioMap
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.decode.mlp import MLPPolicy
from neurofly_core.model import ModelConfig
from neurofly_training.body.actuators import ACTUATOR_NAMES
from neurofly_training.envs import ENV_ARGS, is_pc_task, make_env, make_layout, make_pc_model


def mlp_from_sb3(ppo, layout, vecnormalize_path: str | None = None) -> MLPPolicy:
    """The deterministic action of a Stable-Baselines3 PPO MlpPolicy as an ``MLPPolicy``."""
    import torch.nn as nn
    pol = ppo.policy
    layers, activation = [], "tanh"
    for mod in pol.mlp_extractor.policy_net:
        if isinstance(mod, nn.Linear):
            layers.append((mod.weight.detach().cpu().numpy(), mod.bias.detach().cpu().numpy()))
        elif isinstance(mod, nn.Tanh):
            activation = "tanh"
        elif isinstance(mod, nn.ReLU):
            activation = "relu"
    an = pol.action_net
    layers.append((an.weight.detach().cpu().numpy(), an.bias.detach().cpu().numpy()))
    obs_mean = obs_var = None
    clip, eps = 10.0, 1e-8
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        with open(vecnormalize_path, "rb") as f:
            vn = pickle.load(f)
        if getattr(vn, "norm_obs", False) and vn.obs_rms is not None:
            obs_mean, obs_var = np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var)
            clip, eps = float(vn.clip_obs), float(vn.epsilon)
    return MLPPolicy(layout, layers, activation=activation, obs_mean=obs_mean, obs_var=obs_var,
                     obs_clip=clip, obs_eps=eps)


def load_run_policy(run_dir: str, cfg: dict, layout, n_features: int, body_env=None):
    algo = cfg.get("algo")
    if algo == "ppo":
        from stable_baselines3 import PPO
        ppo = PPO.load(os.path.join(run_dir, "model.zip"), device="cpu")
        if ppo.policy.mlp_extractor.policy_net[0].in_features != n_features:
            raise ValueError("the run's policy expects a different feature size than the "
                             "rebuilt model produces; the config does not match")
        return mlp_from_sb3(ppo, layout, os.path.join(run_dir, "vecnormalize.pkl"))
    if algo in ("es", "imitation") and body_env is None:
        dec = ControlDecoder(n_features, layout)
        best = os.path.join(run_dir, "decoder_best.npz")
        dec.load(best if os.path.exists(best) else os.path.join(run_dir, "decoder.npz"))
        return dec
    if algo == "es" and body_env is not None:
        from neurofly_training.body.decoder import LinearDecoder
        dec = LinearDecoder(body_env.pops, body_env.readout_idx, seed=cfg.get("seed", 0))
        best = os.path.join(run_dir, "decoder_best.npz")
        dec.load(best if os.path.exists(best) else os.path.join(run_dir, "decoder.npz"))
        W = np.zeros((layout.n, n_features))
        for acts, feats, Wb in dec.blocks:
            W[np.ix_(acts, feats)] = Wb
        return ActuatorDecoder(n_features, layout, W=W, b=dec.bias)
    raise ValueError(f"run {run_dir} has algorithm {algo!r}, which has no exportable policy")


def body_model_from_env(env, name: str, meta: dict | None = None) -> BodyModel:
    """A runtime BodyModel with the same brain, sensory map and readout as a BrainInLoopEnv."""
    enc = env.encoder
    touch = (enc.touch_slice.start, enc.touch_slice.stop) if enc.touch_slice is not None else None
    proprio = ProprioMap(env.cx.n, matrix=enc.matrix, gain=enc.gain, touch=touch,
                         touch_scale=enc.touch_scale, device=str(env.brain.device))
    slices = {k: [s.start, s.stop] for k, s in env.body.obs_slices.items()}
    config = ModelConfig(dt=env.brain.dt, brain_ms=env.body.control_timestep * 1000.0,
                         include_proprio=env.include_proprio,
                         plasticity=env.plasticity is not None, name=name,
                         meta=dict(meta or {}, connectome=env.cx.name))
    return BodyModel(env.brain, readout_idx=env.readout_idx, proprio=proprio,
                     layout=ActuatorLayout(ACTUATOR_NAMES), obs_keys=env.body.obs_keys,
                     obs_slices=slices, config=config, punish_idx=env.pops.ppl1,
                     neuron_ids=env.cx.neurons["bodyId"].values,
                     neuron_types=env.cx.neurons["type"].values,
                     neuron_superclass=env.cx.neurons["superclass"].values)


def export_body_run(run_dir: str, cfg: dict, out_dir: str, name: str | None, device: str) -> str:
    opts = {k: cfg[k] for k in ENV_ARGS if k in cfg}
    if opts.get("brain") in (None, "none"):
        raise ValueError("a body run without a brain has nothing to export")
    env = make_env(seed=cfg.get("seed", 0), device=device, **opts)
    name = name or os.path.basename(os.path.normpath(run_dir))
    model = body_model_from_env(env, name, {"brain": opts["brain"], "subset": opts.get("subset")})
    model.policy = load_run_policy(run_dir, cfg, model.layout, model.n_features, body_env=env)
    env.close()
    return save_model(model, out_dir, extra={"run": os.path.abspath(run_dir),
                                             "algo": cfg.get("algo"), "config": cfg})


def export_run(run_dir: str, out_dir: str, name: str | None = None, device: str = "cpu") -> str:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    if not is_pc_task(cfg.get("task", "")):
        return export_body_run(run_dir, cfg, out_dir, name, device)
    opts = {k: cfg[k] for k in ENV_ARGS if k in cfg}
    layout = make_layout(opts.get("keys"), opts.get("buttons"), opts.get("mouse", False),
                         opts.get("scroll", False), opts.get("mouse_speed", 50.0),
                         opts.get("pad_buttons"), opts.get("axes"))
    name = name or os.path.basename(os.path.normpath(run_dir))
    model = make_pc_model(layout=layout, device=device, name=name,
                          **{k: v for k, v in opts.items()
                             if k not in ("keys", "buttons", "mouse", "scroll", "mouse_speed",
                                          "pad_buttons", "axes")})
    model.policy = load_run_policy(run_dir, cfg, layout, model.n_features)
    return save_model(model, out_dir, extra={"run": os.path.abspath(run_dir),
                                             "algo": cfg.get("algo"), "config": cfg})
