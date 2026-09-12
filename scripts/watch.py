"""Roll out a policy in the body (optionally with the brain in the loop) and record a video.

    python scripts/watch.py runs/<ppo run>  --episodes 2 --video videos/ppo.mp4
    python scripts/watch.py runs/<es run>   --video videos/es.mp4
    python scripts/watch.py --task forward --policy random --video videos/random.mp4
    python scripts/watch.py --task forward --brain malecns --policy zero      # brain activity, fly stands
"""
from __future__ import annotations

import argparse
import json
import os

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flybrain_body.body.tasks import TASKS
from flybrain_body.envs import ENV_ARGS, make_env
from flybrain_body.interface.decoder import LinearDecoder

CONTROL_HZ = 500


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("--policy", default=None, choices=["ppo", "es", "random", "zero"],
                   help="default: the run's algorithm, or 'zero' without a run")
    p.add_argument("--task", default=None, choices=sorted(TASKS))
    p.add_argument("--brain", default=None, choices=["none", "malecns", "synthetic"])
    p.add_argument("--subset", default=None)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--video", default=None)
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--every", type=int, default=10, help="render every N control steps")
    args = p.parse_args()

    cfg = {}
    if args.run:
        with open(os.path.join(args.run, "config.json")) as f:
            cfg = json.load(f)
    env_kwargs = {k: cfg.get(k) for k in ENV_ARGS if k in cfg}
    env_kwargs.setdefault("task", "forward")
    env_kwargs.setdefault("brain", "none")
    for k in ("task", "brain", "subset"):
        if getattr(args, k) is not None:
            env_kwargs[k] = getattr(args, k)
    policy = args.policy or cfg.get("algo") or "zero"

    env = make_env(seed=args.seed, render_mode="rgb_array", camera_id=args.camera, **env_kwargs)
    venv = DummyVecEnv([lambda: env])
    model = decoder = None
    if policy == "ppo":
        venv = VecNormalize.load(os.path.join(args.run, "vecnormalize.pkl"), venv)
        venv.training, venv.norm_reward = False, False
        model = PPO.load(os.path.join(args.run, "model.zip"), device="cpu")
    elif policy == "es":
        decoder = LinearDecoder(env.pops, env.readout_idx, seed=cfg.get("seed", 0))
        best = os.path.join(args.run, "decoder_best.npz")
        decoder.load(best if os.path.exists(best) else os.path.join(args.run, "decoder.npz"))

    rng = np.random.default_rng(args.seed)
    frames = []
    obs = venv.reset()
    for ep in range(args.episodes):
        start = env.root_position()
        ret, steps, spikes, done = 0.0, 0, 0, False
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            elif decoder is not None:
                action = decoder(obs[0])[None]
            elif policy == "random":
                action = rng.uniform(-1, 1, size=(1,) + venv.action_space.shape).astype(np.float32)
            else:
                action = np.zeros((1,) + venv.action_space.shape, dtype=np.float32)
            last = env.root_position()
            if args.video and steps % args.every == 0:
                frames.append(env.render())
            obs, reward, dones, infos = venv.step(action)
            ret += float(reward[0])
            spikes += int(infos[0].get("brain_spikes", 0))
            steps += 1
            done = bool(dones[0])
        d = last - start
        extra = f"  brain spikes {spikes:,}" if env_kwargs["brain"] != "none" else ""
        print(f"episode {ep}: return {ret:8.2f}  steps {steps:5d}  "
              f"forward {d[0]:+.3f} cm  sideways {d[1]:+.3f} cm{extra}")

    if args.video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        with imageio.get_writer(args.video, fps=CONTROL_HZ // args.every) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"wrote {len(frames)} frames to {args.video}")
    venv.close()


if __name__ == "__main__":
    main()
