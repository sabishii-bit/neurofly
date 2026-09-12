"""Train a policy with Stable-Baselines3 PPO, with or without the brain in the loop.

    # body only: an MLP policy on proprioception (fast baseline)
    python scripts/train.py --task forward --brain none --timesteps 2000000 --n-envs 8

    # brain in the loop: the policy reads motor/descending neuron rates
    python scripts/train.py --task forward --brain malecns --subset vnc --n-envs 4 --timesteps 500000

    # same, plus reward acting as dopamine on synapses onto leg motor neurons
    python scripts/train.py --task forward --brain malecns --plasticity

Must be run as a script: Windows spawns a process per environment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from flybrain_body.body.tasks import TASKS
from flybrain_body.data.connectome import SUBSETS
from flybrain_body.envs import ENV_ARGS, make_env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # environment
    p.add_argument("--task", default="forward", choices=sorted(TASKS))
    p.add_argument("--brain", default="none", choices=["none", "malecns", "synthetic"])
    p.add_argument("--subset", default="vnc", choices=sorted(SUBSETS))
    p.add_argument("--dt", type=float, default=0.5, help="brain step, ms")
    p.add_argument("--readout", default="motor+descending")
    p.add_argument("--include-proprio", action="store_true",
                   help="also give the policy the raw body observation")
    p.add_argument("--plasticity", action="store_true", help="reward acts as dopamine")
    p.add_argument("--brain-gain", type=float, default=1.0)
    p.add_argument("--encoder-gain", type=float, default=12.0)
    p.add_argument("--synthetic-n", type=int, default=3000)
    # training
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="default 8 body-only, 4 with brain")
    p.add_argument("--n-steps", type=int, default=1024, help="PPO rollout per env")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="run directory to continue")
    p.add_argument("--checkpoint-every", type=int, default=200_000)
    return p.parse_args()


def main():
    args = parse_args()
    if args.resume:
        run_dir = args.resume
        with open(os.path.join(run_dir, "config.json")) as f:
            saved = json.load(f)
        for k in ENV_ARGS:
            setattr(args, k, saved[k])
    else:
        name = args.run_name or f"{args.task}_{args.brain}_{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir = os.path.join("runs", name)
    if args.n_envs is None:
        args.n_envs = 8 if args.brain == "none" else 4
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")

    env_kwargs = {k: getattr(args, k) for k in ENV_ARGS}
    fns = [partial(make_env, seed=args.seed + i, device=args.device, **env_kwargs)
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if args.n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    if args.resume:
        venv = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), venv)
        model = PPO.load(os.path.join(run_dir, "model.zip"), env=venv, device=args.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device=args.device,
                    n_steps=args.n_steps, batch_size=256, n_epochs=10, learning_rate=3e-4,
                    gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       log_std_init=-1.0),
                    tensorboard_log=os.path.join(run_dir, "tb"))
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(dict(vars(args), algo="ppo"), f, indent=2)

    checkpoint = CheckpointCallback(save_freq=max(args.checkpoint_every // args.n_envs, 1),
                                    save_path=os.path.join(run_dir, "checkpoints"),
                                    name_prefix="ckpt", save_vecnormalize=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint,
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("interrupted, saving")
    finally:
        model.save(os.path.join(run_dir, "model.zip"))
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        venv.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
