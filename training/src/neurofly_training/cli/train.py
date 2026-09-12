"""Train a policy with Stable-Baselines3 PPO, with or without the brain in the loop.

    # body only: an MLP policy on proprioception (fast baseline)
    neurofly train --task forward --brain none --timesteps 2000000 --n-envs 8

    # brain in the loop: the policy reads motor/descending neuron rates
    neurofly train --task forward --brain malecns --n-envs 4 --timesteps 500000

    # same, plus reward acting as dopamine on synapses onto leg motor neurons
    neurofly train --task forward --brain malecns --plasticity

    # the PC: the screen goes in through the retina, the policy holds keys and moves the mouse;
    # your Task subclass supplies reward and restarts (one env: there is one screen)
    neurofly train --task pc --brain malecns --window "My App" --keys w,a,s,d --mouse \
        --reward my_project.py:MyTask --max-steps 600

Must be run as a script: Windows spawns a process per environment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from neurofly_training.envs import ENV_ARGS, add_env_args, make_env, resolve_env_args


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(p)
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
            setattr(args, k, saved.get(k))
    env_kwargs = resolve_env_args(args)
    if not args.resume:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = args.run_name or f"{os.path.basename(args.task)}_{args.brain}_{stamp}"
        run_dir = os.path.join("runs", name)
    if args.task == "pc":
        if args.n_envs not in (None, 1):
            print("the live screen is one environment: using --n-envs 1")
        args.n_envs = 1
    if args.n_envs is None:
        args.n_envs = 8 if args.brain == "none" else 4
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir}")

    fns = [partial(make_env, seed=args.seed + i, device=args.device, **env_kwargs)
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if args.n_envs > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    if args.resume:
        venv = VecNormalize.load(os.path.join(run_dir, "vecnormalize.pkl"), venv)
        model = PPO.load(os.path.join(run_dir, "model.zip"), env=venv, device=args.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        if isinstance(venv.action_space, spaces.Box):
            policy_kwargs["log_std_init"] = -1.0
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device=args.device,
                    n_steps=args.n_steps, batch_size=256, n_epochs=10, learning_rate=3e-4,
                    gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=policy_kwargs, tensorboard_log=os.path.join(run_dir, "tb"))
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
