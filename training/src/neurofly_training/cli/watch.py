"""Roll out a policy (body or PC, optionally with the brain in the loop) and record a video.

    neurofly watch runs/<ppo run>  --episodes 2 --video videos/ppo.mp4
    neurofly watch runs/<es run>   --video videos/es.mp4
    neurofly watch runs/<imitation run> --task data/recordings/run1/video.mp4
    neurofly watch --task forward --policy random --video videos/random.mp4
    neurofly watch --task forward --brain malecns --policy zero    # fly stands, brain runs
    neurofly watch --task forward --policy gait --poses assets/gait.json --video videos/gait.mp4

On the live screen (--task pc) this only prints what it would press; use play_pc.py
to let the brain control the PC.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurofly_core.experiments import (ProbeLog, activity_sinks, add_experiment_args,
                                       apply_experiments)
from neurofly_training.envs import ENV_ARGS, is_pc_task, list_tasks, make_env
from neurofly_training.body.export import PoseRecorder
from neurofly_training.experiments import apply_to_body_env
from neurofly_core.decode.linear import ControlDecoder
from neurofly_training.body.decoder import LinearDecoder

CONTROL_HZ = 500


def load_policy(policy: str, run: str | None, cfg: dict, env, venv):
    """Returns (venv, model, decoder) for the requested policy."""
    model = decoder = None
    if policy == "ppo":
        venv = VecNormalize.load(os.path.join(run, "vecnormalize.pkl"), venv)
        venv.training, venv.norm_reward = False, False
        model = PPO.load(os.path.join(run, "model.zip"), device="cpu")
    elif policy in ("es", "imitation", "fixed"):
        if is_pc_task(cfg["task"]):
            decoder = ControlDecoder(env.observation_space.shape[0], env.layout,
                                     seed=cfg.get("seed", 0))
        else:
            decoder = LinearDecoder(env.pops, env.readout_idx, seed=cfg.get("seed", 0))
        if policy != "fixed":
            best = os.path.join(run, "decoder_best.npz")
            decoder.load(best if os.path.exists(best) else os.path.join(run, "decoder.npz"))
    return venv, model, decoder


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", default=None)
    p.add_argument("--policy", default=None,
                   choices=["ppo", "es", "imitation", "fixed", "random", "zero", "gait"],
                   help="default: the run's algorithm, or 'zero' without a run; 'gait' is an "
                        "open-loop tripod gait (body tasks), to see the limbs work")
    p.add_argument("--gait-hz", type=float, default=2.0, help="gait: stride cycles per second")
    p.add_argument("--task", default=None, help=", ".join(list_tasks()))
    p.add_argument("--brain", default=None, choices=["none", "malecns", "synthetic", "toy"])
    p.add_argument("--subset", default=None)
    p.add_argument("--keys", default=None, help="PC: keys the brain may hold")
    p.add_argument("--buttons", default=None, help="PC: mouse buttons it may hold")
    p.add_argument("--mouse", action="store_true", help="PC: it may move the mouse")
    p.add_argument("--window", default=None, help="PC: window to capture")
    p.add_argument("--region", default=None, help="PC: left,top,width,height to capture")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=None, help="cap on steps per episode")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--video", default=None)
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--every", type=int, default=None,
                   help="render every N control steps (default 10 for the body, 1 for the PC)")
    p.add_argument("--poses", default=None,
                   help="body: write world poses of every body per rendered step to this JSON "
                        "(for examples/three_viewer.html with `neurofly export-body`)")
    p.add_argument("--pose-ws", default=None, metavar="HOST:PORT",
                   help="body: stream poses live over a WebSocket and pace to real time "
                        "(examples/three_viewer.html?ws=ws://HOST:PORT)")
    add_experiment_args(p)
    args = p.parse_args()

    cfg = {}
    if args.run:
        with open(os.path.join(args.run, "config.json")) as f:
            cfg = json.load(f)
    env_kwargs = {k: cfg.get(k) for k in ENV_ARGS if k in cfg}
    env_kwargs.setdefault("task", "forward")
    env_kwargs.setdefault("brain", "none")
    for k in ("task", "brain", "subset", "keys", "buttons", "window", "region"):
        if getattr(args, k) is not None:
            env_kwargs[k] = getattr(args, k)
    if args.mouse:
        env_kwargs["mouse"] = True
    pc = is_pc_task(env_kwargs["task"])
    if pc and env_kwargs["brain"] == "none":
        env_kwargs["brain"] = "malecns"
    if pc:
        env_kwargs["dry_run"] = True   # a viewer never touches the PC
        if args.max_steps is not None:
            env_kwargs["max_steps"] = args.max_steps
    if env_kwargs.get("brain") == "none":
        env_kwargs["subset"] = None  # a stale subset from a PC run is meaningless here
    policy = args.policy or cfg.get("algo") or "zero"
    every = args.every or (1 if pc else 10)

    env = make_env(seed=args.seed, render_mode="rgb_array", camera_id=args.camera, **env_kwargs)
    if pc:
        for line in apply_experiments(env.model, args):
            print(line)
        probe = ProbeLog(env.model, args.probe_out)
    else:
        for line in apply_to_body_env(env, args):
            print(line)
        probe = None
    sinks = None
    if getattr(env, "last_activity", "no") != "no" or pc:
        sinks = activity_sinks(env.model if pc else env, args,
                               fps=(env.fps if pc else CONTROL_HZ / every))
        for line in sinks.describe():
            print(line)
    gait = None
    if policy == "gait":
        if pc:
            raise SystemExit("--policy gait is for body tasks")
        from neurofly_training.body.gait import TripodGait, rest_action
        gait = TripodGait(control_hz=CONTROL_HZ, stride_hz=args.gait_hz,
                          rest=rest_action(env.physics.model))
    venv = DummyVecEnv([lambda: env])
    venv, model, decoder = load_policy(policy, args.run, dict(cfg, task=env_kwargs["task"]),
                                       env, venv)

    frames = []
    poses = caster = None
    if args.poses and not pc:
        poses = PoseRecorder(lambda: env.physics, fps=CONTROL_HZ / every)
    if args.pose_ws and not pc:
        from neurofly_training.body.stream import PoseBroadcaster
        host, _, port = args.pose_ws.rpartition(":")
        caster = PoseBroadcaster(lambda: env.physics, fps=CONTROL_HZ / every,
                                 host=host or "127.0.0.1", port=int(port))
        print(f"streaming poses on ws://{caster.host}:{caster.port}; pacing to real time")
    wall = time.perf_counter()
    obs = venv.reset()
    for ep in range(args.episodes):
        start = env.root_position() if hasattr(env, "root_position") else None
        ret, steps, spikes, done = 0.0, 0, 0, False
        held = Counter()
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
            elif decoder is not None:
                action = np.asarray(decoder(obs[0]))[None]
            elif policy == "random":
                action = np.stack([venv.action_space.sample()])
            elif policy == "gait":
                action = gait(steps)[None]
            else:
                action = np.zeros((1,) + venv.action_space.shape, dtype=venv.action_space.dtype)
            last = env.root_position() if start is not None else None
            if args.video and steps % every == 0:
                frames.append(env.render())
            if poses is not None and steps % every == 0:
                poses.record()
            if caster is not None and steps % every == 0:
                caster.broadcast()
                wall += every / CONTROL_HZ
                lag = wall - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
            obs, reward, dones, infos = venv.step(action)
            if probe is not None:
                probe.record()
            if sinks is not None and steps % every == 0:
                sinks.record()
            ret += float(reward[0])
            spikes += int(infos[0].get("brain_spikes", 0))
            held.update(infos[0].get("held", []))
            steps += 1
            done = bool(dones[0]) or (args.max_steps is not None and steps >= args.max_steps)
        extra = f"  brain spikes {spikes:,}" if env_kwargs["brain"] != "none" else ""
        if start is not None:
            d = last - start
            extra += f"  forward {d[0]:+.3f} cm  sideways {d[1]:+.3f} cm"
        elif held:
            extra += "  held " + ", ".join(f"{k} x{n}" for k, n in held.most_common())
        else:
            extra += "  nothing held"
        print(f"episode {ep}: return {ret:8.2f}  steps {steps:5d}{extra}")
        if args.max_steps is not None and not bool(dones[0]):
            obs = venv.reset()

    if args.video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        fps = env.metadata.get("render_fps", 10) if pc else CONTROL_HZ // every
        with imageio.get_writer(args.video, fps=fps, macro_block_size=1) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"wrote {len(frames)} frames to {args.video}")
    if probe is not None and probe.save():
        print(f"probe written to {args.probe_out}")
    if sinks is not None:
        written = sinks.close()
        if written:
            print(f"activity written to {written}")
    if caster is not None:
        caster.close()
    if poses is not None:
        os.makedirs(os.path.dirname(args.poses) or ".", exist_ok=True)
        path = poses.save(args.poses)
        print(f"wrote {len(poses.frames)} poses of {len(poses.bodies)} bodies to {path}")
    venv.close()


if __name__ == "__main__":
    main()
