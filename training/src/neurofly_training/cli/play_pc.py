"""Let the brain use the PC: the screen (and sound) in, keyboard and mouse out.

    # see what it would do, without touching anything
    neurofly play --window "My App" --keys w,a,s,d --mouse --dry-run

    # a trained neuron-to-control map (train_es.py, train_imitation.py) or a PPO policy
    neurofly play --window "My App" --run runs/imitate_myapp
    neurofly play --window "My App" --run runs/pc_malecns_...

    # a random neuron-to-control map, with what the PC is playing as sound input
    neurofly play --region 0,0,800,600 --keys w,a,s,d,space --buttons left \
        --mouse --audio loopback --policy fixed --seed 3

Press Esc to stop (the panic key). Give the target window focus during the countdown;
the brain's key presses and mouse motion go wherever the focus is.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from neurofly_training.envs import ENV_ARGS, add_env_args, make_env, resolve_env_args
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.experiments import (ProbeLog, activity_sinks, add_experiment_args,
                                       apply_experiments)
from neurofly_core.io.controls import PanicKey
from neurofly_core.io.guard import FocusGuard, Watchdog
from neurofly_training.pc.dagger import CorrectionRecorder


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run directory with a decoder or PPO model")
    p.add_argument("--policy", default=None,
                   choices=["ppo", "es", "imitation", "fixed", "random"],
                   help="default: the run's algorithm, or 'fixed' without a run")
    p.add_argument("--steps", type=int, default=None, help="stop after this many steps")
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--panic", default="esc", help="key that stops everything")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--record", default=None, help="save the captured frames to this mp4")
    p.add_argument("--correct-out", default=None, metavar="DIR",
                   help="DAgger: your inputs override the fly and are recorded here as labels")
    p.add_argument("--no-focus-guard", action="store_true",
                   help="keep going even when the keyboard focus leaves the target window")
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    add_experiment_args(p)
    args = p.parse_args()
    args.task = "pc"

    cfg = {}
    if args.run:
        with open(os.path.join(args.run, "config.json")) as f:
            cfg = json.load(f)
        for k in ENV_ARGS:
            untouched = p.get_default(k) == getattr(args, k)
            if k in cfg and cfg[k] is not None and k != "task" and untouched:
                setattr(args, k, cfg[k])
    args.max_steps = args.steps
    args.reward = None   # no task on the live screen: just act
    env_kwargs = resolve_env_args(args)
    policy = args.policy or cfg.get("algo") or "fixed"

    env = make_env(seed=args.seed, device=args.device, **env_kwargs)
    print(f"capturing {env.video.describe()} at {args.fps:g} fps"
          + (f"; sound from {env.audio.name}" if env.audio is not None else "")
          + f"; controls {env.layout.names}; policy {policy}"
          + ("; DRY RUN" if args.dry_run else ""))
    print(env.model.describe())
    for line in apply_experiments(env.model, args):
        print(line)
    probe = ProbeLog(env.model, args.probe_out)
    sinks = activity_sinks(env.model, args, fps=args.fps)
    for line in sinks.describe():
        print(line)

    model = decoder = venv = None
    if policy == "ppo":
        venv = VecNormalize.load(os.path.join(args.run, "vecnormalize.pkl"),
                                 DummyVecEnv([lambda: env]))
        venv.training = False
        model = PPO.load(os.path.join(args.run, "model.zip"), device="cpu")
    elif policy in ("es", "imitation", "fixed"):
        decoder = ControlDecoder(env.observation_space.shape[0], env.layout, seed=args.seed)
        if policy != "fixed":
            best = os.path.join(args.run, "decoder_best.npz")
            decoder.load(best if os.path.exists(best) else os.path.join(args.run, "decoder.npz"))
    rng = np.random.default_rng(args.seed)

    writer = None
    if args.record:
        import imageio
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        writer = imageio.get_writer(args.record, fps=args.fps, macro_block_size=1)

    panic = PanicKey(args.panic)
    corrections = None
    if args.correct_out:
        corrections = CorrectionRecorder(env.layout, args.correct_out, args.fps, panic=args.panic)
    for i in range(int(args.countdown), 0, -1):
        print(f"starting in {i} ... focus the target window; {args.panic} stops")
        time.sleep(1.0)
    guard = FocusGuard(enabled=not args.no_focus_guard and not args.dry_run)
    guard.arm()
    print(guard.describe())
    watchdog = Watchdog(env.controls, timeout=max(1.0, 5.0 / args.fps))
    obs, _ = env.reset()
    t0, last_report, spikes, steps = time.time(), time.time(), 0, 0
    try:
        while not panic.stopped and not (corrections and corrections.stopped):
            if not guard.ok():
                print("keyboard focus left the target window: stopping")
                break
            watchdog.heartbeat()
            if corrections is not None and env.render() is not None:
                human = corrections.step(env.render())
                if human is not None:
                    env.controls.release_all()
                    obs = env.model.observe(env.video.read(),
                                            env.audio.read() if env.audio is not None else None)
                    env.clock.wait()
                    continue
            if model is not None:
                action, _ = model.predict(venv.normalize_obs(obs[None]), deterministic=True)
                action = action[0]
            elif decoder is not None:
                action = decoder(obs)
            else:
                action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            obs, _, term, trunc, info = env.step(action)
            probe.record()
            sinks.record()
            spikes += info["brain_spikes"]
            steps += 1
            if writer is not None:
                writer.append_data(env.render())
            if time.time() - last_report >= 1.0:
                held = ", ".join(info["held"]) or "-"
                print(f"t={time.time() - t0:6.1f}s  step {steps:5d}  holding [{held}]  "
                      f"spikes/s {spikes / max(time.time() - last_report, 1e-6):,.0f}")
                last_report, spikes = time.time(), 0
            if term or trunc:
                break
    except KeyboardInterrupt:
        pass
    finally:
        watchdog.close()
        env.close()
        panic.close()
        if writer is not None:
            writer.close()
        if corrections is not None:
            out = corrections.close()
            print(f"corrections: {corrections.taken_over} frames of yours -> {out}")
    saved = probe.save()
    activity = sinks.close()
    print(f"stopped after {steps} steps; everything released"
          + (f"; probe written to {saved}" if saved else "")
          + (f"; activity written to {activity}" if activity else ""))


if __name__ == "__main__":
    main()
