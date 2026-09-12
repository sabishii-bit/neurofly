"""Evolution strategies over the neuron-to-output map (actuators, or keys and mouse).

No gradients through the brain: a population of decoder weight vectors is
evaluated by rolling out the brain-in-the-loop env, and the mean is moved
toward the better ones (OpenAI-ES with antithetic sampling and rank
shaping). Each worker process owns one brain and one body; the live screen
is a single worker.

    neurofly es --task forward --brain malecns --workers 8 --generations 200
    neurofly es --task pc --brain malecns --window "My App" --keys w,a,s,d \
        --reward my_project.py:MyTask --episode-steps 300
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from neurofly_training.envs import ENV_ARGS, add_env_args, is_pc_task, make_env, resolve_env_args
from neurofly_core.decode.linear import ControlDecoder
from neurofly_training.body.decoder import LinearDecoder

_ENV = None
_DEC = None


def build_decoder(env, cfg: dict):
    if is_pc_task(cfg["task"]):
        return ControlDecoder(env.observation_space.shape[0], env.layout, seed=cfg["seed"])
    return LinearDecoder(env.pops, env.readout_idx, seed=cfg["seed"])


def _init_worker(cfg: dict):
    global _ENV, _DEC
    env_kwargs = {k: cfg[k] for k in ENV_ARGS if k in cfg}
    _ENV = make_env(seed=cfg["seed"], **env_kwargs)
    _DEC = build_decoder(_ENV, cfg)


def _evaluate(job):
    theta, max_steps = job
    _DEC.set_params(theta)
    feats, _ = _ENV.reset()
    ret, steps = 0.0, 0
    while steps < max_steps:
        feats, r, term, trunc, _ = _ENV.step(_DEC(feats))
        ret += r
        steps += 1
        if term or trunc:
            break
    return ret


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_args(p, brain_default="malecns", brain_choices=("malecns", "synthetic", "toy"))
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=32, help="even; half are antithetic")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--episode-steps", type=int, default=500,
                   help="body: 500 steps = 1 s of sim; PC: steps of 1/fps s")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    args.include_proprio = False
    cfg = resolve_env_args(args)
    cfg["seed"] = args.seed
    if args.task == "pc" and args.workers != 1:
        print("the live screen is one environment: using --workers 1")
        args.workers = 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = args.run_name or f"es_{os.path.basename(args.task)}_{args.brain}_{stamp}"
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), algo="es"), f, indent=2)

    _init_worker(cfg)
    dec = _DEC
    _ENV.close()
    theta = dec.get_params()
    print(f"run dir: {run_dir}; decoder has {len(theta)} parameters; {args.workers} workers")
    rng = np.random.default_rng(args.seed)
    half = args.population // 2
    best = -np.inf
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for gen in range(args.generations):
            t0 = time.time()
            eps = rng.standard_normal((half, len(theta)))
            cands = np.concatenate([theta + args.sigma * eps, theta - args.sigma * eps])
            returns = np.array(pool.map(_evaluate, [(c, args.episode_steps) for c in cands]))
            ranks = np.empty(len(returns))
            ranks[np.argsort(returns)] = np.arange(len(returns))
            util = ranks / (len(returns) - 1) - 0.5
            grad = (util[:half] - util[half:]) @ eps / (half * args.sigma)
            theta = theta + args.lr * grad
            dec.set_params(theta)
            dec.save(os.path.join(run_dir, "decoder.npz"))
            if returns.max() > best:
                best = returns.max()
                dec_best = build_decoder(_ENV, cfg)
                dec_best.set_params(cands[int(returns.argmax())])
                dec_best.save(os.path.join(run_dir, "decoder_best.npz"))
            print(f"gen {gen:4d}  mean {returns.mean():8.2f}  max {returns.max():8.2f}  "
                  f"best-ever {best:8.2f}  ({time.time() - t0:.0f} s)")
    print(f"saved decoder.npz and decoder_best.npz in {run_dir}")


if __name__ == "__main__":
    main()
