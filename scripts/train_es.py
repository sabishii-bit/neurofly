"""Evolution strategies over the linear neuron-to-actuator decoder.

No gradients through the brain: a population of decoder weight vectors is
evaluated by rolling out the brain-in-the-loop env, and the mean is moved
toward the better ones (OpenAI-ES with antithetic sampling and rank
shaping). Each worker process owns one brain and one body.

    python scripts/train_es.py --task forward --brain malecns --subset vnc --workers 8 --generations 200
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from flybrain_body.body.tasks import TASKS
from flybrain_body.data.connectome import SUBSETS
from flybrain_body.envs import ENV_ARGS, BrainInLoopEnv, load_connectome, make_body_env
from flybrain_body.interface.decoder import LinearDecoder

_ENV = None
_DEC = None


def _init_worker(cfg: dict):
    global _ENV, _DEC
    body = make_body_env(cfg["task"], seed=cfg["seed"])
    cx = load_connectome(cfg["brain"], subset=cfg["subset"], synthetic_n=cfg["synthetic_n"])
    env = BrainInLoopEnv(body, cx, readout=cfg["readout"], dt=cfg["dt"],
                         brain_gain=cfg["brain_gain"], encoder_gain=cfg["encoder_gain"],
                         plasticity=cfg["plasticity"], seed=cfg["seed"])
    _ENV, _DEC = env, LinearDecoder(env.pops, env.readout_idx, seed=cfg["seed"])


def build_decoder(cfg: dict) -> LinearDecoder:
    """Same construction as the workers, in the main process (for shapes and saving)."""
    _init_worker(cfg)
    return _DEC


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="forward", choices=sorted(TASKS))
    p.add_argument("--brain", default="malecns", choices=["malecns", "synthetic"])
    p.add_argument("--subset", default="vnc", choices=sorted(SUBSETS))
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--readout", default="motor+descending")
    p.add_argument("--plasticity", action="store_true")
    p.add_argument("--brain-gain", type=float, default=1.0)
    p.add_argument("--encoder-gain", type=float, default=12.0)
    p.add_argument("--synthetic-n", type=int, default=3000)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--population", type=int, default=32, help="even; half are antithetic")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--episode-steps", type=int, default=500, help="500 steps = 1 s of sim")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    args.include_proprio = False
    cfg = {k: getattr(args, k) for k in ENV_ARGS}
    cfg["seed"] = args.seed

    run_dir = os.path.join("runs", args.run_name or f"es_{args.task}_{args.brain}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(vars(args), algo="es"), f, indent=2)

    dec = build_decoder(cfg)
    theta = dec.get_params()
    print(f"run dir: {run_dir}; decoder has {len(theta)} parameters; {args.workers} workers")
    rng = np.random.default_rng(args.seed)
    half = args.population // 2
    best = -np.inf
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
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
                np.savez(os.path.join(run_dir, "decoder_best.npz"), theta=cands[int(returns.argmax())])
            print(f"gen {gen:4d}  mean {returns.mean():8.2f}  max {returns.max():8.2f}  "
                  f"best-ever {best:8.2f}  ({time.time() - t0:.0f} s)")
    print(f"saved decoder.npz and decoder_best.npz in {run_dir}")


if __name__ == "__main__":
    main()
