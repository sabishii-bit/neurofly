# Performance

## Where the time goes

A brain step is one pass over the synapses plus a handful of element-wise operations over
the neurons. The synapses dominate: the central brain has 9.5 million of them, the whole
brain 21 million.

Two backends compute the synaptic input:

* **`event`** (the default on CPU, needs numba): only the neurons that spiked propagate,
  through a compiled loop over their outgoing synapses, which are stored by presynaptic
  neuron. One to three percent of neurons spike per step, so this touches a small fraction
  of the synapses.
* **`torch`**: one sparse matrix-vector product per step, weights stored by postsynaptic
  neuron. It streams every synapse every step (76 MB for the central brain), which makes it
  memory-bound. It is the backend for CUDA.

Both produce identical spikes. Measured on one CPU process, dt = 0.5 ms:

| subset | neurons | synapses | `torch` | `event` |
|---|---:|---:|---:|---:|
| `vnc` | 24k | 4.0M | 4.7 ms | 0.34 ms |
| `central` | 49k | 9.5M | 11 ms | 1.0 ms |
| `visual` | 77k | 10M | 13 ms | 1.2 ms |
| `brain` | 142k | 21M | 26 ms | 1.4 ms |

`neurofly bench --subset <name> --backend event|torch` prints the numbers for
your machine.

## What a task step costs

* **Body**: a 2 ms control step is 4 brain sub-steps at dt 0.5. With the `vnc` subset that is
  about 1.5 ms of brain, and the MuJoCo body itself (about 5 ms) becomes the larger cost.
  Expect around 150 control steps per second per process with the brain, 200 without.
* **PC**: 10 ms of brain time is 20 sub-steps: about 20 ms on `central` or `visual`, plus a
  few ms for capture, encoding and features. A live loop keeps up with 10 steps per second.
  Raise `--brain-ms` or `--fps` from there as the machine allows.
* **Plasticity** adds the eligibility-trace update over the plastic synapses each sub-step:
  about 4 ms for the 430k synapses onto the descending neurons.
* **Training** parallelises across processes: `--n-envs` for PPO, `--workers` for ES. Each
  process holds its own copy of the brain (about 100 MB for `central`).

## GPU

A CUDA build of torch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`
or as the torch site says for your driver) makes `--device cuda` available; the brain then
uses the `torch` backend on the GPU, where the full brain steps in under a millisecond. The
event backend is CPU-only.

## Why not another language

The per-step Python overhead is well under a millisecond; the rest was always inside
compiled code (the sparse product in torch, the propagation loop in numba). Moving the
whole simulation to another language would not change that. What would gain more:

* the GPU, for the full brain or many brains at once;
* a hand-written propagation kernel in C++ or Rust with SIMD, perhaps 2x over numba;
* larger `--dt` (0.5 is already 5x the paper's 0.1) or smaller subsets.

## Tips

* Use `synthetic` as the brain while developing a task or a source: it builds in
  milliseconds and exercises every code path.
* `--max-frames` on `imitate` and `--max-steps` on everything else bound a first run.
* `watch --policy zero` on a body task, or `play --dry-run --steps 50` on the PC, is the
  fastest check that the brain is neither silent nor saturated; adjust `--brain-gain`,
  `--encoder-gain`, `--retina-gain`, `--audio-gain` from there.
