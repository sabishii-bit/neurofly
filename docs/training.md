# Training

Four ways to make the fly do something, all of which leave the neurons and the encoders
alone and change the map from readout to actions (and, with plasticity, some synapses).

| Method | Command | What it trains | Needs |
|---|---|---|---|
| PPO | `train` | a two-layer policy network on the readout | reward from the task, many episodes |
| evolution strategies | `es` | the linear neuron-to-output table | reward from the task; no gradients, parallel workers |
| imitation | `imitate` | the same linear table, fitted to a recording of you | a recording; no reward needed |
| surrogate gradients | `surrogate` | the retina's input map and a linear head, through the spiking dynamics | a recording |
| plasticity | `--plasticity` on any of the above | synapses onto the readout, dopamine-gated | reward |

## PPO (`train`)

Stable-Baselines3 PPO with observation and reward normalisation. Defaults: rollout of 1024
steps per environment, batch 256, 10 epochs, learning rate 3e-4, gamma 0.99, a 256-256
network for policy and value. `--n-envs` environments run in parallel processes (8 without
a brain, 4 with; 1 on the live screen). Checkpoints every `--checkpoint-every` steps.

```powershell
neurofly train --task forward --brain malecns --timesteps 500000 --run-name walk1
neurofly train --resume runs/walk1 --timesteps 500000      # continue
tensorboard --logdir runs
```

The number to watch is `rollout/ep_rew_mean`. A run directory holds `config.json`,
`model.zip`, `vecnormalize.pkl`, `checkpoints/` and `tb/`.

## Evolution strategies (`es`)

OpenAI-style ES with antithetic sampling and rank shaping over the flat parameter vector of
the decoder. Each of `--workers` processes owns one brain and one body (or video); the live
screen is a single worker. `--population` candidates per generation (even; half are
mirrored), `--sigma` noise, `--lr` step, `--episode-steps` per rollout.

```powershell
neurofly es --task forward --brain malecns --workers 8 --generations 200 --run-name es1
```

The run directory holds `config.json`, `decoder.npz` (the current mean) and
`decoder_best.npz` (the best candidate ever seen); `watch` and `play` prefer the best.

## Imitation (`imitate`)

Plays recordings through the brain, collects the readout on each frame, and fits by gradient
descent: logistic regression for keys and buttons (class-balanced per control), least
squares through tanh for mouse and scroll. `--epochs`, `--lr`, `--l2` control the fit;
`--holdout` keeps the tail of each recording for evaluation; `--max-frames` truncates long
recordings. The run directory holds `config.json`, `decoder.npz` and `features.npz` (the
collected features and labels, so a refit does not need the brain again).

See [The PC](pc.md) for recording and interpreting the report.

## Through the brain: surrogate gradients (`surrogate`)

The three methods above treat the brain as a fixed feature extractor. `surrogate`
backpropagates through the spiking dynamics instead: the spike is a step function forward
and a fast sigmoid backward, so a loss on the readout can move parameters upstream of the
neurons. It trains the retina's input map (the projection matrix, or per-neuron gains in
hex mode) together with a linear head, on recordings, with truncated backpropagation inside
each frame's brain sub-steps. The synapses of the connectome stay fixed.

```powershell
neurofly surrogate data/recordings/run1 --brain malecns --epochs 5 --bptt-window 8 --run-name through_brain
```

The result is an artifact at `runs/<name>/artifact`, scored on the held-out tail of each
recording like `imitate`. It is slower per frame than the other methods (a backward pass
through 20 sub-steps of a 49k-neuron brain) and worth it when `imitate` shows the readout
does not carry what the task needs.

## Learning from footage without an input log (`idm` and `label`)

An inverse dynamics model predicts the action at frame t from the luminance grids of the
frames around it (t-k to t+k). Seeing the future makes that much easier than acting, so a
few minutes of your own labelled play train it well enough to label hours of footage that
has no input log. The labelled footage then feeds `imitate`, `surrogate` or `eval` like
any recording. The model lives outside the brain.

```powershell
neurofly idm   data/recordings/run1 data/recordings/run2 --run-name idm1
neurofly label footage/a.mp4 footage/b.mp4 --idm runs/idm1 --out data/recordings/labelled
neurofly imitate data/recordings/labelled/a data/recordings/labelled/b --run-name from_footage
```

## Calibrating the brain (`calibrate`)

The gains decide whether the brain is silent, sparse or saturated. `calibrate` sweeps one
of them over a grid, runs the model on a few frames (from the screen or a video) at each
value, and prints the fraction of readout neurons that fire, the fraction of all neurons
that fire and the mean readout rate, then picks the value closest to `--target`:

```powershell
neurofly calibrate --brain malecns --video data/recordings/run1/video.mp4 --param brain_gain
```

## A brain for tests and examples

`--brain toy` is a small connectome with a designed path: each eye's retina columns drive
their own half of the visual projection neurons, then of the central interneurons, then of
the descending neurons, on top of random background wiring. Its descending readout tells
left from right, which the random `--brain synthetic` cannot, so examples and smoke tests on
it behave like the real brain does. Both build in milliseconds.

## Plasticity

`--plasticity` makes reward act as dopamine on a subset of existing synapses: those onto the
leg motor neurons for body tasks, onto the readout for PC tasks. Each synapse keeps an
eligibility trace that grows when its presynaptic neuron was recently active and its
postsynaptic neuron spikes; dopamine turns eligibility into weight change. Weights keep
their sign and stay within three times their anatomical value. The learning rate is 1e-3
per unit of reward per brain step; reward is in [0, 1] per step for the body.

On the PC, `--dopamine-punish <mV>` additionally drives the PPL1 dopamine neurons while the
reward is negative, whether or not plasticity is on.

Plasticity changes the brain within an episode and across episodes; the changed weights are
not saved with the run.

## Run directories and reproducibility

Every run writes `config.json` with all environment options (the `ENV_ARGS` list in
`neurofly_training.envs`), the algorithm, and the seed. `watch` and `play` read it to rebuild the same
environment: same subset, readout, encoder gains, layout. Options passed explicitly to
`watch` or `play` override the stored ones.

The connectome and the encoders are built from a fixed seed (0 for the connectome, the env
seed for the encoders' random projections), so every worker and every playback sees the same
neurons in the same order.

## Choosing what the agent sees

* **Readout.** `descending` is the brain's own motor output and the natural choice; add
  `cbmotor` for the head, `visual` for the optic lobe's output, `motor` for the leg motor
  neurons in body tasks. Bigger readouts give the policy more to work with and the decoder
  more parameters.
* **Raw features.** `--include-proprio` (body), `--include-frame` and `--include-audio` (PC)
  bypass the brain for part of the observation. Useful as a baseline for what the brain
  adds, and as a crutch when the brain's response does not carry what the task needs.
* **Brain time.** `--brain-ms` on the PC (default 10) is how long the brain integrates each
  frame; rates have a 50 ms time constant, so more brain time gives steadier readouts at
  the cost of speed.
