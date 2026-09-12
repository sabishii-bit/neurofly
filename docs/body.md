# The body

The simulated fly is the MuJoCo model from the `flybody` package, wrapped as a Gymnasium
environment (`neurofly_training.body.gym_wrapper`). The action is 59 actuators in [-1, 1] (head, abdomen,
eight joints and one adhesion claw per leg; the order is in `neurofly_training.body.actuators`). The
observation is the flattened dm_control observable dict: joint positions and velocities,
touch sensors, gyro, accelerometer and so on; the wrapper remembers where each lives so the
sensory encoder can find them. A control step is 2 ms.

## Tasks

| Task | What it is | Reward | Ends when |
|---|---|---|---|
| `forward` | free fly on a floor | speed toward +x, times upright, times at standing height, each in [0, 1] | it falls over or explodes, or after 2 s |
| `ball` | tethered on a ball | the `flybody` walk-on-ball task | after its time limit |

Zero action holds the default standing pose, so learning starts from a stable posture. A
2-second episode is 1000 steps; per-step reward is in [0, 1].

## Run it

Without the brain, the policy sees the raw observation and drives the actuators directly:

```powershell
neurofly train --task forward --brain none --timesteps 2000000 --n-envs 8
```

With the brain, the body observation is encoded into drive on the nerve cord's sensory
neurons, the brain runs 2 ms (four sub-steps at `--dt 0.5`), and the policy sees the firing
rates of the leg motor and descending neurons. `--include-proprio` also gives it the raw
observation, so you can tell what the brain adds:

```powershell
neurofly train --task forward --brain malecns --n-envs 4 --timesteps 500000
neurofly train --task ball --brain malecns --include-proprio
neurofly train --task forward --brain malecns --plasticity    # reward = dopamine on synapses onto leg motor neurons
```

Evolution strategies over the structured neuron-to-actuator table, no policy network:

```powershell
neurofly es --task forward --brain malecns --workers 8 --generations 200
```

Watch any run, or a fixed policy, and write a video:

```powershell
neurofly watch runs/<name> --episodes 2 --video videos/<name>.mp4
neurofly watch --task forward --policy random --video videos/random.mp4
neurofly watch --task forward --brain malecns --policy zero      # stands still, counts spikes
```

`watch` prints per episode the return, the number of steps, the forward and sideways
distance the thorax moved, and, with a brain, the spike count. `--camera` picks the MuJoCo
camera and `--every` how many control steps per video frame (default 10, so 50 fps video).

## Calibrating the brain

The two knobs are `--encoder-gain` (how strongly sensors drive sensory neurons; default
12 mV) and `--brain-gain` (a multiplier on every synapse). The aim is sparse, not runaway,
activity in the nerve cord when the fly stands. `fly.py bench --subset vnc` prints how many
neurons are active and at what rate under a constant drive; `watch --policy zero` prints the
spike count of a standing fly.

## Task options from Python

The task factories accept keyword arguments the command line does not expose. From Python:

```python
from neurofly_training.envs import make_env

env = make_env("forward", brain="malecns", target_speed=2.0, claw_friction=0.8, time_limit=3.0)
```

`forward` takes `target_speed` (cm/s, default 1.5), `claw_friction` (default 1.0) and
`time_limit` (s, default 2.0). See [Extending](extending.md) for adding a task.

## Next steps that pay off

1. **Use the real gait.** `flybody` ships a walking-imitation task and a dataset of real fly
   walking. Reward matching those trajectories to get natural gaits instead of whatever PPO
   invents.
2. **Make the decoder anatomical.** Motor neuron types in the annotation table say what
   muscle they drive (tibia flexor, femur reductor, tarsus depressor, ...). Map them to the
   matching joints instead of a random initialisation.
