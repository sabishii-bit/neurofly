# flybrain-body

The fruit fly's brain and body in one loop, on your own machine:

* **Brain**: the MaleCNS v1.0 connectome (Janelia + Google, September 2026; 165,122 traced
  neurons, 25.6 M edges, brain **and** nerve cord) run as a leaky integrate-and-fire spiking
  network with the parameters of Shiu et al., Nature 2024. This is the same setup the
  "fly brain plays Doom / Minecraft / Beat Saber" projects use.
* **Body**: [flybody](https://github.com/TuragaLab/flybody), the MuJoCo fruit fly from
  Google DeepMind and Janelia, instead of a game.
* **Interface**: each of the six legs is wired to its own real neurons. Joint angles and
  ground contact drive that leg's proprioceptive and tactile sensory neurons in the nerve
  cord; that leg's motor neurons (plus descending neurons) are read out to its actuators.
* **Training tools**: PPO (Stable-Baselines3) on top of the brain readout, evolution
  strategies over the neuron-to-actuator map, and a dopamine-gated plasticity rule that turns
  reward into synaptic change. Plus a body-only baseline so you can tell what the brain adds.

Everything runs on Windows on CPU. A GPU makes the full brain practical.

## Layout

```
flybrain_body/
  data/connectome.py     load MaleCNS -> signed sparse matrix; subsets; synthetic test brain
  data/download.py       fetch the 3 public files (about 570 MB)
  brain/lif.py           LIF network on a torch sparse matrix
  brain/plasticity.py    dopamine-gated Hebbian rule on a synapse subset
  body/tasks.py          flybody tasks: 'forward' (free walking), 'ball' (tethered)
  body/gym_wrapper.py    dm_control -> Gymnasium
  interface/populations.py  leg motor / sensory / descending neuron groups
  interface/encoder.py   body observation -> sensory neuron drive
  interface/decoder.py   readout rates -> 59 actuators (structured linear map)
  envs.py                BrainInLoopEnv and make_env()
scripts/
  download_data.py  inspect_connectome.py  bench_brain.py  train.py  train_es.py  watch.py
tests/
```

## Setup

The project shares the virtual environment of the flybody checkout next door
(`..\flybody\.venv`, Python 3.10.11 from pyenv). It is already installed there.

```powershell
cd W:\Repositories\flybrain-body
..\flybody\.venv\Scripts\Activate.ps1
python scripts/download_data.py        # once; files already present are skipped
python -m pytest -q                    # about 3 minutes; real-data test included when data exists
```

On a fresh machine: `pip install -e .[flybody,dev]` (pulls flybody from GitHub) then the same.

## Look at the brain

```powershell
python scripts/inspect_connectome.py --subset vnc
python scripts/bench_brain.py --subset vnc --dt 0.5
python scripts/bench_brain.py --subset full --dt 0.5 --device cuda   # needs a CUDA torch build
```

Subsets: `full` (165k neurons), `no-optic` (drops the ~95k optic lobe neurons), `vnc`
(nerve cord plus descending, ascending and leg sensory neurons; about 23k neurons). `vnc`
is the default for training because it is where walking lives and it steps in a few ms.

## Train

Body-only baseline, an MLP on proprioception:

```powershell
python scripts/train.py --task forward --brain none --timesteps 2000000 --n-envs 8
```

Brain in the loop. The policy sees firing rates of leg motor and descending neurons, which
in turn are driven by the body's sensors through the real wiring:

```powershell
python scripts/train.py --task forward --brain malecns --subset vnc --n-envs 4 --timesteps 500000
python scripts/train.py --task forward --brain malecns --plasticity     # reward = dopamine
python scripts/train.py --task ball --brain malecns --include-proprio   # rates + raw sensors
```

Evolution strategies over the linear neuron-to-actuator decoder (no policy network, the
"fixed interface" of the game demos, but learned):

```powershell
python scripts/train_es.py --task forward --brain malecns --subset vnc --workers 8 --generations 200
```

Runs land in `runs/<name>/` with `config.json`, model or decoder weights, and TensorBoard logs
for PPO (`tensorboard --logdir runs`). The number to watch is `rollout/ep_rew_mean`; per-step
reward is in [0, 1], a 2-second episode is 1000 steps.

## Watch

```powershell
python scripts/watch.py runs/<name> --episodes 2 --video videos/<name>.mp4
python scripts/watch.py --task forward --policy random --video videos/random.mp4
python scripts/watch.py --task forward --brain malecns --policy zero      # counts brain spikes
```

## Speed

| setting                      | control steps / s (one process, CPU) |
|------------------------------|--------------------------------------|
| body only                    | about 200                            |
| body + `vnc` brain, dt 0.5   | tens                                 |
| body + `full` brain, dt 0.5  | a few (use `--device cuda`)          |

The body's control step is 2 ms; the brain takes `2 / dt` sub-steps per control step. Shiu
et al. used dt = 0.1 ms; 0.5 ms is fine for closed loop and 5x faster. `bench_brain.py`
prints the real numbers for your machine.

## What is real and what is engineered

Real: the neurons, their synapse counts, their transmitter signs, and which leg each motor
and sensory neuron belongs to. Engineered by you: the LIF parameters (uniform across
neurons), how sensor values become sensory drive (`encoder.py`, a fixed random sparse
projection), and how motor neuron rates become joint targets (the decoder, or the PPO
policy). Those two maps are exactly the parts the Doom and Minecraft demos hand-pick, and
here they are the parts you train. Do not read a walking fly as "the connectome knows how
to walk"; read it as "a policy learned to walk through the connectome's dynamics".

## Where to go next

1. **Calibrate the brain.** `bench_brain.py` shows how many neurons are active; tune
   `--brain-gain` and `--encoder-gain` until leg sensory input produces sparse, not
   runaway, activity in the nerve cord.
2. **Use the real gait.** flybody ships a walking-imitation task and a dataset of real fly
   walking (`flybody.download_data.figshare_download('walking-imitation-dataset')`). Reward
   matching those trajectories to get natural gaits instead of whatever PPO invents.
3. **Make the decoder anatomical.** Motor neuron types in the annotation table say what
   muscle they drive (`Ti flexor MN`, `Fe reductor MN`, `Ta depressor MN`, ...). Map them to
   the matching flybody joints instead of a random initialisation.
4. **Query neuPrint live.** `neuprint-python` is installed; dataset `male-cns:v1.0` at
   neuprint.janelia.org gives synapse locations, ROIs and more (needs a free token).

## Data and citations

* MaleCNS v1.0: Janelia FlyEM + Google Research, *Cell* 2026. Files from
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (public).
* LIF model: Shiu et al., "A leaky integrate-and-fire computational model based on the
  connectome of the entire adult Drosophila brain", Nature 2024.
* Body: Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
  Nature 2025.
