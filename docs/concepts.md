# Concepts

## The loop

Every environment in this project has the same shape:

```
world  --sensors-->  encoder  --drive (mV)-->  BRAIN  --firing rates-->  readout  --policy or decoder-->  actions  --> world
```

* The **world** is either the simulated fly body or your PC.
* An **encoder** turns what the world reports (joint angles, a frame, a chunk of sound) into
  external drive on specific sensory neurons.
* The **brain** is the connectome, stepped as a leaky integrate-and-fire network.
* The **readout** is a population whose firing rates the agent sees.
* A **policy** (PPO) or a **decoder** (a linear table, trained by evolution or imitation)
  turns those rates into actions.

Training changes the policy or decoder, and optionally some synapses. The neurons, their
wiring, and the encoders are fixed.

## The connectome

`neurofly_training.data.connectome` loads the MaleCNS v1.0 flat connectome: 165,122 traced neurons and
25.6 million connections across the brain and the nerve cord. Each neuron carries annotation
columns used to pick populations: superclass, class, subclass, type, side, neuromere, the
nerve it enters or leaves, and, for columnar visual neurons, the hex coordinates of its eye
column. Synapse counts are signed by the presynaptic neuron's transmitter (acetylcholine
excites; GABA, glutamate and histamine inhibit; monoamines are treated as excitatory).

### Subsets

The full brain is slow to step, and most tasks only need part of it. `--subset` chooses:

| Subset | Neurons | What it is | Default for |
|---|---:|---|---|
| `vnc` | 24k | the nerve cord plus the descending, ascending and leg sensory neurons | body tasks |
| `central` | 49k | everything above the neck except the optic lobes, plus the descending neurons | PC tasks |
| `visual` | 77k | `central` plus the optic-lobe columnar neurons that carry eye coordinates, and the photoreceptors | PC tasks that need the real retina |
| `brain` | 142k | everything but the nerve cord | |
| `no-optic` | 72k | everything but the optic lobes | |
| `full` | 165k | all of it | |

`neurofly inspect --subset <name>` prints the superclass counts and the
populations the interfaces found.

### Populations

`neurofly_training.data.populations` names the neuron groups the interfaces talk to:

* **leg motor neurons**, per leg (T1/T2/T3, left/right)
* **leg sensory neurons**, per leg, split into proprioceptive and tactile
* **descending neurons**: brain to nerve cord, the brain's motor output (1,314)
* **ascending neurons**: nerve cord to brain
* **head mechanosensory**: wind/gravity and haltere afferents
* **auditory neurons**: the Johnston's organ neurons of the antenna (114)
* **retina columns**: columnar optic-lobe cells (L1, L2, L3, Mi1, Tm1, Tm2) with their eye
  and hex column; about 875 columns per eye
* **visual projection neurons**: the cells that leave the optic lobe (9,201)
* **dopamine neurons**, and the PPL1 cluster among them (the punishment input)
* **head motor neurons** (`cbmotor`, 107)

## The brain

`neurofly_core.brain.lif` steps every neuron as a leaky integrate-and-fire unit with the parameters of
Shiu et al., Nature 2024: membrane time constant 20 ms, synaptic time constant 5 ms, rest and
reset at -52 mV, threshold at -45 mV, refractory period 2.2 ms, and 0.275 mV of drive per
synapse per spike. Drive is expressed in mV; a neuron at rest needs a sustained 7 mV to fire.

The step is `--dt` milliseconds (default 0.5; the paper used 0.1). Each neuron also keeps an
exponential trace of its firing rate (time constant 50 ms) in Hz; readouts see these rates
divided by 100.

Two backends compute the synaptic input; both give identical spikes. `event` (the default on
CPU) propagates only the neurons that spiked through a compiled loop; `torch` does a sparse
matrix product and is the backend for CUDA. See [Performance](performance.md).

`--brain-gain` multiplies every weight. It is the one global calibration knob: raise it if
the brain stays silent, lower it if activity runs away.

## Encoders: world to drive

### Body: proprioception (`neurofly_training.body.encoder`)

Each leg's joint angles and velocities are projected, through a fixed random sparse map, onto
that leg's proprioceptive neurons; its ground-contact sensor drives that leg's tactile
neurons; the gyro and accelerometer drive the head's wind/gravity and haltere neurons. Drive
is rectified and scaled by `--encoder-gain` (default 12 mV).

### PC: the retina (`neurofly_core.encode.vision`)

The frame is reduced to a 24 x 32 luminance grid and split down the middle: the left half is
the left eye's view, the right half the right eye's. In `hex` mode each columnar neuron
samples the pixel where its hex column sits on that half; ON-pathway cells (L1, L3, Mi1)
are driven by brightness and OFF-pathway cells (L2, Tm1, Tm2) by darkness. This needs the
optic lobe, so the `visual` or `brain` subset. In `projection` mode (the `central` subset)
the grid is projected through a random sparse map onto the visual projection neurons, with
the values contrast-normalised across neurons so the frame's overall brightness does not
matter. `--retina-mode auto` picks by what the subset contains; `--retina-gain` is the drive
at full brightness (15 mV); `--retina-temporal 1` makes cells respond to change between
frames rather than to brightness.

### PC: audition (`neurofly_core.encode.audition`)

The sound since the last step is turned into 16 log-spaced band levels between 50 Hz and
half the sample rate (the shape of the spectrum, scaled by loudness) and projected through
a random sparse map onto the auditory neurons. Tonotopy is not annotated, so which neuron
hears which band is engineered. `--audio-gain` is the drive at full loudness.

### PC: detected objects (`neurofly_core.encode.detection`)

A detector outside the brain (see [The PC](pc.md)) turns the frame into boxes with class
names. Each box is painted onto a coarse grid for its class, and every cell of every class
drives its own set of central-brain interneurons, a labelled line: "an enemy, upper left"
is a particular group of neurons firing. Nothing in the connectome says which neurons
those should be, so the assignment is engineered, like the audition map. The classes
travel in the artifact; the detector does not.

### PC: olfaction (`neurofly_core.encode.olfaction`)

The connectome has about 2,600 olfactory receptor neurons in 53 types, one type per
glomerulus of the antennal lobe, and the mushroom body behind them is the fly's learning
circuit. An odour channel (`--odours health,danger`) drives the receptor neurons of one
glomerulus at a level in [0, 1]. Which glomerulus is engineered; what the antennal lobe
and mushroom body do with it is the connectome's. With `--plasticity` and
`--dopamine-punish`, punishment arriving while an odour is present changes the synapses
that odour's activity reached, which is the fly's own form of aversive learning.

## Readouts and outputs

`--readout` names the population the agent sees, joined by `+`: `motor` (leg motor
neurons), `descending`, `ascending`, `cbmotor`, `visual`. Body tasks default to
`motor+descending`, PC tasks to `descending`.

Three things turn readout rates into actions:

* a **PPO policy**: a two-layer network trained by reinforcement learning (`train`)
* a **linear decoder**: for the body, a structured table where each leg's actuators read
  that leg's motor neurons (`neurofly_training.body.decoder`); for the PC, a table from rates to one
  action vector of keys, buttons, mouse and scroll (`neurofly_core.decode.linear`). Trained
  by evolution strategies (`es`) or fitted to a recording of you (`imitate`).
* the **plasticity rule** (`--plasticity`): reward acts as dopamine on the synapses onto the
  readout, through a three-factor Hebbian rule with eligibility traces
  (`neurofly_core.brain.plasticity`). On the PC, `--dopamine-punish` additionally stimulates the PPL1
  dopamine neurons while reward is negative.

## What is real and what is engineered

Real: the neurons, their synapse counts, their transmitter signs, which leg each motor and
sensory neuron serves, which eye and column each optic-lobe neuron belongs to, and which
head neurons are auditory.

Engineered by you: the LIF parameters (uniform across neurons), every encoder (the maps from
sensors, pixels, sound, detected objects and odours onto neurons), the object detector
itself when you use one (a separate network with no biological counterpart), what counts
as an odour, and the map from readout rates to actions. Those
maps are the parts you train. A fly that walks or uses the PC is "a policy that learned to
do it through the connectome's dynamics", not "the connectome knows how".
