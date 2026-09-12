# neurofly

A kit for building things on a fruit fly's brain. The MaleCNS connectome runs as a spiking
network; the kit wires it to senses and outputs by the connectome's own annotations, trains
it, saves it as a portable artifact and serves it, so that a project that wants a fly brain
in a game, a robot or an experiment starts at the interesting part instead of at the
loading code.

## What it is for

Getting a connectome to do anything takes a lot of boilerplate before the first
interesting result: reading the data files, turning annotations into populations, picking a
simulation that steps 50,000 neurons in a millisecond, mapping pixels onto the right
columnar neurons and sound onto the auditory ones, getting keyboard and mouse events out
without locking yourself out of your own PC, keeping the result in a form another program
can use. neurofly-kit is that boilerplate, done once, behind three interfaces:

* **The `neurofly` command** for the Python side: build a brain with the senses you want
  (`--detect`, `--odours`, `--tastes`, `--thermo`, `--touch`, `--audio`), record yourself,
  train, watch, export.
* **The artifact**, a directory of a manifest and flat arrays with no Python objects, which
  is the whole trained brain and its encoders. It is the thing you ship.
* **The runtime and its bindings**: `neurofly-core` serves an artifact over stdio, WebSocket
  or gRPC, locally or from a container on a server (a brain per client, a token, recordings
  written for the trainers), and the TypeScript (subprocess and browser), Rust and Go
  bindings wrap that, so the project that uses the fly can be written in whatever language
  it is written in and run wherever it runs.
  [Using neurofly from your own project](docs/from-your-project.md) is the step by step and
  [Hosting the brain](docs/deploy.md) the deployment.

It is a template as much as a library: the pieces are meant to be replaced. Write a `Task`
for your game, swap the detector, add a sense, point the readout at other neurons.

* **Brain**: the MaleCNS v1.0 connectome (165,122 traced neurons, 25.6 M synapses, brain and
  nerve cord) as a leaky integrate-and-fire network with the parameters of Shiu et al.,
  Nature 2024. Event-driven propagation steps the central brain in about a millisecond on CPU.
* **Worlds**: the PC (screen and sound in through the retina and the auditory neurons;
  keyboard and mouse out of the descending neurons), and the `flybody` MuJoCo fruit fly
  (joints and contacts in, 59 actuators out).
* **What training means here**: the neurons and their wiring are fixed. What is trained is
  the policy (the map from readout firing rates to controls), optionally the encoder tables,
  and optionally some synapses through dopamine-gated plasticity. A `Task` you write
  supplies reward and episode structure; `neurofly eval` scores any artifact against
  recordings.
* **Training from any language**: `neurofly build` makes a base artifact (brain and encoders,
  no policy). A trainer in Node, Rust, Go or anything else serves it, reads the feature
  vector with `observe` (or `body_step` for the body), runs whatever optimiser it likes,
  installs the result with `set_policy` and writes a complete artifact with `save`. Passing
  a reward drives plasticity inside the brain. `examples/node_train_es.js` does all of this
  in JavaScript.
* **Bundled Python trainers**: PPO, evolution strategies, imitation of your own recorded use
  of the PC, and surrogate-gradient training through the spiking dynamics. The last one is
  the only method that cannot run from another language, because it needs gradients through
  the brain rather than features out of it.
* **Experiments**: stimulate, silence and probe any neurons by connectome type while the
  brain runs, from the command line or the API; replay videos with the spike raster beside
  the frames; a gain calibration sweep; a small `toy` brain with a designed path for tests.
* **Objects on screen**: `--detect "owl2:enemy,health pack"` runs an open-vocabulary
  detector (no training, no labelling) into a detection encoder on central-brain neurons.
  Detectors are swappable backends behind one spec (OWLv2, Grounding DINO, YOLO-World,
  YOLO11, RT-DETRv2, D-FINE, SSDLite, ONNX): `neurofly detect-list` shows them with their
  licences, `detect-install` fetches one, and `detect-label` plus `detect-train` distil the
  open-vocabulary detector into a fast fine-tuned one without hand labelling.
* **Smell, taste, temperature, touch**: `--odours`, `--tastes`, `--thermo` and `--touch`
  drive the olfactory, gustatory, thermo/hygrosensory and bristle/grooming/leg-tactile
  neurons from your Task or from the detector's classes; the body's own touch (leg
  contacts, wind, gravity) is wired by annotation. `pulses` drive any named population
  for one step: the giant fibre for a startle, the clock, the dopamine neurons.
* **Learning where the fly learns**: `--dopamine-punish` and `--dopamine-reward` drive
  the PPL1 and PAM dopamine neurons, and `--plasticity-target mbon` puts the three-factor
  rule on the Kenyon-cell-to-MBON synapses, so punishment during a smell weakens it and
  reward strengthens it. The compass and the mushroom body output are readouts.
* **Learning from footage**: an inverse dynamics model labels video that has no input log;
  corrections while the fly plays become new labels (DAgger); template matching and OCR
  helpers turn what is on screen into reward.
* **Seeing the brain**: every artifact carries each neuron's soma position from the
  connectome; `--activity-out` records every spike per step and `--activity-ws` streams
  them live, and `examples/brain_viewer.html` draws the brain as a point cloud that lights
  up as it fires, with the strongest synapses flashing between neurons.
* **Safety**: a focus guard stops the fly when the keyboard focus leaves its window, and a
  watchdog releases every key if the loop stalls.
* **The body elsewhere**: body runs export as artifacts too (observation in, 59 actuators
  out, served over JSON or gRPC); `neurofly export-body` writes the fly as a glTF and, with
  `--mjcf`, the complete MuJoCo model for any MuJoCo build; `watch --poses` or `--pose-ws`
  feed `examples/three_viewer.html`, recorded or live.
* **Deployment**: `neurofly export` writes an artifact (a manifest plus flat binary arrays,
  no Python objects); `neurofly-core` loads it and either drives the PC itself (keyboard,
  mouse, a virtual gamepad) or serves a JSON-lines or gRPC protocol that the Node, Rust and
  Go bindings speak.

## Layout

```
core/            neurofly-core: the runtime. Brain, encoders, decoders, artifact loading,
                 PC input/output, the server, the `neurofly-core` command. No training code.
training/        neurofly-training: connectome loading, body and PC environments, PPO / ES /
                 imitation, recording, export, the `neurofly` command. Depends on core.
artifact/        SPEC.md: the artifact format and the server protocol, for other languages.
bindings/        node/, rust/ and go/ packages that spawn the runtime; python/ points at core.
examples/        a Task to copy, Node programs that consume and that train an artifact, a
                 Three.js viewer for the body.
docs/            how to use and extend everything.
tests/           pytest suite: tests/core and tests/training.
data/            inputs you download or record (git-ignored): malecns/, recordings/
runs/            raw training outputs (git-ignored)
artifacts/       exported controllers, what you ship (git-ignored by default)
```

## Quick start

```powershell
pip install -e core[pc] -e training[dev]        # see docs/installation.md for extras, GPU, bindings
neurofly download                                # the connectome files (about 570 MB)
python -m pytest -q                              # about 25 s

neurofly play  --window "My App" --keys w,a,s,d --mouse --brain malecns --dry-run   # prints what it would press
neurofly record  --window "My App" --keys w,a,s,d --mouse --audio loopback --out data/recordings/run1
neurofly imitate data/recordings/run1 --run-name imitate_myapp
neurofly export  runs/imitate_myapp artifacts/myapp
neurofly-core run artifacts/myapp --window "My App"       # the brain uses the PC
neurofly-core serve artifacts/myapp                       # ... or serves any language
```

```js
const { NeuroFly } = require("neurofly");                 // bindings/node
const fly = new NeuroFly("artifacts/myapp"); await fly.start();
const r = await fly.step({ frame, width: 320, height: 240 });   // r.keys, r.dx, r.dy, ...
```

```js
// training from Node: features in, your optimiser, policy out (examples/node_train_es.js)
const base = new NeuroFly("artifacts/base"); await base.start();
const { features } = await base.observe({ frame, width: 320, height: 240, reward });
await base.setPolicy({ type: "linear", W, b });
await base.save("artifacts/trained");
```

Full documentation: [docs/](docs/README.md). The runtime alone: [core/](core/README.md).

## Where Python is required

Three things, and only these:

1. **Building the base artifact** from the connectome: loading the data, choosing a subset,
   building the encoder tables from the annotations. One command, once
   (`neurofly build`), and the artifact is the hand-off.
2. **Surrogate-gradient training**, which backpropagates through the spiking dynamics and
   therefore needs the brain in-process rather than behind a protocol.
3. **The body simulation** during training, which uses MuJoCo through `flybody`. A trainer
   in another language runs the exported MuJoCo model (`neurofly export-body --mjcf`) in
   its own MuJoCo and feeds observations to `body_step`.

Everything else, running and training included, works through the protocol. PPO is bundled
because the library was handy, not because the brain requires it; the feature vector goes
over a JSON or gRPC round trip (about 13 ms here), which suits evolution strategies and
imitation better than optimisers that need millions of steps.

## What is real and what is engineered

Real: the neurons, their synapse counts, their transmitter signs, which leg each motor and
sensory neuron serves, which eye and column each optic-lobe neuron belongs to, which head
neurons are auditory. Engineered: the LIF parameters (uniform across neurons), the encoders
(sensors, pixels and sound onto neurons), and the map from readout rates to actions. Those
maps are the parts you train. Read a fly that uses the PC as "a policy learned to do it
through the connectome's dynamics", not as "the connectome knows how".

## Licence

AGPL-3.0. The project is built to plug in whichever detector or model gives the best
result, and the strongest ones (Ultralytics YOLO, YOLO-World) come under the AGPL; a
permissive licence here would only mislead. Every backend's own licence is listed by
`neurofly detect-list`, and the artifacts, the protocol and the bindings work with any of
them.

## Data and citations

* MaleCNS v1.0: Janelia FlyEM + Google Research, *Cell* 2026. Files from
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (public).
* LIF model: Shiu et al., "A leaky integrate-and-fire computational model based on the
  connectome of the entire adult Drosophila brain", Nature 2024.
* Body: Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
  Nature 2025 (the `flybody` package).
