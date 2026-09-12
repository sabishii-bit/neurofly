# neurofly

A fruit fly connectome run as a spiking network and wired to a world. Build the brain once
in Python, then train it and run it from Python, Node, Rust, Go, or anything that can start
a process: the controller travels as a plain artifact, and the runtime speaks a small
protocol that covers both using the brain and training it.

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
  detector (no training, no labelling) into a detection encoder on central-brain neurons;
  `detect-label` and `detect-train` distil it into a fast detector exported to ONNX, with
  YOLO as an opt-in backend.
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

## Data and citations

* MaleCNS v1.0: Janelia FlyEM + Google Research, *Cell* 2026. Files from
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (public).
* LIF model: Shiu et al., "A leaky integrate-and-fire computational model based on the
  connectome of the entire adult Drosophila brain", Nature 2024.
* Body: Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
  Nature 2025 (the `flybody` package).
