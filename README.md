# neurofly

A fruit fly connectome run as a spiking network and wired to a world. Train it in Python,
export the result as a plain artifact, run that artifact from Python, Node, Rust, or
anything that can start a process.

* **Brain**: the MaleCNS v1.0 connectome (165,122 traced neurons, 25.6 M synapses, brain and
  nerve cord) as a leaky integrate-and-fire network with the parameters of Shiu et al.,
  Nature 2024. Event-driven propagation steps the central brain in about a millisecond on CPU.
* **Worlds**: the PC (screen and sound in through the retina and the auditory neurons;
  keyboard and mouse out of the descending neurons), and the `flybody` MuJoCo fruit fly
  (joints and contacts in, 59 actuators out).
* **Training**: PPO, evolution strategies over a neuron-to-control table, imitation of your
  own recorded use of the PC, surrogate-gradient training through the spiking dynamics, and
  dopamine-gated plasticity. A `Task` you write supplies reward and episode structure;
  `neurofly eval` scores any artifact against recordings.
* **Experiments**: stimulate, silence and probe any neurons by connectome type while the
  brain runs, from the command line or the API; replay videos with the spike raster beside
  the frames; a gain calibration sweep; a small `toy` brain with a designed path for tests.
* **Learning from footage**: an inverse dynamics model labels video that has no input log;
  corrections while the fly plays become new labels (DAgger); template matching and OCR
  helpers turn what is on screen into reward.
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
examples/        a Task to copy, and a Node program that consumes an artifact.
docs/            how to use and extend everything.
tests/           pytest suite: tests/core and tests/training.
data/            inputs you download or record (git-ignored): malecns/, recordings/
runs/            raw training outputs (git-ignored)
artifacts/       exported controllers, what you ship (git-ignored by default)
```

## Quick start

```powershell
pip install -e core[pc] -e training[dev]        # plus flybody for body tasks: training[flybody]
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

Full documentation: [docs/](docs/README.md). The runtime alone: [core/](core/README.md).

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
