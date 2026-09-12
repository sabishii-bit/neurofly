# Runtime and artifacts

Training produces a run directory (Python objects, framework checkpoints). `export` turns
one into an **artifact**: a directory with a `manifest.json` and flat binary arrays, and
nothing else. `neurofly-core` loads artifacts; it has no training code, no data loading, and
no dependency on the training package.

## Export

```powershell
neurofly export runs/imitate_myapp artifacts/myapp
neurofly export runs/pc_malecns_20260911-1200 artifacts/myapp --name myapp
```

The model is rebuilt from the run's `config.json` (same subset, encoders, readout and
layout; the encoders use a fixed seed so they come out identical), the trained policy is
converted to plain arrays, and the whole thing is written and validated:

| Run | Policy in the artifact |
|---|---|
| `imitate`, `es` | a linear table: `action = tanh(W @ features + b)` |
| `train` (PPO) | the policy network as dense layers with its observation normalisation |

Body runs export too (kind `body`); see "The body outside Python" below for what the
runtime does with them.

## What is inside

```
artifacts/myapp/
  manifest.json      sizes, parameters, layout, provenance, and a {file, dtype, shape} entry per array
  brain/             the synapses: indptr, indices, values (mV per spike), by presynaptic neuron
  readout/           which neurons are the features
  retina/ audition/  the encoder tables
  policy/            the weights
  punish/ neurons/   punishment neurons; source ids of every neuron (provenance)
```

Arrays are raw little-endian binaries. A consumer in any language needs a JSON parser and
`fread`. The complete contract, including the dynamics and every formula, is
[artifact/SPEC.md](../artifact/SPEC.md).

The central brain is about 150 MB (9.5 M synapses at 16 bytes each); the synthetic test
brain is under 1 MB.

## Run it

```powershell
neurofly-core info     artifacts/myapp
neurofly-core validate artifacts/myapp
neurofly-core run      artifacts/myapp --window "My App"                # the brain uses the PC
neurofly-core run      artifacts/myapp --region 0,0,800,600 --audio loopback --dry-run
```

`run` captures the screen and sends keys and mouse motion itself, like `neurofly play`
does during training, with none of the training stack loaded. Esc stops it.

From Python:

```python
import neurofly_core as fc
model = fc.load_model("artifacts/myapp")
model.reset()
state, info = model.step(frame, audio)     # ControlState: keys, buttons, dx, dy, scroll
features = model.observe(frame, audio)     # or only the readout, for your own policy
```

## Serve it

```powershell
neurofly-core serve artifacts/myapp                     # JSON lines on stdin / stdout
neurofly-core serve artifacts/myapp --ws 127.0.0.1:8765  # the same over a WebSocket
```

One request per line, one response per line; the first line out announces readiness and
carries the controls, feature and action sizes. A step sends a frame (raw RGB bytes,
base64, or a PNG or JPEG) and optionally the sound since the last step, and gets back the
keys and buttons held, the mouse motion and scroll, the raw action vector, and the spike
count. Errors come back as `{"ok": false, "error": ...}` and the session continues. The
exact messages are in [artifact/SPEC.md](../artifact/SPEC.md).

## From other languages

`bindings/` holds thin clients that spawn `neurofly-core serve` and speak the protocol:

```ts
// bindings/node (TypeScript; npm install && npm run build once)
import { NeuroFly } from "neurofly";
const fly = new NeuroFly("artifacts/myapp");            // or { python: "path/to/python" }
const info = await fly.start();
const r = await fly.step({ frame, width: 320, height: 240, audio, sampleRate: 16000 });
await fly.close();
```

```rust
// bindings/rust
use neurofly::{Client, Step};
let mut fly = Client::spawn("artifacts/myapp")?;
let r = fly.step(&Step::rgb(&frame, 320, 240).audio(&samples, 16000, 1))?;
fly.close()?;
```

```go
// bindings/go
fly, err := neurofly.Spawn("artifacts/myapp")
r, err := fly.Step(neurofly.Step{Frame: rgb, Width: 320, Height: 240, Audio: samples, SampleRate: 16000})
fly.Close()
```

All three need `neurofly-core` installed in a Python on the PATH (or an interpreter you
name), and all three have the training calls too (`observe`, `set_policy`, `save`).
`examples/node_consumer.js` and `bindings/go/example` are complete programs. Anything else that can start a process
and write lines works the same way; the WebSocket flavour serves browsers and languages
without subprocess control.

An artifact built with `--detect` has a detection encoder and expects detected objects on
every step: `detections` in the JSON request (`{"class": id or name, "box": [x0, y0, x1,
y1], "score"}` with the box in fractions of the frame), `StepRequest.detections` over gRPC,
`detections` on the Node, Rust and Go step inputs. `info` lists the classes it expects
(`detection_classes`). The consumer runs the detector itself: `neurofly detect-train`
exports `detector.onnx`, which ONNX Runtime loads in every language the bindings cover, and
`neurofly_training.pc.detect.OnnxDetector` shows the pre- and post-processing to copy. A
step without detections is a step with nothing detected.

An artifact built with `--odours` likewise expects `odours` on the step: a vector in the
order of `info`'s `odour_channels` (or a name-to-value object over JSON), values in [0, 1];
a step without them keeps the last values. What the channels mean is the consumer's
business: health, ammo, "an enemy is in view".

## Training from another language

The trainable part of a controller is the policy: a linear table or a small MLP on top of
the feature vector. Everything a trainer needs is in the protocol, so the training loop can
live in Node, Rust, or anything else, with the runtime as a dependency:

1. Build a **base artifact** in Python once: the brain and its encoders, no policy.

   ```powershell
   neurofly build artifacts/base --brain malecns --keys w,a,s,d --mouse --audio loopback
   ```

2. Serve it and read **features** with `observe` (your program owns the frames, the sound,
   the reward and the world). Pass `reward` on `observe` if you want plasticity and
   punishment to act inside the brain.
3. Train whatever you like on those features: evolution strategies, imitation of your own
   inputs, your own RL. The action vector is defined by the layout; `tanh(W f + b)` is the
   linear policy.
4. Install it with `set_policy` and write a complete artifact with `save`. That artifact
   runs anywhere, including `neurofly-core run` on the PC.

```js
const fly = new NeuroFly("artifacts/base");  await fly.start();
const r = await fly.observe({ frame, width, height, reward });   // r.features
// ... your optimiser ...
await fly.setPolicy({ type: "linear", W, b });
await fly.save("artifacts/trained", "trained");
```

`examples/node_train_es.js` is a complete trainer: evolution strategies over the linear
table, in Node, on a toy world, saving a runnable artifact at the end. The Rust and Go clients have
the same three calls (`observe`, `set_policy_linear` / `SetPolicyLinear`, `save`).

What stays in Python: building the base artifact from the connectome (a one-time step), the
MuJoCo body, and the bundled PPO trainer. A trainer in another language can implement any
algorithm it likes on the features; PPO is not special.

## gRPC

For a low-latency loop, or a language with a protobuf toolchain, the same service speaks
gRPC:

```powershell
neurofly-core serve artifacts/myapp --grpc 127.0.0.1:50051
```

Frames and audio travel as bytes and `Stream` keeps one connection open. The definition is
[core/src/neurofly_core/rpc/neurofly.proto](../core/src/neurofly_core/rpc/neurofly.proto);
`protoc` generates a client for your language. Every operation of the JSON protocol is
there, including the training seam and the experiments below.

## Experiments on a running brain

The one thing a connectome model can do that an ordinary policy cannot is answer "what if
these neurons were silent, or driven". Three operations, available on `neurofly play`,
`neurofly watch`, `neurofly-core run`, `neurofly-core serve` and through the protocol:

```powershell
neurofly-core run artifacts/myapp --window "My App" \
    --stimulate "type_re=^PPL1:20"                 # 20 mV on the PPL1 dopamine cluster, every step
    --silence   "superclass=descending_neuron"     # these never spike
    --probe     "name=readout" --probe-out probe.npz   # their spikes and rates, every step
```

A selection is `indices=...`, `ids=...` (the connectome's ids), `type_re=...`,
`superclass=...`, or `name=` one of `readout`, `retina`, `audition`, `punish`. Artifacts
carry every neuron's type and superclass, so the regexes work at runtime without the
training package. The probe file holds `t`, `spikes` (steps x neurons), `rates`, `indices`
and `ids`. From Python: `model.stimulate(sel, mv)`, `model.silence(sel)`,
`model.probe(sel)` then `model.last_probe`, and `model.clear()`.

On the body (`neurofly watch --task forward`), `--stimulate` and `--silence` work the same
way through the connectome's annotations; probes are for PC tasks.

## Watching the brain

Every neuron in an artifact has a position: the soma location from the MaleCNS annotations,
in micrometres (`neurons.positions` in the manifest). Sensory neurons, whose cell bodies
lie outside the nervous system, are placed near the neurons they are annotated with and
flagged in `neurons.positions_known`. Two more experiment options record what every neuron
does with them:

```powershell
neurofly-core run artifacts/myapp --window "My App" --activity-out activity.json   # a file
neurofly play --window "My App" --run runs/x --activity-ws 127.0.0.1:8767          # live
neurofly watch runs/walk1 --activity-out activity.json                             # the body too
python -m http.server 8000   # then examples/brain_viewer.html?activity=../activity.json
                             # or   examples/brain_viewer.html?ws=ws://127.0.0.1:8767
```

`examples/brain_viewer.html` draws the brain as a point cloud coloured by superclass. A
spike lights its neuron and the light fades over a few frames, so waves of activity are
visible crossing the brain. With `&artifact=../artifacts/myapp` it also reads the synapses
from the artifact and draws the strongest few per neuron (`&edges=2`) as lines that flash
when their presynaptic neuron fires. `--activity-substeps` records per brain step instead of
per observation, for a finer animation at about twenty times the size.

The same works for a brain another program is driving. `neurofly-core serve --ws` accepts
`{"op": "activity", "on": true}` from any client; that client is then *subscribed* and
receives every later step's activity as a pushed message, whoever sent the step. So a Node
game loop drives the brain over the WebSocket while a browser tab watches it:
`brain_viewer.html?ws=ws://127.0.0.1:8765&subscribe=1`. Over stdio or gRPC the step replies
themselves carry `activity` once it is on, and `{"op": "positions"}` (gRPC `Positions`)
returns the map. From Python: `model.watch_activity(True)` then `model.last_activity`
(`indices` and `counts` of the neurons that fired) and `model.activity_map()`.

## Gamepads

A layout can include gamepad buttons and axes: `--pad-buttons a,b,rb --axes lx,ly,rt`. They
become entries in the action vector like everything else (sticks in [-1, 1], triggers in
[0, 1]); `neurofly record` reads a real controller through XInput, and `play` and
`neurofly-core run` drive a virtual Xbox controller through ViGEm (`pip install vgamepad`
and the ViGEmBus driver, Windows). A layout can mix keys, mouse and pad.

## Replays

```powershell
neurofly-core run artifacts/myapp --window "My App" --probe name=readout --probe-out probe.npz
neurofly replay --video videos/live.mp4 --probe probe.npz --out videos/replay.mp4
```

`replay` writes a video with the frames on the left and, on the right, a raster of the
probed neurons over the last steps plus the controls held on that frame (from a recording's
`actions.npy` when `--recording` is given instead of `--video`).

## Scoring an artifact

```powershell
neurofly eval artifacts/myapp data/recordings/run1 data/recordings/run2 --reward my_project.py:Score
```

Plays each recording through the artifact, compares the policy's controls with what you did
(precision, recall and F1 per key and button, correlation per axis, overall agreement) and,
with a `Task`, reports the reward the policy would have earned frame by frame. The report
is printed and written as `eval.json` next to the artifact.

## The body outside Python

Body runs export like PC runs, as artifacts of kind `body`: the brain, the sensory map
from the body observation onto the nerve cord, the readout and the policy, plus the
observation layout and the actuator names. The runtime serves them with `body_step`
(JSON) or `BodyStep` (gRPC): an observation vector in, 59 actuator commands out.

```powershell
neurofly export runs/walk1 artifacts/walk
neurofly-core info  artifacts/walk                      # kind body, 286 observation entries, 59 actuators
neurofly-core serve artifacts/walk --grpc 127.0.0.1:50051
```

The simulation is not in the artifact; it runs wherever MuJoCo runs. To take the fly with
you, export the complete MuJoCo model (XML plus every mesh) and the same geometry as glTF:

```powershell
neurofly export-body --out assets/fly.glb --mjcf assets/mujoco    # fly.xml + 85 meshes, loads in any MuJoCo
```

A consumer in another language or engine then loads `fly.xml` in its own MuJoCo (the C
library, the Unity plugin, the WebAssembly build for browsers), assembles the observation
vector the artifact describes (`obs_keys` and `obs_slices` in the manifest), calls
`body_step`, and applies the returned actuator commands. Python consumers do the same with
`flybody` and `neurofly_core.BodyModel`.

To only watch: `examples/three_viewer.html` loads the `.glb` in Three.js and animates it from
a pose stream, either recorded or live:

```powershell
neurofly watch --task forward --brain malecns --policy zero --poses assets/poses.json         # a file
neurofly watch runs/walk1 --pose-ws 127.0.0.1:8766                                          # live, paced to real time
python -m http.server 8000   # then examples/three_viewer.html?glb=../assets/fly.glb&ws=ws://127.0.0.1:8766
```

The live stream sends a header with the body names, then every body's world position (cm)
and quaternion per rendered frame, the same layout as `poses.json`. The viewer rotates from
MuJoCo's z-up to y-up and sets each node's transform directly.

## Where this is going

The artifact is the contract. The plan is a native runtime (a Rust crate with a C ABI, and
Node and Python bindings over it) that reads the same artifact and replaces the spawned
process behind the same client API. Training would then use the native core too, so the two
never drift. Nothing about artifacts or the bindings' public API changes when that lands.
