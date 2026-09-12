# Using neurofly from your own project

neurofly is a kit: it does the parts of a fly-brain project that are hard to get right
(loading the connectome, simulating it fast, wiring senses and outputs to real annotations,
saving the result in a portable form, serving it) so that your project can be about the
game, the robot, the experiment. Your project can be in any language. This page is the
step-by-step for the common shapes.

## The shape of every integration

1. **Python builds the brain, once.** `neurofly build` (or `train`, `imitate`, ...) writes an
   *artifact*: a directory of a manifest plus flat binary arrays with no Python objects in it
   ([artifact/SPEC.md](../artifact/SPEC.md)). This is the only step that needs the training
   package and the connectome data. It can happen on another machine.
2. **`neurofly-core` runs the artifact.** A small Python package with the simulation and the
   encoders, and a command that serves the artifact over JSON lines (stdio), a WebSocket or
   gRPC. It needs no training stack and no data files.
3. **Your program talks to it.** Frames, sound, detections, odours in; controls or features
   out; and, for training, `set_policy` and `save`. The bindings in `bindings/` wrap that
   for TypeScript, Rust and Go; any other language speaks the protocol directly.

Where Python is unavoidable, and where it is not, is spelled out in the
[README](../README.md#where-python-is-required).

## Step 0: an artifact to talk to

On any machine with the training package and the data:

```powershell
pip install -e core[pc,ws,grpc] -e training
neurofly download                                              # the connectome, once
neurofly build artifacts/base --brain malecns --keys w,a,s,d --mouse \
    --detect "owl2:enemy,health pack" --odours health,danger     # senses you want
```

`artifacts/base` has the brain and encoders and no policy: it answers `observe` with feature
vectors, which is what a trainer in another language wants. A trained artifact (from
`neurofly export` or from your own `set_policy` + `save`) answers `step` with controls too.
Copy the directory wherever your program runs. Artifacts are self-describing: `neurofly-core
info` prints what one expects (its classes, channels, controls).

## TypeScript / Node

The runtime machine needs Python 3.10+ with `neurofly-core` (`pip install -e core[pc]`,
or `core[ws]` for browsers). Then:

```powershell
npm install <path-or-git-url>/bindings/node     # builds dist/ on install (tsc)
```

```ts
import { NeuroFly } from "neurofly";

const fly = new NeuroFly("artifacts/base", { python: "C:/path/to/python.exe" });
const info = await fly.start();          // controls, feature/action sizes, detection classes, odour channels
console.log(info.controls, info.n_features);

// inference: a frame (RGB bytes) in, controls out
const r = await fly.step({ frame: rgb, width: 320, height: 240,
                           detections: [{ class: "enemy", box: [0.4, 0.3, 0.6, 0.7], score: 0.9 }],
                           odours: { health: 0.8 } });
if (r.keys.includes("w")) { /* press w in your game */ }

// training: features in, your optimiser, a policy back into the runtime, an artifact out
const f = await fly.observe({ frame: rgb, width: 320, height: 240, reward: 0.5 });   // f.features
await fly.setPolicy({ type: "linear", W, b });                                        // actions x features
await fly.save("artifacts/trained", "my-game");
await fly.close();
```

`examples/node_consumer.js` drives a served brain from a video; `examples/node_train_es.js`
trains one with evolution strategies entirely in JavaScript. A browser cannot spawn a
process: run `neurofly-core serve artifacts/base --ws 127.0.0.1:8765` and speak the same
JSON over a WebSocket (`examples/brain_viewer.html` is a browser client).

## Rust

```toml
[dependencies]
neurofly = { path = "../neurofly-kit/bindings/rust" }   # or git = "https://github.com/sabishii-bit/neurofly-kit", ...
```

```rust
use neurofly::{Client, Detection, Step};

let mut fly = Client::with_python("C:/path/to/python.exe", "artifacts/base")?;
println!("{:?}", fly.info.controls);
let r = fly.step(&Step::rgb(&frame, 320, 240)
    .detections(vec![Detection { class: 0, bbox: [0.4, 0.3, 0.6, 0.7], score: 0.9 }])
    .odours(vec![0.8, 0.0]))?;
if r.keys.iter().any(|k| k == "w") { /* press w */ }

let f = fly.observe(&Step::rgb(&frame, 320, 240).reward(0.5))?;   // f.features
fly.set_policy_linear(&w, &b)?;
fly.save("artifacts/trained", Some("my-game"))?;
fly.close()?;
```

For a Rust program that must not spawn Python at all, generate a gRPC client from
`core/src/neurofly_core/rpc/neurofly.proto` with `tonic` and connect to
`neurofly-core serve --grpc` running as a service.

## Go

```go
import neurofly "github.com/sabishii-bit/neurofly-kit/bindings/go"   // or a replace directive to a local path

fly, err := neurofly.WithPython("C:/path/to/python.exe", "artifacts/base")
r, err := fly.Step(neurofly.Step{Frame: rgb, Width: 320, Height: 240,
    Detections: []neurofly.Detection{{Class: 0, Box: [4]float32{0.4, 0.3, 0.6, 0.7}, Score: 0.9}},
    Odours: []float32{0.8, 0}})
f, err := fly.Observe(neurofly.Step{Frame: rgb, Width: 320, Height: 240, Reward: &reward})
err = fly.SetPolicyLinear(w, b)
path, err := fly.Save("artifacts/trained", "my-game")
fly.Close()
```

`bindings/go/example` is a complete program.

## Python, without the training package

```powershell
pip install -e core[pc]
```

```python
from neurofly_core.artifact import load_model
model = load_model("artifacts/trained")
state, info = model.step(frame, audio, detections=dets, odours={"health": 0.8})
```

## Any other language

Start `neurofly-core serve <artifact>` and write one JSON object per line to its stdin;
read one per line from its stdout. The first line it writes is the `info`. The whole
protocol is a table in [artifact/SPEC.md](../artifact/SPEC.md#the-server-protocol).
`--ws host:port` and `--grpc host:port` serve the same thing to programs that cannot own a
subprocess.

## Detectors and the senses from your side

* **Detections.** Your program runs the detector and sends boxes; `neurofly detect-train`
  exports `detector.onnx`, which ONNX Runtime loads in every language above. The artifact
  says which classes it expects (`info.detection_classes`).
* **Odours, tastes, temperature, touch.** Slow scalars in [0, 1], one per named channel,
  sent on the step (`odours`, `tastes`, `thermo`, `touch`); a step without them keeps the
  last values. The names are yours (`--odours health,danger` at build time).
* **Pulses.** One-step drives by population name (`pulses: {"giantfibre": 20}` startles the
  fly; `{"clock": 5}`, `{"punish": 15}`, any name in `info.populations`).
* **Reward.** `reward` on `observe` or `step` is dopamine: it drives the PAM neurons when
  positive and PPL1 when negative (if the artifact was built with those options) and feeds
  plasticity.

## Building the artifact somewhere else

`neurofly build` needs the training package and the data. It does not need to be the
machine that runs the game. A typical arrangement is a Python environment on one machine
(or a CI job) that builds and exports artifacts, and the artifact directory checked into
or downloaded by the project that uses it, with only `neurofly-core` installed there.

## Versioning

An artifact records its format version. `neurofly-core validate <artifact>` checks a
directory without loading the brain, and `neurofly-core info` prints what it contains.
The protocol only adds fields; a client that ignores unknown keys keeps working.
