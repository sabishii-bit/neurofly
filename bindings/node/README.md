# neurofly for Node

TypeScript, compiled to CommonJS with declarations. Build once after checking out:

```
cd bindings/node && npm install && npm run build
```

```ts
import { NeuroFly } from "neurofly";          // or require("../bindings/node")

const fly = new NeuroFly("artifacts/myapp");  // or { python: "C:/path/python.exe" }
const info = await fly.start();                // info.controls, info.n_features, info.layout, ...
const r = await fly.step({ frame, width: 320, height: 240, audio, sampleRate: 16000 });
// r.keys ["w"], r.buttons [], r.dx 3.1, r.dy -0.4, r.scroll 0, r.pad_buttons, r.axes, r.action, r.spikes
await fly.close();
```

`frame` is raw RGB bytes (row-major, 3 per pixel) or an encoded image with
`format: "png"`. `audio` is a `Float32Array` of samples since the last step. Apply the
returned controls however your program takes input.

Training from Node: `observe` returns the feature vector, `setPolicy` installs a policy you
trained, `save` writes a complete artifact. Experiments: `stimulate`, `silence`, `probe`,
`clear`, `select`. Every call is typed; see `src/index.ts`.

The body: `NeuroFlyBody` drives a fly hosted by `neurofly body-serve` over a WebSocket
(per-leg joints, named actuators, the tripod gait, kinematic poses and replays, the served
brain acting) and receives every rendered pose; `node test_body.js [python]` is its smoke
test. See `src/body.ts` and `docs/runtime.md`.

The package spawns `neurofly-core serve <artifact>`, so the runtime must be installed in a
Python on the PATH (`pip install neurofly-core`), or pass `{ python }`. The protocol is in
`artifact/SPEC.md`; `test.js` is a smoke test:

```
node test.js ../../artifacts/myapp
```
