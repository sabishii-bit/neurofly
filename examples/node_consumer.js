// A Node program that owns its own frames and applies the controls itself.
// Run:  node examples/node_consumer.js artifacts/myapp
// Needs neurofly-core installed in a Python on the PATH (see bindings/node/README.md).
const path = require("path");
const { NeuroFly } = require(path.join(__dirname, "..", "bindings", "node"));

async function main() {
  const artifact = process.argv[2];
  if (!artifact) throw new Error("usage: node examples/node_consumer.js <artifact>");
  const fly = new NeuroFly(artifact);
  const info = await fly.start();
  console.log(`loaded ${info.name}: ${info.n_neurons} neurons, controls ${info.controls}`);

  // Your program's frame: here a moving gradient at 64x48.
  const w = 64, h = 48, frame = new Uint8Array(w * h * 3);
  for (let t = 0; t < 20; t++) {
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 3, v = ((x + t * 4) * 4) & 255;
        frame[i] = frame[i + 1] = frame[i + 2] = v;
      }
    const r = await fly.step({ frame, width: w, height: h });
    // Apply r.keys / r.buttons / r.dx / r.dy / r.scroll to your world here. A base artifact
    // (no policy yet) answers with r.features instead: train one, see node_train_es.js.
    if (r.features) console.log(`t=${r.t} no policy: ${r.features.length} features, ${r.spikes} spikes`);
    else console.log(`t=${r.t} holds ${JSON.stringify(r.keys)} mouse ${r.dx.toFixed(1)},${r.dy.toFixed(1)}`);
  }
  await fly.close();
}

main().catch((e) => { console.error(e.message); process.exit(1); });
