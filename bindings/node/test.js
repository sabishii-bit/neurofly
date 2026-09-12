// Smoke test against a built package: npm run build && node test.js <artifact> [python]
const { NeuroFly } = require("./dist");

async function main() {
  const [artifact, python] = process.argv.slice(2);
  if (!artifact) { console.error("usage: node test.js <artifact> [python]"); process.exit(2); }
  const fly = new NeuroFly(artifact, python ? { python } : {});
  const info = await fly.start();
  console.log("controls:", info.controls, "features:", info.n_features, "populations:", info.populations);
  const w = 64, h = 48;
  const frame = new Uint8Array(w * h * 3);
  const audio = new Float32Array(1600);
  await fly.probe({ name: "readout" });
  for (let t = 0; t < 5; t++) {
    for (let i = 0; i < frame.length; i++) frame[i] = (i * 7 + t * 31) & 255;
    for (let i = 0; i < audio.length; i++) audio[i] = 0.2 * Math.sin(i * 0.1);
    const r = await fly.step({ frame, width: w, height: h, audio, sampleRate: 16000 });
    const held = r.held ? r.held.join(",") : `${r.features.length} features`;
    console.log(`t=${r.t} ${held} spikes=${r.spikes} probe=${r.probe ? r.probe.spikes.length : 0}`);
  }
  await fly.close();
  console.log("ok");
}

main().catch((e) => { console.error(e); process.exit(1); });
