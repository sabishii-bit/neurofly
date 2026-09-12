// Smoke test of the WebSocket client against a brain served with a token:
//   npm run build && node test_ws.js <artifact> [python]
// Starts `python -m neurofly_core serve <artifact> --ws 127.0.0.1:0 --per-client --token t`,
// connects twice (each connection gets its own brain), records a few steps, and stops.
const { spawn } = require("node:child_process");
const { NeuroFlyWS } = require("./dist");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

async function main() {
  const [artifact, python = "python"] = process.argv.slice(2);
  if (!artifact) { console.error("usage: node test_ws.js <artifact> [python]"); process.exit(2); }
  const port = 18765 + Math.floor(Math.random() * 1000);
  const proc = spawn(python, ["-m", "neurofly_core", "serve", artifact, "--ws", `127.0.0.1:${port}`,
                              "--per-client", "--token", "t"], { stdio: ["ignore", "ignore", "pipe"] });
  await new Promise((res, rej) => {
    proc.stderr.on("data", (d) => { if (String(d).includes("listening")) res(); });
    proc.on("exit", (c) => rej(new Error(`server exited ${c}`)));
  });
  try {
    const url = `ws://127.0.0.1:${port}`;
    const bad = new NeuroFlyWS(url, { token: "wrong" });
    await bad.connect().then(() => { throw new Error("wrong token was accepted"); }, () => {});
    const a = new NeuroFlyWS(url, { token: "t" });
    const b = new NeuroFlyWS(url, { token: "t" });
    const info = await a.connect();
    await b.connect();
    console.log("controls:", info.controls, "per_client:", info.per_client);
    const w = 32, h = 24, frame = new Uint8Array(w * h * 3);
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "neurofly-ws-"));
    await a.record(path.join(dir, "rec"), 10);
    for (let t = 0; t < 3; t++) {
      frame.fill(40 * t);
      const r = await a.observe({ frame, width: w, height: h, action: [1, 0, 0] });
      console.log(`a t=${r.t} features=${r.features.length} recording=${r.recording}`);
    }
    const rb = await b.observe({ frame, width: w, height: h });
    if (rb.t !== 1) throw new Error(`b should have its own brain, got t=${rb.t}`);
    const stopped = await a.stopRecording();
    if (stopped.n_frames !== 3 || !fs.existsSync(path.join(dir, "rec", "actions.npy"))) throw new Error("recording missing");
    console.log("recorded", stopped.n_frames, "frames to", stopped.path);
    console.log("ping:", (await a.ping()).pong);
    await a.close();
    await b.close();
    console.log("ok");
  } finally {
    proc.kill();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
