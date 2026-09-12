// Smoke test of the body client against a served body:
//   npm run build && node test_body.js [python]
// Starts `python -m neurofly_training body-serve --ws 127.0.0.1:<port> --no-realtime`, drives the
// fly with named actuators and the gait, sets a pose, and counts the poses pushed meanwhile.
const { spawn } = require("node:child_process");
const { NeuroFlyBody } = require("./dist");

async function main() {
  const [python = "python"] = process.argv.slice(2);
  const port = 18767 + Math.floor(Math.random() * 1000);
  const proc = spawn(python, ["-m", "neurofly_training", "body-serve", "--ws", `127.0.0.1:${port}`,
                              "--no-realtime"], { stdio: ["ignore", "ignore", "pipe"] });
  await new Promise((res, rej) => {
    proc.stderr.on("data", (d) => { if (String(d).includes("listening")) res(); });
    proc.on("exit", (c) => rej(new Error(`server exited ${c}`)));
  });
  try {
    let pushed = 0;
    const fly = new NeuroFlyBody(`ws://127.0.0.1:${port}`, { onPose: () => { pushed++; } });
    const info = await fly.connect();
    console.log("task:", info.task, "actuators:", info.n_actions, "bodies:", info.bodies.length, "fps:", info.fps);
    if (info.actuators[0] !== "adhere_claw_T1_left") throw new Error("unexpected action order");
    const r = await fly.step({ legs: { T1L: { coxa: 0.4 } }, steps: 20 });
    console.log(`step t=${r.t} reward=${r.reward.toFixed(3)} obs=${r.obs.length} pose=${r.pose.length}`);
    const g = await fly.gait({ steps: 200, stride_hz: 2 });
    console.log(`gait t=${g.t} done=${g.done} root=${g.root.map((v) => v.toFixed(3))}`);
    const p = await fly.setPose({ coxa_T1_left: 0.8 });
    if (p.qpos.length < 100) throw new Error("qpos missing");
    const f = await fly.frame({ width: 64, height: 48 });
    if (!f.png.startsWith("iVBOR")) throw new Error("not a PNG");
    const o = await fly.observe();
    if (o.t !== 220) throw new Error(`expected t=220, got ${o.t}`);
    await new Promise((res) => setTimeout(res, 200));
    if (pushed < 22) throw new Error(`expected at least 22 pushed poses, got ${pushed}`);
    console.log("pushed poses:", pushed);
    await fly.reset();
    await fly.close();
    console.log("ok");
  } finally {
    proc.kill();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
