// Train a policy from Node, with the runtime as a dependency: evolution strategies over the
// linear neuron-to-control table, on a toy world made of synthetic frames.
//
//   neurofly build artifacts/base --brain synthetic --keys w --include-frame   # once, in Python
//   node examples/node_train_es.js artifacts/base artifacts/trained_by_node
//   neurofly-core info artifacts/trained_by_node
//
// The world: frames alternate bright and dark every 10 steps; reward is +1 for holding "w"
// while bright and for releasing it while dark. The brain turns frames into features
// (observe); this script owns the policy, the reward, and the optimiser. Replace the world
// with your program's frames and reward, and the optimiser with whatever you like.
const path = require("path");
const { NeuroFly } = require(path.join(__dirname, "..", "bindings", "node"));

const W = 64, H = 48;
function frame(t) {
  const bright = Math.floor(t / 10) % 2 === 0;
  const f = new Uint8Array(W * H * 3);
  f.fill(bright ? 230 : 20);
  return { f, bright };
}
function gauss() { return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()); }
function policy(theta, nF, nA, features) {           // action = tanh(W f + b)
  const out = new Array(nA);
  for (let a = 0; a < nA; a++) {
    let s = theta[nA * nF + a];
    for (let i = 0; i < nF; i++) s += theta[a * nF + i] * features[i];
    out[a] = Math.tanh(s);
  }
  return out;
}

async function episode(fly, theta, nF, nA, steps) {
  await fly.reset();
  let ret = 0;
  for (let t = 0; t < steps; t++) {
    const { f, bright } = frame(t);
    const r = await fly.observe({ frame: f, width: W, height: H });
    const action = policy(theta, nF, nA, r.features);
    const holds = action[0] > 0;
    ret += holds === bright ? 1 : 0;
  }
  return ret / steps;
}

async function main() {
  const [base, out] = process.argv.slice(2);
  if (!base || !out) throw new Error("usage: node examples/node_train_es.js <base artifact> <out artifact>");
  const fly = new NeuroFly(base);
  const info = await fly.start();
  const nF = info.n_features, nA = info.n_actions;
  console.log(`${info.name}: ${nF} features, ${nA} actions ${JSON.stringify(info.controls)}`);

  // OpenAI-style ES with antithetic sampling, on the flat vector [W (nA x nF), b (nA)].
  let theta = new Float64Array(nA * nF + nA);
  const pop = 8, sigma = 0.1, lr = 0.05, generations = 15, steps = 30;
  for (let g = 0; g < generations; g++) {
    const eps = [], returns = [];
    for (let k = 0; k < pop; k++) {
      const e = Float64Array.from(theta, () => gauss());
      eps.push(e);
      const plus = theta.map((v, i) => v + sigma * e[i]);
      const minus = theta.map((v, i) => v - sigma * e[i]);
      returns.push([await episode(fly, plus, nF, nA, steps), await episode(fly, minus, nF, nA, steps)]);
    }
    const grad = new Float64Array(theta.length);
    for (let k = 0; k < pop; k++) {
      const d = (returns[k][0] - returns[k][1]) / (2 * sigma * pop);
      for (let i = 0; i < grad.length; i++) grad[i] += d * eps[k][i];
    }
    theta = theta.map((v, i) => v + lr * grad[i]);
    const mean = returns.flat().reduce((a, b) => a + b, 0) / (2 * pop);
    console.log(`gen ${g}: mean reward ${mean.toFixed(2)}  (1.0 = always right)`);
  }

  // Install the trained table into the runtime and save it as a complete artifact.
  const Wm = [];
  for (let a = 0; a < nA; a++) Wm.push(Array.from(theta.slice(a * nF, (a + 1) * nF)));
  const b = Array.from(theta.slice(nA * nF));
  await fly.setPolicy({ type: "linear", W: Wm, b });
  const saved = await fly.save(out, "trained_by_node");
  console.log("saved", saved.path);
  const r = await fly.step({ frame: frame(0).f, width: W, height: H });   // now returns controls
  console.log("bright frame ->", r.keys, "; dark frame ->", (await fly.step({ frame: frame(10).f, width: W, height: H })).keys);
  await fly.close();
}

main().catch((e) => { console.error(e.message); process.exit(1); });
