// A trained neurofly controller from Node. Spawns `neurofly-core serve <artifact>` and talks
// JSON lines to it (see artifact/SPEC.md). Needs neurofly-core installed in a Python on PATH,
// or pass { python: "path/to/python" } to run it as `python -m neurofly_core`.
import { spawn, ChildProcess } from "node:child_process";
import * as readline from "node:readline";

export interface Layout {
  keys: string[]; buttons: string[]; mouse: boolean; scroll: boolean;
  mouse_speed: number; scroll_speed: number; pad_buttons: string[]; axes: string[];
}

export interface Info {
  ok: boolean; ready?: boolean; name: string; n_neurons: number; n_features: number;
  n_actions: number; controls: string[]; layout: Layout; brain_ms: number; has_policy: boolean;
  has_audition: boolean; sample_rate: number | null; retina_grid: [number, number];
  has_annotations: boolean; has_positions: boolean; populations: Record<string, number>;
  detection_classes: string[] | null;
  odour_channels: string[] | null; taste_channels: string[] | null; thermo_channels: string[] | null;
}

export interface StepInput {
  /** Raw RGB bytes (row-major, 3 per pixel) of width x height, or an encoded image with `format`. */
  frame: Uint8Array | Buffer;
  width: number;
  height: number;
  format?: "rgb" | "png" | "jpeg";
  /** Samples since the last step in [-1, 1], interleaved if channels > 1. */
  audio?: Float32Array;
  sampleRate?: number;
  channels?: number;
  /** Dopamine for plasticity; negative values drive the punishment neurons. */
  reward?: number;
  observeOnly?: boolean;
  /** Objects a detector found, for an artifact with a detection encoder (see Info.detection_classes). */
  detections?: Detection[];
  /** Odour channel values in [0, 1] (a vector in Info.odour_channels order, or name to value);
   *  omitted keeps the last odours. */
  odours?: number[] | Record<string, number>;
  /** Taste and temperature/humidity channels, the same way (Info.taste_channels, thermo_channels). */
  tastes?: number[] | Record<string, number>;
  thermo?: number[] | Record<string, number>;
}

/** One detected object: a class id or name from the artifact's classes, a box in fractions
 *  of the frame (left, top, right, bottom), and a confidence in [0, 1]. */
export interface Detection { class: number | string; box: [number, number, number, number]; score?: number; }

export interface Probe { spikes: number[]; rates: number[]; }

/** Every neuron that fired during one step (see `activity()`). */
export interface Activity { t: number; indices: number[]; counts: number[]; steps?: number[][]; }

/** Where every neuron is, for drawing the brain (see `positions()`). */
export interface BrainMap {
  ok: true; n: number; unit: string; positions: number[][] | null; known: boolean[] | null;
  superclass: string[] | null; populations: Record<string, number[]>;
}

export interface StepResult {
  ok: true; t: number; spikes: number; action: number[]; held: string[];
  keys: string[]; buttons: string[]; dx: number; dy: number; scroll: number;
  pad_buttons?: string[]; axes?: Record<string, number>;
  features?: number[]; probe?: Probe; activity?: Activity;
}

export interface ObserveResult {
  ok: true; t: number; spikes: number; features: number[]; probe?: Probe; activity?: Activity;
}

export type LinearPolicy = { type: "linear"; W: number[][]; b: number[] };
export type MlpPolicy = {
  type: "mlp"; layers: { W: number[][]; b: number[] }[]; activation?: "tanh" | "relu";
  obs_mean?: number[]; obs_var?: number[]; obs_clip?: number; obs_eps?: number;
};

/** Which neurons: exactly one of these. */
export type Selection =
  | { indices: number[] } | { ids: number[] } | { type_re: string }
  | { superclass: string } | { name: "readout" | "retina" | "audition" | "punish" };

export interface Options {
  /** The runtime executable (default "neurofly-core"). */
  command?: string;
  /** Run `python -m neurofly_core` with this interpreter instead. */
  python?: string;
  device?: string;
}

interface Pending { resolve: (v: any) => void; reject: (e: Error) => void; }

export class NeuroFly {
  readonly artifact: string;
  readonly opts: Options;
  info: Info | null = null;
  private proc: ChildProcess | null = null;
  private queue: Pending[] = [];

  constructor(artifact: string, opts: Options = {}) {
    this.artifact = artifact;
    this.opts = opts;
  }

  /** Start the runtime; resolves with its info (controls, feature and action sizes). */
  start(): Promise<Info> {
    const device = this.opts.device ?? "cpu";
    const [cmd, args] = this.opts.python
      ? [this.opts.python, ["-m", "neurofly_core", "serve", this.artifact, "--device", device]]
      : [this.opts.command ?? "neurofly-core", ["serve", this.artifact, "--device", device]];
    const proc = spawn(cmd, args, { stdio: ["pipe", "pipe", "inherit"] });
    this.proc = proc;
    proc.on("exit", (code) => {
      while (this.queue.length) this.queue.shift()!.reject(new Error(`neurofly-core exited (${code})`));
    });
    const rl = readline.createInterface({ input: proc.stdout! });
    rl.on("line", (line) => {
      const pending = this.queue.shift();
      if (!pending) return;
      try { pending.resolve(JSON.parse(line)); } catch (e) { pending.reject(e as Error); }
    });
    return this.next<Info>().then((info) => { this.info = info; return info; });
  }

  private next<T>(): Promise<T> {
    return new Promise<T>((resolve, reject) => this.queue.push({ resolve, reject }));
  }

  private send<T extends { ok: boolean; error?: string }>(req: object): Promise<T> {
    if (!this.proc) return Promise.reject(new Error("call start() first"));
    const p = this.next<T>();
    this.proc.stdin!.write(JSON.stringify(req) + "\n");
    return p.then((r) => { if (!r.ok) throw new Error(r.error); return r; });
  }

  private stepRequest(op: "step" | "observe", s: StepInput): object {
    const req: Record<string, unknown> = {
      op, frame: Buffer.from(s.frame.buffer, s.frame.byteOffset, s.frame.byteLength).toString("base64"),
      width: s.width, height: s.height,
    };
    if (s.format) req.format = s.format;
    if (s.audio) {
      req.audio = Buffer.from(s.audio.buffer, s.audio.byteOffset, s.audio.byteLength).toString("base64");
      req.sample_rate = s.sampleRate ?? 16000;
      req.channels = s.channels ?? 1;
    }
    if (s.reward !== undefined) req.reward = s.reward;
    if (s.observeOnly) req.observe_only = true;
    if (s.detections && s.detections.length) req.detections = s.detections;
    if (s.odours !== undefined) req.odours = s.odours;
    if (s.tastes !== undefined) req.tastes = s.tastes;
    if (s.thermo !== undefined) req.thermo = s.thermo;
    return req;
  }

  /** Start a fresh episode. */
  reset(): Promise<{ ok: true }> { return this.send({ op: "reset" }); }

  /** One step: frame (and sound) in, controls out. Without a policy the result carries `features`. */
  step(input: StepInput): Promise<StepResult> { return this.send(this.stepRequest("step", input)); }

  /** Like step, but returns the feature vector instead of controls. */
  observe(input: StepInput): Promise<ObserveResult> { return this.send(this.stepRequest("observe", input)); }

  /** Install a policy you trained. */
  setPolicy(policy: LinearPolicy | MlpPolicy): Promise<{ ok: true; type: string }> {
    return this.send({ op: "set_policy", ...policy });
  }

  /** Write the runtime's current model (with the installed policy) as an artifact. */
  save(dir: string, name?: string): Promise<{ ok: true; path: string }> {
    return this.send({ op: "save", path: dir, name });
  }

  /** Extra drive (mV) on the selected neurons on every brain step. Resolves with the count. */
  stimulate(sel: Selection, mv: number): Promise<{ ok: true; n: number }> {
    return this.send({ op: "stimulate", ...sel, mv });
  }

  /** The selected neurons stop spiking. */
  silence(sel: Selection): Promise<{ ok: true; n: number }> { return this.send({ op: "silence", ...sel }); }

  /** Later step / observe results carry `probe` for these neurons; null turns it off. */
  probe(sel: Selection | null): Promise<{ ok: true; n: number }> {
    return this.send(sel ? { op: "probe", ...sel } : { op: "probe", off: true });
  }

  /** Undo every stimulation and silencing. */
  clear(): Promise<{ ok: true }> { return this.send({ op: "clear" }); }

  /** Later step / observe results carry `activity`: every neuron that fired. */
  activity(on = true, substeps = false): Promise<{ ok: true; n: number }> {
    return this.send({ op: "activity", on, substeps });
  }

  /** Every neuron's position (micrometres) and superclass, for a viewer. */
  positions(): Promise<BrainMap> { return this.send({ op: "positions" }); }

  /** Neuron indices of a selection. */
  select(sel: Selection): Promise<{ ok: true; n: number; indices: number[] }> {
    return this.send({ op: "select", ...sel });
  }

  /** Stop the runtime and release everything. */
  async close(): Promise<void> {
    if (!this.proc) return;
    try { await this.send({ op: "close" }); } catch { /* already gone */ }
    this.proc.kill();
    this.proc = null;
  }
}

export default NeuroFly;
