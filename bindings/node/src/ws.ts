// The same client over a WebSocket, for browsers and for Node talking to a hosted brain
// (`neurofly-core serve <artifact> --ws host:port`, or the Docker image). No Node-only
// imports: this file runs in a browser as is. Node 22+ has a global WebSocket; older Node
// can pass one in (`{ WebSocket: require("ws") }`).
import type {
  Activity, BrainMap, Info, LinearPolicy, MlpPolicy, ObserveResult, Selection, StepInput, StepResult,
} from "./index";

export interface WSOptions {
  /** The token the server was started with (`--token` / NEUROFLY_TOKEN). */
  token?: string;
  /** A WebSocket constructor, when the runtime has no global one. */
  WebSocket?: any;
  /** Called with every activity message the server pushes (after `activity(true)`). */
  onActivity?: (a: Activity) => void;
}

interface Pending { resolve: (v: any) => void; reject: (e: Error) => void; }

function toBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64");
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + 0x8000)));
  return btoa(s);
}

export class NeuroFlyWS {
  readonly url: string;
  readonly opts: WSOptions;
  info: Info | null = null;
  private ws: any = null;
  private queue: Pending[] = [];

  constructor(url: string, opts: WSOptions = {}) {
    this.url = url;
    this.opts = opts;
  }

  /** Open the connection; resolves with the server's info. */
  connect(): Promise<Info> {
    const Ctor = this.opts.WebSocket ?? (globalThis as any).WebSocket;
    if (!Ctor) throw new Error("no WebSocket available: pass one in the options");
    const url = this.opts.token ? `${this.url}${this.url.includes("?") ? "&" : "?"}token=${encodeURIComponent(this.opts.token)}` : this.url;
    const ws = new Ctor(url);
    this.ws = ws;
    const ready = this.next<Info>();
    ws.onmessage = (ev: any) => {
      let msg: any;
      try { msg = JSON.parse(typeof ev.data === "string" ? ev.data : String(ev.data)); } catch { return; }
      if (msg.ok === undefined && Array.isArray(msg.indices)) { this.opts.onActivity?.(msg); return; }
      const pending = this.queue.shift();
      pending?.resolve(msg);
    };
    ws.onclose = () => { while (this.queue.length) this.queue.shift()!.reject(new Error("connection closed")); };
    ws.onerror = () => { while (this.queue.length) this.queue.shift()!.reject(new Error(`could not connect to ${this.url}`)); };
    return ready.then((info) => { if (!info.ok) throw new Error((info as any).error); this.info = info; return info; });
  }

  private next<T>(): Promise<T> {
    return new Promise<T>((resolve, reject) => this.queue.push({ resolve, reject }));
  }

  private send<T extends { ok: boolean; error?: string }>(req: object): Promise<T> {
    if (!this.ws) return Promise.reject(new Error("call connect() first"));
    const p = this.next<T>();
    this.ws.send(JSON.stringify(req));
    return p.then((r) => { if (!r.ok) throw new Error(r.error); return r; });
  }

  private stepRequest(op: "step" | "observe", s: StepInput & { action?: number[] }): object {
    const req: Record<string, unknown> = { op, frame: toBase64(s.frame instanceof Uint8Array ? s.frame : new Uint8Array(s.frame)), width: s.width, height: s.height };
    if (s.format) req.format = s.format;
    if (s.audio) { req.audio = toBase64(new Uint8Array(s.audio.buffer, s.audio.byteOffset, s.audio.byteLength)); req.sample_rate = s.sampleRate ?? 16000; req.channels = s.channels ?? 1; }
    if (s.reward !== undefined) req.reward = s.reward;
    if (s.observeOnly) req.observe_only = true;
    if (s.detections && s.detections.length) req.detections = s.detections;
    for (const k of ["odours", "tastes", "thermo", "touch", "pulses", "action"] as const) {
      if ((s as any)[k] !== undefined) req[k] = (s as any)[k];
    }
    return req;
  }

  /** A canvas as an encoded frame for `step` / `observe` (browsers). */
  static async fromCanvas(canvas: any, type = "image/jpeg", quality = 0.85): Promise<Pick<StepInput, "frame" | "width" | "height" | "format">> {
    const blob: Blob = await new Promise((res) => canvas.toBlob(res, type, quality));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return { frame: bytes, width: canvas.width, height: canvas.height, format: type === "image/png" ? "png" : "jpeg" };
  }

  ping(): Promise<{ ok: true; pong: true; t: number }> { return this.send({ op: "ping" }); }
  reset(): Promise<{ ok: true }> { return this.send({ op: "reset" }); }
  step(input: StepInput): Promise<StepResult> { return this.send(this.stepRequest("step", input)); }
  /** `action`: what your program actually did this step, recorded as the label when recording. */
  observe(input: StepInput & { action?: number[] }): Promise<ObserveResult> { return this.send(this.stepRequest("observe", input)); }
  setPolicy(policy: LinearPolicy | MlpPolicy): Promise<{ ok: true; type: string }> { return this.send({ op: "set_policy", ...policy }); }
  save(dir: string, name?: string): Promise<{ ok: true; path: string }> { return this.send({ op: "save", path: dir, name }); }
  stimulate(sel: Selection, mv: number): Promise<{ ok: true; n: number }> { return this.send({ op: "stimulate", ...sel, mv }); }
  silence(sel: Selection): Promise<{ ok: true; n: number }> { return this.send({ op: "silence", ...sel }); }
  probe(sel: Selection | null): Promise<{ ok: true; n: number }> { return this.send(sel ? { op: "probe", ...sel } : { op: "probe", off: true }); }
  clear(): Promise<{ ok: true }> { return this.send({ op: "clear" }); }
  select(sel: Selection): Promise<{ ok: true; n: number; indices: number[] }> { return this.send({ op: "select", ...sel }); }
  activity(on = true, substeps = false): Promise<{ ok: true; n: number }> { return this.send({ op: "activity", on, substeps }); }
  positions(): Promise<BrainMap> { return this.send({ op: "positions" }); }
  /** Start writing every later step's frame, sound and action to a recording directory on the server. */
  record(path: string, fps = 10): Promise<{ ok: true; path: string; recording: boolean }> { return this.send({ op: "record", path, fps }); }
  stopRecording(): Promise<{ ok: true; recording: boolean; n_frames: number; path?: string }> { return this.send({ op: "record", off: true }); }

  async close(): Promise<void> {
    if (!this.ws) return;
    try { await this.send({ op: "close" }); } catch { /* already gone */ }
    this.ws.close();
    this.ws = null;
  }
}
