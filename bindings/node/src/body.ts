// The fly body over a WebSocket: a client for `neurofly body-serve`, for browsers and Node.
// The server hosts MuJoCo; this drives it (actions, the tripod gait, poses, replays) and
// receives every rendered pose, so a page can animate the exported .glb while a program
// controls the limbs. No Node-only imports.

export interface BodyInfo {
  ok: boolean; ready?: boolean; kind: "body-server"; task: string; tasks: string[];
  control_hz: number; every: number; fps: number; realtime: boolean;
  n_obs: number; obs_keys: string[]; obs_slices: Record<string, [number, number]>;
  actuators: string[]; n_actions: number; leg_joints: string[]; legs: string[];
  rest: number[]; bodies: string[]; units: "cm"; up: "z";
  brain: string | null; brain_has_policy: boolean; t: number; done: boolean;
}

/** One rendered pose: every body's [x, y, z, qw, qx, qy, qz], in `BodyInfo.bodies` order. */
export interface Pose { t: number; pose: number[]; }

export interface StepResult {
  ok: true; t: number; steps: number; reward: number; done: boolean;
  obs: number[]; pose: number[]; root: number[]; action: number[]; spikes?: number;
}

export type Leg = "T1L" | "T1R" | "T2L" | "T2R" | "T3L" | "T3R";
export type LegJoint = "coxa_abduct" | "coxa_twist" | "coxa" | "femur_twist" | "femur" | "tibia" | "tarsus" | "tarsus2" | "claw";

export interface StepInput {
  /** The full action, 59 values in [-1, 1] in `BodyInfo.actuators` order. */
  action?: number[];
  /** Offsets from the standing pose for named actuators (`BodyInfo.actuators`). */
  actuators?: Record<string, number>;
  /** Offsets per leg and joint; `claw` is the adhesion in [-1, 1]. */
  legs?: Partial<Record<Leg, Partial<Record<LegJoint, number>>>>;
  /** Let the loaded brain choose (the server must have been started with --artifact). */
  brain?: boolean;
  /** Control steps to run with this action (2 ms each). Default 1. */
  steps?: number;
  reward?: number;
  pulses?: Record<string, number>;
  /** Pace to real time (the server's default) or run as fast as possible. */
  realtime?: boolean;
}

export interface GaitInput { steps?: number; stride_hz?: number; stride?: number; lift?: number; adhesion?: number; realtime?: boolean; }

export interface BodyOptions {
  token?: string;
  WebSocket?: any;
  /** Called with every pose the server pushes (every rendered frame of every step, whoever drove it). */
  onPose?: (p: Pose) => void;
}

interface Pending { resolve: (v: any) => void; reject: (e: Error) => void; }

export class NeuroFlyBody {
  readonly url: string;
  readonly opts: BodyOptions;
  info: BodyInfo | null = null;
  private ws: any = null;
  private queue: Pending[] = [];

  constructor(url: string, opts: BodyOptions = {}) {
    this.url = url;
    this.opts = opts;
  }

  /** Open the connection; resolves with the server's description of the body. */
  connect(): Promise<BodyInfo> {
    const Ctor = this.opts.WebSocket ?? (globalThis as any).WebSocket;
    if (!Ctor) throw new Error("no WebSocket available: pass one in the options");
    const url = this.opts.token ? `${this.url}${this.url.includes("?") ? "&" : "?"}token=${encodeURIComponent(this.opts.token)}` : this.url;
    const ws = new Ctor(url);
    this.ws = ws;
    const ready = this.next<BodyInfo>();
    ws.onmessage = (ev: any) => {
      let msg: any;
      try { msg = JSON.parse(typeof ev.data === "string" ? ev.data : String(ev.data)); } catch { return; }
      if (msg.ok === undefined && Array.isArray(msg.pose)) { this.opts.onPose?.(msg); return; }
      this.queue.shift()?.resolve(msg);
    };
    ws.onclose = () => { while (this.queue.length) this.queue.shift()!.reject(new Error("connection closed")); };
    ws.onerror = () => { while (this.queue.length) this.queue.shift()!.reject(new Error(`could not connect to ${this.url}`)); };
    return ready.then((info) => { if (info.ok === false) throw new Error((info as any).error); this.info = info; return info; });
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

  ping(): Promise<{ ok: true; pong: true; t: number }> { return this.send({ op: "ping" }); }
  /** A fresh episode, optionally of another task. */
  reset(task?: string, seed?: number): Promise<{ ok: true; task: string; t: number; obs: number[]; pose: number[] }> {
    return this.send({ op: "reset", task, seed });
  }
  /** Run control steps with an action (see `StepInput`); an empty input stands still. */
  step(input: StepInput = {}): Promise<StepResult> { return this.send({ op: "step", ...input }); }
  /** The open-loop tripod gait through the physics for `steps` control steps (default 500: one second). */
  gait(input: GaitInput = {}): Promise<StepResult & { gait: { stride_hz: number; stride: number; lift: number } }> {
    return this.send({ op: "gait", ...input });
  }
  /** Put joints where you say, no physics: joint name (e.g. "coxa_T1_left") to angle in radians, or the whole qpos. */
  setPose(joints?: Record<string, number>, qpos?: number[]): Promise<{ ok: true; t: number; pose: number[]; qpos: number[] }> {
    return this.send({ op: "set_pose", joints, qpos });
  }
  /** Play a joint trajectory kinematically: the tripod gait, or a real fly's walking from flybody's dataset. */
  replay(source: "gait" | "real" = "gait", opts: { index?: number; steps?: number; stride_hz?: number; realtime?: boolean } = {}):
    Promise<{ ok: true; source: string; frames: number; steps: number; fps: number; pose: number[] }> {
    return this.send({ op: "replay", source, ...opts });
  }
  /** The current observation, pose and root position without stepping. */
  observe(): Promise<{ ok: true; t: number; done: boolean; obs: number[]; pose: number[]; root: number[]; action: number[] }> {
    return this.send({ op: "observe" });
  }
  /** A rendered frame as PNG bytes (base64 in `png`). */
  frame(opts: { camera?: number; width?: number; height?: number } = {}): Promise<{ ok: true; width: number; height: number; format: "png"; png: string }> {
    return this.send({ op: "frame", ...opts });
  }
  /** Receive (or stop receiving) pushed poses on this connection; on by default. */
  poses(on = true): Promise<{ ok: true; on: boolean }> { return this.send({ op: "poses", on }); }

  async close(): Promise<void> {
    if (!this.ws) return;
    try { await this.send({ op: "close" }); } catch { /* already gone */ }
    this.ws.close();
    this.ws = null;
  }
}
