"""The fly body as a server: MuJoCo hosted next to the brain, driven over JSON.

    neurofly body-serve                                   # ws://127.0.0.1:8767, task forward
    neurofly body-serve --artifact artifacts/walk         # with a brain that can act
    neurofly body-serve --stdio                           # JSON lines on stdin/stdout

A client in any language sends one request per line (stdio) or per message (WebSocket)
and gets one JSON response; every response carries the current pose of every body, and
WebSocket clients are also pushed ``{"t": n, "pose": [...]}`` for every rendered frame
of every step, whoever drove it, so ``examples/three_viewer.html?ws=ws://host:port``
shows the body live while a program controls it.

Requests (``op``):

    hello / info                 what is here: task, actuator names, observation layout, ...
    reset  {task?, seed?}        a fresh episode (optionally another task)
    step   {action | actuators | legs | brain, steps, reward, realtime}
                                 run control steps with an action: the full 59-vector, named
                                 actuators over the standing pose, per-leg joints, or the
                                 loaded brain's own choice; returns obs, reward, done, pose
    gait   {steps, stride_hz, stride, lift, adhesion}
                                 the open-loop tripod gait through the physics for N steps
    set_pose {joints | qpos}     put the joints where you say, no physics (kinematic)
    replay {source: gait|real, index?, steps?}
                                 play a joint trajectory kinematically (the tripod gait, or
                                 a real fly's walking from flybody's dataset)
    observe                      the current observation and pose, no step
    frame  {camera?, width?, height?}
                                 a rendered PNG, base64
    poses  {on}                  (WebSocket) receive or stop receiving pushed poses
    ping, close

Actions are in [-1, 1] per actuator, in the action order of ``body.actuators``
(claws, head, abdomen, legs). ``actuators`` and ``legs`` are added to ``rest_action``,
the action that holds the standing pose, so an empty request stands still.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time

import numpy as np

from neurofly_training.body.actuators import ACTUATOR_NAMES, LEG_JOINTS, NEUROMERES, SIDE_NAME
from neurofly_training.body.export import PoseRecorder
from neurofly_training.body.gait import TripodGait, rest_action

CONTROL_HZ = 500.0
MAX_STEPS = 50_000          # one request may run at most 100 s of simulated time

LEG_KEYS = {f"{t}{s}": (t, s) for t in NEUROMERES for s in ("L", "R")}
LEG_KEYS.update({f"{t}_{SIDE_NAME[s]}": (t, s) for t in NEUROMERES for s in ("L", "R")})


class BodySession:
    """One simulated fly, driven by request dicts. ``handle`` never raises."""

    def __init__(self, task: str = "forward", seed: int = 0, brain=None, camera_id: int = 1,
                 width: int = 640, height: int = 480, every: int = 10, realtime: bool = True,
                 time_limit: float = float("inf")):
        self.task, self.seed = task, seed
        self.camera_id, self.width, self.height = camera_id, width, height
        self.every = max(1, int(every))               # control steps per rendered pose
        self.realtime = realtime
        self.time_limit = float(time_limit)            # seconds per episode (training tasks: 2)
        self.brain = brain                             # a neurofly_core BodyModel, or None
        self.env = self._make_env(task, seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.rest = rest_action(self.env.physics.model)
        self.recorder = PoseRecorder(lambda: self.env.physics, fps=CONTROL_HZ / self.every)
        self.bodies = self.recorder.bodies
        self.t = 0                                     # control steps since reset
        self.frames = 0                                # poses pushed since start
        self.done = False
        self.last_action = self.rest.copy()
        self.listeners: list = []                      # called with (t, pose_row) per frame
        self.lock = threading.Lock()
        if brain is not None:
            if getattr(brain, "kind", None) != "body":
                raise ValueError("body-serve needs a body artifact (kind body)")
            if brain.n_obs != self.env.observation_space.shape[0]:
                raise ValueError(f"the artifact expects {brain.n_obs} observation entries, "
                                 f"the {task} task gives {self.env.observation_space.shape[0]}")
            brain.reset()

    def _make_env(self, task: str, seed: int):
        """The task's environment, with the served body's episode length where the task
        has one (a hosted fly should not stop after the training tasks' two seconds)."""
        import inspect
        from neurofly_training.body.tasks import TASKS
        from neurofly_training.envs import make_body_env
        kw = {}
        if "time_limit" in inspect.signature(TASKS[task]).parameters:
            kw["time_limit"] = self.time_limit
        return make_body_env(task, seed=seed, render_mode="rgb_array", camera_id=self.camera_id,
                             width=self.width, height=self.height, **kw)

    # --- what is here ---------------------------------------------------------------------

    def info(self) -> dict:
        from neurofly_training.body.tasks import TASKS
        return {"ok": True, "kind": "body-server", "task": self.task, "tasks": sorted(TASKS),
                "time_limit": self.time_limit if self.time_limit != float("inf") else None,
                "control_hz": CONTROL_HZ, "every": self.every, "fps": CONTROL_HZ / self.every,
                "realtime": self.realtime, "n_obs": int(self.env.observation_space.shape[0]),
                "obs_keys": list(self.env.obs_keys),
                "obs_slices": {k: [s.start, s.stop] for k, s in self.env.obs_slices.items()},
                "actuators": list(ACTUATOR_NAMES), "n_actions": len(ACTUATOR_NAMES),
                "leg_joints": list(LEG_JOINTS), "legs": sorted(k for k in LEG_KEYS if "_" not in k),
                "rest": self.rest.tolist(), "bodies": self.bodies, "units": "cm", "up": "z",
                "brain": (getattr(self.brain, "name", "artifact") if self.brain else None),
                "brain_has_policy": bool(self.brain is not None and self.brain.policy is not None),
                "t": self.t, "done": self.done}

    def pose(self) -> list[float]:
        self.recorder.record()
        return self.recorder.frames.pop()

    def _emit(self) -> list[float]:
        row = self.pose()
        for fn in list(self.listeners):
            try:
                fn(self.frames, row)
            except Exception:
                self.listeners.remove(fn)
        self.frames += 1
        return row

    # --- actions ----------------------------------------------------------------------------

    def action_from(self, req: dict) -> np.ndarray:
        """The 59-vector a request means: ``action`` as is; else ``rest`` plus named
        ``actuators`` and per-leg ``legs``; ``brain: true`` asks the loaded brain."""
        if req.get("brain"):
            if self.brain is None:
                raise ValueError("no brain loaded: start body-serve with --artifact")
            action, _ = self.brain.step(self.obs, float(req.get("reward", 0.0)),
                                        pulses=req.get("pulses"))
            return np.asarray(action, np.float32)
        if req.get("action") is not None:
            a = np.asarray(req["action"], dtype=np.float32)
            if a.shape != (len(ACTUATOR_NAMES),):
                raise ValueError(f"action must have {len(ACTUATOR_NAMES)} entries, got {a.shape}")
            return np.clip(a, -1.0, 1.0)
        a = self.rest.copy()
        for name, value in (req.get("actuators") or {}).items():
            if name not in ACTUATOR_NAMES:
                raise ValueError(f"unknown actuator {name!r}")
            a[ACTUATOR_NAMES.index(name)] += float(value)
        for leg, joints in (req.get("legs") or {}).items():
            if leg not in LEG_KEYS:
                raise ValueError(f"unknown leg {leg!r}; use T1L .. T3R")
            t, s = LEG_KEYS[leg]
            for joint, value in joints.items():
                if joint == "claw":
                    name = f"adhere_claw_{t}_{SIDE_NAME[s]}"
                    a[ACTUATOR_NAMES.index(name)] = float(value)     # 0..1 -> -1..1 below
                    continue
                if joint not in LEG_JOINTS:
                    raise ValueError(f"unknown leg joint {joint!r}; one of {LEG_JOINTS} or claw")
                a[ACTUATOR_NAMES.index(f"{joint}_{t}_{SIDE_NAME[s]}")] += float(value)
        return np.clip(a, -1.0, 1.0)

    def _run(self, actions, steps: int, realtime: bool) -> dict:
        """Step the physics ``steps`` times with ``actions(k)``; poses every ``every``."""
        steps = int(steps)
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be 1..{MAX_STEPS}")
        total, start, wall = 0.0, self.t, time.perf_counter()
        info = {}
        for k in range(steps):
            a = np.asarray(actions(k), np.float32)
            self.obs, r, term, trunc, info = self.env.step(a)
            self.last_action = a
            total += float(r)
            self.t += 1
            if self.t % self.every == 0:
                self._emit()
                if realtime:
                    lag = wall + (self.t - start) / CONTROL_HZ - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
            if term or trunc:
                self.done = True
                break
        out = {"ok": True, "t": self.t, "steps": self.t - start, "reward": total,
               "done": self.done, "obs": self.obs.tolist(), "pose": self.pose(),
               "root": self.env.root_position().tolist()}
        if "brain_spikes" in info:
            out["spikes"] = int(info["brain_spikes"])
        return out

    def step(self, req: dict) -> dict:
        if self.done:
            raise ValueError("episode over: reset first")
        realtime = bool(req.get("realtime", self.realtime))
        if req.get("brain"):
            def actions(k):
                return self.action_from(req)        # the brain sees each new observation
        else:
            a = self.action_from(req)
            def actions(k):                         # noqa: E306
                return a
        out = self._run(actions, req.get("steps", 1), realtime)
        out["action"] = self.last_action.tolist()
        return out

    def gait(self, req: dict) -> dict:
        if self.done:
            raise ValueError("episode over: reset first")
        g = TripodGait(CONTROL_HZ, float(req.get("stride_hz", 2.0)),
                       stride=float(req.get("stride", 0.3)), lift=float(req.get("lift", 0.3)),
                       adhesion=float(req.get("adhesion", 0.0)), rest=self.rest)
        start = self.t
        out = self._run(lambda k: g(start + k), req.get("steps", 500),
                        bool(req.get("realtime", self.realtime)))
        out["gait"] = {"stride_hz": g.stride_hz, "stride": g.stride, "lift": g.lift}
        return out

    # --- kinematics -------------------------------------------------------------------------

    def set_pose(self, req: dict) -> dict:
        physics = self.env.physics
        m = physics.model
        if req.get("qpos") is not None:
            q = np.asarray(req["qpos"], dtype=np.float64)
            if q.shape != (m.nq,):
                raise ValueError(f"qpos must have {m.nq} entries")
            physics.data.qpos[:] = q
        for name, angle in (req.get("joints") or {}).items():
            j = m.name2id(name if "/" in name else f"walker/{name}", "joint")
            physics.data.qpos[int(m.jnt_qposadr[j])] = float(angle)
        physics.forward()
        return {"ok": True, "t": self.t, "pose": self._emit(),
                "qpos": np.asarray(physics.data.qpos).tolist()}

    def replay(self, req: dict) -> dict:
        from neurofly_training.body.kinematics import gait_qpos, real_walking_qpos
        source = req.get("source", "gait")
        realtime = bool(req.get("realtime", self.realtime))
        if source == "gait":
            n = int(req.get("steps", 500))
            g = TripodGait(CONTROL_HZ, float(req.get("stride_hz", 2.0)))
            q = gait_qpos(self.env.physics, g, n, amplitude=float(req.get("amplitude", 1.0)))
            fps = CONTROL_HZ
        elif source == "real":
            from neurofly_training.cli.body_replay import find_walking_dataset, longest_trajectory
            path = req.get("path") or find_walking_dataset()
            if not path:
                raise ValueError("no walking dataset: run `neurofly body-replay --download`")
            index = req.get("index")
            index = longest_trajectory(path) if index is None else int(index)
            q, fps = real_walking_qpos(path, index)
            if req.get("steps"):
                q = q[: int(req["steps"])]
        else:
            raise ValueError("source must be gait or real")
        physics = self.env.physics
        shown, wall = 0, time.perf_counter()
        for k in range(0, len(q), self.every):
            physics.data.qpos[:] = q[k][: physics.model.nq]
            physics.forward()
            self._emit()
            shown += 1
            if realtime:
                lag = wall + k / fps - time.perf_counter()
                if lag > 0:
                    time.sleep(lag)
        return {"ok": True, "source": source, "frames": shown, "steps": int(len(q)),
                "fps": fps / self.every, "pose": self.pose()}

    # --- looking ----------------------------------------------------------------------------

    def frame(self, req: dict) -> dict:
        import imageio
        cam = int(req.get("camera", self.camera_id))
        w, h = int(req.get("width", self.width)), int(req.get("height", self.height))
        rgb = self.env.physics.render(camera_id=cam, width=w, height=h)
        buf = io.BytesIO()
        imageio.imwrite(buf, rgb, format="png")
        return {"ok": True, "width": w, "height": h, "format": "png",
                "png": base64.b64encode(buf.getvalue()).decode()}

    def reset(self, req: dict) -> dict:
        task = req.get("task")
        seed = req.get("seed", self.seed)
        if task is not None and task != self.task:
            from neurofly_training.body.tasks import TASKS
            if task not in TASKS:
                raise ValueError(f"unknown task {task!r}; one of {sorted(TASKS)}")
            self.env.close()
            self.env = self._make_env(task, int(seed))
            self.task = task
            self.recorder = PoseRecorder(lambda: self.env.physics, fps=CONTROL_HZ / self.every)
        self.obs, _ = self.env.reset(seed=int(seed))
        self.rest = rest_action(self.env.physics.model)
        self.t, self.done = 0, False
        self.last_action = self.rest.copy()
        if self.brain is not None:
            self.brain.reset()
        return {"ok": True, "task": self.task, "t": 0, "obs": self.obs.tolist(),
                "pose": self._emit()}

    def handle(self, req: dict) -> dict:
        try:
            op = req.get("op")
            with self.lock:
                if op in ("hello", "info"):
                    return self.info()
                if op == "reset":
                    return self.reset(req)
                if op in ("step", "act"):
                    return self.step(req)
                if op == "gait":
                    return self.gait(req)
                if op == "set_pose":
                    return self.set_pose(req)
                if op == "replay":
                    return self.replay(req)
                if op == "observe":
                    return {"ok": True, "t": self.t, "done": self.done, "obs": self.obs.tolist(),
                            "pose": self.pose(), "root": self.env.root_position().tolist(),
                            "action": self.last_action.tolist()}
                if op == "frame":
                    return self.frame(req)
                if op == "ping":
                    return {"ok": True, "pong": True, "t": self.t}
                if op == "poses":
                    return {"ok": True, "on": bool(req.get("on", True))}
                if op == "close":
                    return {"ok": True, "bye": True}
                return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # the protocol must survive bad input
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def close(self) -> None:
        self.env.close()


def serve_stdio(session: BodySession) -> None:
    """JSON lines on stdin and stdout. Poses of every rendered frame of a step come back
    inside its response as ``poses`` (there is no channel to push them on)."""
    out = sys.stdout
    buffer: list = []
    session.listeners.append(lambda t, row: buffer.append(row))
    print(json.dumps({"ready": True, **session.info()}), file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), file=out, flush=True)
            continue
        buffer.clear()
        resp = session.handle(req)
        if buffer and req.get("poses", True):
            resp["poses"] = list(buffer)
        print(json.dumps(resp), file=out, flush=True)
        if resp.get("bye"):
            break
    session.close()


def serve_ws(session: BodySession, host: str = "127.0.0.1", port: int = 8767, *,
             token: str | None = None, origins=None, ready=None) -> None:
    """The same protocol over a WebSocket; one body shared by every client. Requests run
    in a worker thread, one at a time, and every rendered pose is pushed to every
    connected client as ``{"t": n, "pose": [...]}`` while the request runs (a viewer sees
    the body move while a program drives it). ``ready``: called with the bound port."""
    import asyncio
    from urllib.parse import parse_qs, urlparse

    import websockets

    clients: set = set()
    muted: set = set()
    loop_holder: dict = {}

    def on_pose(t, row):
        loop = loop_holder.get("loop")
        if loop is None:
            return
        msg = json.dumps({"t": t, "pose": row})
        loop.call_soon_threadsafe(lambda: [asyncio.ensure_future(_push(ws, msg))
                                           for ws in list(clients) if ws not in muted])
    session.listeners.append(on_pose)

    async def _push(ws, msg):
        try:
            await ws.send(msg)
        except Exception:
            clients.discard(ws)

    def query_token(ws) -> str | None:
        req = getattr(ws, "request", None)
        path = getattr(req, "path", None) or getattr(ws, "path", "") or ""
        return (parse_qs(urlparse(path).query).get("token") or [None])[0]

    async def handler(ws):
        try:
            given = query_token(ws)
            if token is not None and given != token:
                first = {} if given is not None else json.loads(await ws.recv())
                if not (first.get("op") == "hello" and first.get("token") == token):
                    await ws.send(json.dumps({"ok": False, "error": "unauthorised: pass ?token= "
                                              "on the URL or send {\"op\": \"hello\", \"token\": "
                                              "...} first"}))
                    return
            await ws.send(json.dumps({"ready": True, **session.info()}))
            clients.add(ws)
            async for msg in ws:
                try:
                    req = json.loads(msg)
                except json.JSONDecodeError as e:
                    await ws.send(json.dumps({"ok": False, "error": f"bad JSON: {e}"}))
                    continue
                if req.get("op") == "poses":
                    (muted.discard if req.get("on", True) else muted.add)(ws)
                resp = await asyncio.to_thread(session.handle, req)
                await ws.send(json.dumps(resp))
                if resp.get("bye"):
                    break
        except (websockets.ConnectionClosed, json.JSONDecodeError):
            pass
        finally:
            clients.discard(ws)
            muted.discard(ws)

    async def main():
        loop_holder["loop"] = asyncio.get_running_loop()
        kw = {"max_size": 64 * 1024 * 1024}
        if origins:
            kw["origins"] = list(origins)
        async with websockets.serve(handler, host, port, **kw) as server:
            bound = next(iter(server.sockets)).getsockname()[1]
            print(f"neurofly body listening on ws://{host}:{bound} (task {session.task}"
                  + (f", brain {session.info()['brain']}" if session.brain else "")
                  + (", token required" if token else "") + ")", file=sys.stderr, flush=True)
            if ready is not None:
                ready(bound)
            await asyncio.Future()

    asyncio.run(main())
