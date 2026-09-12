"""Environment factories and the shared command-line options.

Two kinds of task share one ``make_env``:

* **Body tasks** (``forward``, ``ball``): the MuJoCo fly. With ``brain='none'``
  the agent controls the 59 actuators from proprioception; with a brain,
  ``BrainInLoopEnv`` puts the spiking connectome between the body's sensors
  and the agent.
* **PC tasks** (``pc`` for the live screen, or a video file): a
  ``neurofly_core.Model`` built from the connectome, wrapped in ``PCEnv`` with
  video and audio sources, controls and a ``Task``. These always need a brain.
"""
from __future__ import annotations

import numpy as np

import os

import gymnasium as gym

from neurofly_core.controls import ControlLayout, parse_names
from neurofly_core.io.audio import AudioFile, SilentAudio, make_audio
from neurofly_core.io.controls import NullControls, make_controls
from neurofly_core.io.video import VIDEO_EXTS, ScreenCapture, VideoFile
from neurofly_training.body.env import BrainInLoopEnv  # noqa: F401  (re-exported)
from neurofly_training.body.gym_wrapper import DmEnvToGym
from neurofly_training.body.tasks import TASKS
from neurofly_training.body.tasks import make_env as make_dm_env
from neurofly_training.build import build_model
from neurofly_training.data.connectome import Connectome
from neurofly_training.pc.env import PCEnv
from neurofly_training.pc.task import load_task

# the repository root is four levels above this file (training/src/neurofly_training/envs.py)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_DATA_DIR = os.environ.get("NEUROFLY_DATA", os.path.join(REPO_ROOT, "data", "malecns"))

# Which env-construction arguments train scripts must store so watch, play and
# export can rebuild the same environment and model.
BODY_ARGS = ["task", "brain", "subset", "dt", "readout", "include_proprio", "plasticity",
             "brain_gain", "encoder_gain", "synthetic_n"]
PC_ARGS = ["fps", "window", "region", "monitor", "audio", "keys", "buttons", "mouse",
           "mouse_speed", "scroll", "pad_buttons", "axes", "reward", "max_steps", "dry_run",
           "brain_ms", "retina_mode",
           "retina_gain", "retina_temporal", "audio_gain", "include_frame", "include_audio",
           "detect", "detection_grid", "detection_gain", "include_detections",
           "odours", "odour_gain", "odour_adapt", "include_odours",
           "dopamine_punish"]
ENV_ARGS = BODY_ARGS + PC_ARGS


def is_pc_task(task) -> bool:
    return task == "pc" or str(task).lower().endswith(VIDEO_EXTS)


def list_tasks() -> list[str]:
    return sorted(TASKS) + ["pc", "<video file>"]


def default_subset(task: str) -> str:
    """Nerve cord for walking; central brain (no optic lobes, no cord) for the PC."""
    return "central" if is_pc_task(task) else "vnc"


def default_readout(task: str) -> str:
    return "descending" if is_pc_task(task) else "motor+descending"


def make_body_env(task: str = "forward", seed: int = 0, render_mode: str | None = None,
                  camera_id: int = 1, width: int = 640, height: int = 480,
                  **task_kwargs) -> DmEnvToGym:
    return DmEnvToGym(make_dm_env(task, seed=seed, **task_kwargs), camera_id=camera_id,
                      width=width, height=height, render_mode=render_mode)


def load_connectome(brain: str = "malecns", subset: str = "vnc",
                    data_dir: str | None = None, synthetic_n: int = 3000) -> Connectome:
    """The connectome must not depend on the env seed: parallel workers, playback
    and export all need identical neuron populations."""
    if brain == "synthetic":
        return Connectome.synthetic(n=synthetic_n, seed=0).subset(subset)
    if brain == "toy":
        return Connectome.toy(n=synthetic_n, seed=0).subset(subset)
    if brain == "malecns":
        cx = Connectome.load(data_dir or DEFAULT_DATA_DIR, verbose=False)
        return cx.subset(subset)
    raise ValueError(f"brain must be 'none', 'malecns', 'synthetic' or 'toy', got {brain!r}")


def parse_region(region):
    if region is None or region == "":
        return None
    if isinstance(region, str):
        region = region.split(",")
    return tuple(int(v) for v in region)


def make_layout(keys=None, buttons=None, mouse: bool = False, scroll: bool = False,
                mouse_speed: float = 50.0, pad_buttons=None, axes=None) -> ControlLayout:
    return ControlLayout(keys=parse_names(keys), buttons=parse_names(buttons), mouse=bool(mouse),
                         scroll=bool(scroll), mouse_speed=float(mouse_speed),
                         pad_buttons=parse_names(pad_buttons), axes=parse_names(axes))


def odour_channels(odours, detect_classes=None) -> list[str]:
    """``--odours``: None -> no olfaction; 'detections' -> one channel per detected class;
    otherwise comma-separated channel names (or a list)."""
    if odours is None or str(odours).lower() in ("", "none"):
        return []
    if isinstance(odours, str) and odours.lower() == "detections":
        if not detect_classes:
            raise ValueError("--odours detections needs --detect")
        return list(detect_classes)
    return parse_names(odours)


def odour_source(model, task=None):
    """A callable ``(frame, chunk, info, detections) -> odours or None`` for a model with
    an olfaction encoder: the detection classes' presence when it was built with
    ``--odours detections``, else what the Task's ``odours`` hook returns."""
    if model.olfaction is None:
        return None
    if model.config.meta.get("odours") == "detections":
        channels = list(model.olfaction.channels)

        def from_detections(frame, chunk, info, detections):
            v = np.zeros(len(channels), np.float32)
            for d in detections or []:
                c = d.get("label") if isinstance(d, dict) else None
                if c is None and isinstance(d, dict):
                    c = d["class"] if isinstance(d["class"], str) else \
                        (model.detection.classes[int(d["class"])] if model.detection else None)
                if c in channels:
                    j = channels.index(c)
                    v[j] = max(v[j], float(d.get("score", 1.0)))
            return v

        return from_detections
    if task is None:
        return None
    return lambda frame, chunk, info, detections: task.odours(frame, chunk, info)


def parse_grid(grid, default=(6, 8)) -> tuple[int, int]:
    if grid is None or grid == "":
        return tuple(default)
    if isinstance(grid, str):
        grid = grid.split(",")
    return int(grid[0]), int(grid[1])


def wants_audio(audio) -> bool:
    return audio is not None and str(audio).lower() != "none"


def make_pc_model(task: str = "pc", brain: str = "malecns", subset: str | None = None,
                  dt: float = 0.5, readout: str | None = None, plasticity: bool = False,
                  brain_gain: float = 1.0, synthetic_n: int = 3000, data_dir: str | None = None,
                  device: str = "cpu", audio=None, layout: ControlLayout | None = None,
                  keys=None, buttons=None, mouse: bool = False, mouse_speed: float = 50.0,
                  scroll: bool = False, pad_buttons=None, axes=None, brain_ms: float = 10.0,
                  retina_mode: str = "auto",
                  retina_gain: float = 15.0, retina_temporal: float = 0.0,
                  audio_gain: float = 15.0, include_frame: bool = False,
                  include_audio: bool = False, dopamine_punish: float = 0.0,
                  detect=None, detection_grid=(6, 8), detection_gain: float = 15.0,
                  include_detections: bool = False, odours=None, odour_gain: float = 15.0,
                  odour_adapt: float = 0.0, include_odours: bool = False,
                  sample_rate: int = 16000, name: str | None = None, policy=None,
                  **_ignored):
    """The ``neurofly_core.Model`` a PC task uses, without any sources. This is what
    ``export`` saves; the same call inside ``make_pc_env`` guarantees the exported model
    matches the trained one."""
    if brain == "none":
        raise ValueError("PC tasks need a brain ('malecns' or 'synthetic'): "
                         "the controls are read out of neurons")
    layout = layout or make_layout(keys, buttons, mouse, scroll, mouse_speed, pad_buttons, axes)
    cx = load_connectome(brain, subset=subset or default_subset(task), data_dir=data_dir,
                         synthetic_n=synthetic_n)
    from neurofly_training.pc.detect import classes_for
    classes = list(detect) if isinstance(detect, (list, tuple)) else classes_for(detect)
    channels = odour_channels(odours, classes)
    return build_model(cx, layout, readout=readout or default_readout(task), dt=dt,
                       brain_ms=brain_ms, brain_gain=brain_gain, retina_mode=retina_mode,
                       retina_gain=retina_gain, retina_temporal=retina_temporal,
                       audio=wants_audio(audio), sample_rate=sample_rate,
                       audio_gain=audio_gain, include_frame=include_frame,
                       include_audio=include_audio, plasticity=plasticity,
                       detect_classes=classes, detection_grid=parse_grid(detection_grid),
                       detection_gain=detection_gain, include_detections=include_detections,
                       odour_channels=channels, odour_gain=odour_gain, odour_adapt=odour_adapt,
                       include_odours=include_odours,
                       dopamine_punish=dopamine_punish, policy=policy, name=name,
                       meta={"brain": brain, "subset": subset or default_subset(task),
                             "detect": detect if isinstance(detect, str) else None,
                             "odours": "detections" if str(odours).lower() == "detections"
                             else (",".join(channels) if channels else None)},
                       device=device)


def make_pc_env(task: str = "pc", seed: int = 0, fps: float = 10.0, window: str | None = None,
                region=None, monitor: int = 1, audio=None, reward: str | None = None,
                max_steps: int | None = None, dry_run: bool = False, video_source=None,
                audio_source=None, controls=None, task_obj=None, model=None, detector=None,
                detect=None, device: str = "cpu", **model_kwargs) -> PCEnv:
    """The live screen (``task='pc'``) or a video file, with the brain in the loop.

    Pass ``video_source`` / ``audio_source`` / ``controls`` / ``task_obj`` / ``model``
    objects to bypass the command-line style arguments (tests, custom projects).
    """
    if video_source is None:
        if task == "pc":
            video_source = ScreenCapture(window=window, region=parse_region(region),
                                         monitor=monitor, fps=fps)
        else:
            video_source = VideoFile(task)
            fps = video_source.fps or fps
    sample_rate = 16000
    if audio_source is None and wants_audio(audio):
        if video_source.live:
            audio_source = make_audio(audio, fps=fps, sample_rate=sample_rate)
        else:
            wav = os.path.join(os.path.dirname(os.path.abspath(task)), "audio.wav")
            audio_source = (AudioFile(wav, fps=fps) if os.path.exists(wav)
                            else SilentAudio(fps=fps, sample_rate=sample_rate))
    if audio_source is not None:
        sample_rate = audio_source.sample_rate
    if model is None:
        model = make_pc_model(task, audio=audio if audio_source is None else "yes",
                              sample_rate=sample_rate, detect=detect, device=device,
                              **model_kwargs)
    if detector is None and model.detection is not None:
        from neurofly_training.pc.detect import cached, make_detector
        if detect:
            detector = make_detector(detect, device=device)
        elif model.config.meta.get("detect"):
            detector = make_detector(model.config.meta["detect"], device=device)
        else:
            raise ValueError("the model has a detection encoder; pass detect= (a detector "
                             "spec) or detector= (a Detector) so it gets detections")
        if not video_source.live:
            detector = cached(detector, os.path.abspath(task))
    if detector is not None and list(detector.classes) != list(model.detection.classes
                                                                 if model.detection else []):
        raise ValueError(f"the detector's classes {detector.classes} differ from the model's "
                         f"{model.detection.classes if model.detection else []}")
    if controls is None:
        if not video_source.live:
            controls = NullControls()
        else:
            controls = make_controls("log" if dry_run else "pc", model.layout)
    task_obj = task_obj if task_obj is not None else load_task(reward)
    return PCEnv(model, video=video_source, audio=audio_source, controls=controls, task=task_obj,
                 fps=fps, max_steps=max_steps, detector=detector,
                 odours=odour_source(model, task_obj))


def make_env(task: str = "forward", seed: int = 0, brain: str = "none", subset: str | None = None,
             dt: float = 0.5, readout: str | None = None, include_proprio: bool = False,
             plasticity: bool = False, brain_gain: float = 1.0, encoder_gain: float = 12.0,
             synthetic_n: int = 3000, data_dir: str | None = None, device: str = "cpu",
             render_mode: str | None = None, camera_id: int = 1, width: int = 640,
             height: int = 480, **kwargs) -> gym.Env:
    """Build a body-only env (brain='none'), a brain-in-the-loop body env, or a PC env.
    Extra keyword arguments go to ``make_pc_env`` for PC tasks and to the body task
    otherwise."""
    if is_pc_task(task):
        return make_pc_env(task, seed=seed, brain=brain, subset=subset, dt=dt, readout=readout,
                           plasticity=plasticity, brain_gain=brain_gain,
                           synthetic_n=synthetic_n, data_dir=data_dir, device=device, **kwargs)
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {list_tasks()}")
    task_kwargs = {k: v for k, v in kwargs.items() if k not in PC_ARGS}
    body = make_body_env(task, seed=seed, render_mode=render_mode, camera_id=camera_id,
                         width=width, height=height, **task_kwargs)
    if brain == "none":
        return body
    cx = load_connectome(brain, subset=subset or default_subset(task), data_dir=data_dir,
                         synthetic_n=synthetic_n)
    return BrainInLoopEnv(body, cx, readout=readout or default_readout(task),
                          include_proprio=include_proprio, dt=dt, brain_gain=brain_gain,
                          encoder_gain=encoder_gain, plasticity=plasticity, device=device,
                          seed=seed)


def resolve_env_args(args) -> dict:
    """Fill in task-dependent defaults on an argparse namespace and return the
    ENV_ARGS dict to store in a run's config.json."""
    if args.subset is None:
        args.subset = default_subset(args.task)
    if getattr(args, "readout", None) is None:
        args.readout = default_readout(args.task)
    if is_pc_task(args.task) and args.brain == "none":
        raise SystemExit(f"task {args.task!r} needs --brain malecns (or synthetic)")
    if not is_pc_task(args.task) and args.task not in TASKS:
        raise SystemExit(f"unknown task {args.task!r}; choose from {list_tasks()}")
    return {k: getattr(args, k) for k in ENV_ARGS if hasattr(args, k)}


def add_env_args(p, *, brain_default: str = "none",
                 brain_choices=("none", "malecns", "synthetic", "toy")):
    """The environment options shared by the train scripts."""
    p.add_argument("--task", default="forward",
                   help=f"body task, 'pc' (the live screen) or a video file: {list_tasks()}")
    p.add_argument("--brain", default=brain_default, choices=list(brain_choices))
    p.add_argument("--subset", default=None,
                   help="connectome subset; default vnc for body tasks, central for the PC")
    p.add_argument("--dt", type=float, default=0.5, help="brain step, ms")
    p.add_argument("--readout", default=None,
                   help="motor, descending, ascending, cbmotor, visual joined by '+'; "
                        "default motor+descending (body) or descending (PC)")
    p.add_argument("--include-proprio", action="store_true",
                   help="body: also give the policy the raw body observation")
    p.add_argument("--plasticity", action="store_true", help="reward acts as dopamine")
    p.add_argument("--brain-gain", type=float, default=1.0)
    p.add_argument("--encoder-gain", type=float, default=12.0, help="body: sensory drive, mV")
    p.add_argument("--synthetic-n", type=int, default=3000,
                   help="neurons in the synthetic or toy brain")
    g = p.add_argument_group("PC tasks: what the brain sees and may touch")
    g.add_argument("--fps", type=float, default=10.0, help="steps per second (live sources)")
    g.add_argument("--window", default=None, help="capture the window whose title contains this")
    g.add_argument("--region", default=None, help="capture left,top,width,height instead")
    g.add_argument("--monitor", type=int, default=1, help="capture a whole monitor (fallback)")
    g.add_argument("--audio", default=None,
                   help="sound in: 'loopback' (what the PC plays), a capture device name or "
                        "index, or none (default)")
    g.add_argument("--keys", default=None, help="keys the brain may hold, e.g. w,a,s,d,space")
    g.add_argument("--buttons", default=None, help="mouse buttons it may hold: left,right,middle")
    g.add_argument("--mouse", action="store_true", help="it may move the mouse")
    g.add_argument("--mouse-speed", type=float, default=50.0, help="pixels per step at full tilt")
    g.add_argument("--scroll", action="store_true", help="it may scroll")
    g.add_argument("--pad-buttons", default=None,
                   help="gamepad buttons it may hold: a,b,x,y,lb,rb,start,back,ls,rs,dup,ddown,"
                        "dleft,dright (a virtual controller through ViGEm)")
    g.add_argument("--axes", default=None, help="gamepad axes it may move: lx,ly,rx,ry,lt,rt")
    g.add_argument("--reward", default=None,
                   help="Task giving reward and episode ends: module:Name or file.py:Name")
    g.add_argument("--max-steps", type=int, default=None, help="episode length cap")
    g.add_argument("--dry-run", action="store_true",
                   help="print the controls instead of sending them to the PC")
    g.add_argument("--brain-ms", type=float, default=10.0, help="brain time per step")
    g.add_argument("--retina-mode", default="auto", choices=["auto", "hex", "projection"])
    g.add_argument("--retina-gain", type=float, default=15.0, help="drive at full brightness, mV")
    g.add_argument("--retina-temporal", type=float, default=0.0,
                   help="0 = respond to brightness, 1 = only to brightness change")
    g.add_argument("--audio-gain", type=float, default=15.0, help="drive at full loudness, mV")
    g.add_argument("--include-frame", action="store_true",
                   help="also give the policy a 12x16 luminance grid of the frame")
    g.add_argument("--include-audio", action="store_true",
                   help="also give the policy the audio band levels")
    g.add_argument("--dopamine-punish", type=float, default=0.0,
                   help="mV of drive on PPL1 dopamine neurons while reward is negative")
    g.add_argument("--detect", default=None, metavar="SPEC",
                   help="an object detector feeding a detection encoder: 'owl2:enemy,health "
                        "pack' (open vocabulary, no training), a directory from `neurofly "
                        "detect-train`, 'onnx:DIR' or 'yolo:weights.pt'")
    g.add_argument("--detection-grid", default="6,8", help="cells per class: rows,cols")
    g.add_argument("--detection-gain", type=float, default=15.0,
                   help="drive at full coverage of a cell, mV")
    g.add_argument("--include-detections", action="store_true",
                   help="also give the policy the detection grids")
    g.add_argument("--odours", default=None, metavar="CHANNELS",
                   help="odour channels onto olfactory receptor neurons: names your Task's "
                        "odours() fills (e.g. health,danger), or 'detections' for one channel "
                        "per detected class")
    g.add_argument("--odour-gain", type=float, default=15.0, help="drive at channel value 1, mV")
    g.add_argument("--odour-adapt", type=float, default=0.0,
                   help="0 to 1: how much the drive fades while a channel stays constant")
    g.add_argument("--include-odours", action="store_true",
                   help="also give the policy the odour channels")
    return p
