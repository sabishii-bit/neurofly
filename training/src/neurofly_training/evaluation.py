"""Score an artifact offline against recordings.

For every recording the artifact's model watches the video (and hears the
sound, if both have it) and its policy proposes controls on every frame. The
report compares those with what the human did: precision, recall and F1 per
key and button, correlation and mean absolute error per mouse axis and
gamepad axis. With a ``Task`` it also reports the reward the policy would
have earned, frame by frame, on that footage.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from neurofly_core.artifact import load_model
from neurofly_core.io.audio import AudioFile, SilentAudio
from neurofly_core.io.video import VideoFile
from neurofly_training.pc.imitation import evaluate as score_policy
from neurofly_training.pc.task import load_task


def load_recording(path: str):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    actions = np.load(os.path.join(path, "actions.npy"))
    return os.path.join(path, "video.mp4"), meta, actions


def run_recording(model, recording: str, *, task=None, max_frames: int | None = None,
                  progress: bool = False, detector=None) -> dict:
    video_path, meta, actions = load_recording(recording)
    video = VideoFile(video_path)
    if model.detection is not None and detector is None:
        raise ValueError("this artifact has a detection encoder: pass --detect")
    if detector is not None:
        from neurofly_training.pc.detect import cached
        detector = cached(detector, os.path.abspath(video_path))
        detector.reset()
    from neurofly_training.envs import sense_sources
    senses = sense_sources(model, task)
    fps = video.fps or float(meta.get("fps", 10.0))
    audio = None
    if model.audition is not None:
        wav = os.path.join(recording, "audio.wav")
        audio = AudioFile(wav, fps=fps) if os.path.exists(wav) else SilentAudio(fps=fps)
    model.reset()
    frame = video.reset()
    chunk = audio.reset() if audio is not None else None
    feats, acts, rewards, t = [], [], [], 0
    try:
        while frame is not None and (max_frames is None or t < max_frames):
            dets = detector.detect(frame) if detector is not None else None
            info = {"t": t}
            f = model.observe(frame, chunk, detections=dets,
                              **{kw: fn(frame, chunk, info, dets) for kw, fn in senses.items()})
            feats.append(f)
            if model.policy is not None:
                a = model.act(f)
                acts.append(a)
                if task is not None:
                    rewards.append(float(task.reward(frame, chunk, model.layout.decode(a), info)))
            t += 1
            if progress and t % 100 == 0:
                print(f"  {t} frames")
            frame = video.read()
            if audio is not None:
                chunk = audio.read()
    finally:
        video.close()
        if audio is not None:
            audio.close()
        if detector is not None:
            detector.close()
    n = min(len(feats), len(actions))
    out = {"recording": recording, "frames": n, "fps": fps}
    if acts:
        X, Y = np.stack(feats[:n]), np.asarray(actions[:n], np.float32)
        out["controls"] = score_policy(model.policy, X, Y)
        P = np.stack(acts[:n])
        mask = model.layout.binary_mask
        if mask.any():
            out["agreement"] = float(np.mean((P[:, mask] > 0) == (Y[:, mask] > 0)))
    if rewards:
        r = np.asarray(rewards)
        out["reward"] = {"mean": float(r.mean()), "sum": float(r.sum()), "min": float(r.min()),
                         "max": float(r.max())}
    return out


def evaluate_artifact(artifact: str, recordings: list[str], *, reward: str | None = None,
                      max_frames: int | None = None, device: str = "cpu",
                      progress: bool = False, detect=None) -> dict:
    model = load_model(artifact, device=device)
    task = load_task(reward) if reward else None
    detector = None
    if model.detection is not None:
        from neurofly_training.pc.detect import make_detector
        detector = make_detector(detect or model.config.meta.get("detect"), device=device)
    per = [run_recording(model, r, task=task, max_frames=max_frames, progress=progress,
                         detector=detector) for r in recordings]
    report = {"artifact": os.path.abspath(artifact), "name": model.config.name,
              "evaluated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "has_policy": model.policy is not None, "recordings": per}
    if model.policy is not None and per:
        ctrls = {}
        for name in model.layout.names:
            vals = [p["controls"][name] for p in per if "controls" in p]
            if vals:
                ctrls[name] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        report["controls"] = ctrls
        report["agreement"] = float(np.mean([p["agreement"] for p in per if "agreement" in p]))
        if all("reward" in p for p in per):
            report["reward_mean"] = float(np.mean([p["reward"]["mean"] for p in per]))
    return report


def format_report(report: dict) -> str:
    lines = [f"{report['name']} ({report['artifact']})",
             f"  recordings: {len(report['recordings'])}, frames: "
             f"{sum(p['frames'] for p in report['recordings'])}"]
    if not report.get("has_policy"):
        lines.append("  no policy in this artifact: nothing to score")
        return "\n".join(lines)
    lines.append(f"  agreement on keys and buttons: {report.get('agreement', float('nan')):.1%}")
    if "reward_mean" in report:
        lines.append(f"  mean reward per frame: {report['reward_mean']:+.3f}")
    for name, m in report.get("controls", {}).items():
        if "f1" in m:
            lines.append(f"  {name:14s} held {m['rate']:5.1%}   precision {m['precision']:.2f}  "
                         f"recall {m['recall']:.2f}  f1 {m['f1']:.2f}")
        else:
            lines.append(f"  {name:14s} mean |x| {m['rate']:.2f}   corr {m['corr']:+.2f}  "
                         f"mae {m['mae']:.2f}")
    return "\n".join(lines)
