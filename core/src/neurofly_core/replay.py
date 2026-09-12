"""The replay format: neural activity over time, indexed by the connectome's own ids.

A small JSON contract so that a viewer, ours or anyone's, can draw activity on the
brain atlas without knowing whose model produced it:

    {"version": 1, "dataset": "male-cns:v1.0",
     "source": {"kind": "predicted" | "measured" | "synthetic", "name": "...",
                "normalization": "rate / 50 Hz, clipped to [0, 1]"},
     "frames": [{"time": 0.0, "values": [[bodyId, value], ...]}, ...]}

``time`` is seconds from the start and must increase; ``value`` is in [0, 1]; neurons not
listed in a frame are 0 for that frame (frames are complete, not deltas). ``from_activity``
turns the activity file neurofly writes (``--activity-out``) into this, and
``validate_replay`` checks a file the way the workbench does before drawing it.

    neurofly-core replay-export activity.json --out model-output.json
    neurofly-core replay-validate model-output.json --atlas assets/brain-atlas
"""
from __future__ import annotations

import json
import os

import numpy as np

REPLAY_VERSION = 1
MAX_FRAMES = 10_000
MAX_BYTES = 50 * 1024 * 1024
RATE_MAX_HZ = 50.0


def from_activity(activity: dict, *, name: str = "neurofly", kind: str = "predicted",
                  rate_max: float = RATE_MAX_HZ, max_frames: int = MAX_FRAMES) -> dict:
    """A replay from an activity file: each step's spike counts become firing rates
    (counts times the step rate) normalised by ``rate_max`` and clipped to [0, 1]. Needs
    the file's ``ids`` (written by neurofly since the replay format exists)."""
    ids = activity.get("ids")
    if not ids:
        raise ValueError("the activity file has no neuron ids; record it again with a "
                         "current neurofly, or export from an artifact with neurons/ids.bin")
    fps = float(activity.get("fps") or 10.0)
    ids = np.asarray(ids, dtype=np.int64)
    frames = []
    for step in activity["steps"][:max_frames]:
        idx = np.asarray(step["indices"], dtype=np.int64)
        counts = np.asarray(step.get("counts") or np.ones(len(idx)), dtype=np.float64)
        values = np.clip(counts * fps / rate_max, 0.0, 1.0)
        pairs = [[int(ids[i]), round(float(v), 4)] for i, v in zip(idx, values) if v > 0]
        frames.append({"time": round(int(step.get("t", len(frames))) / fps, 6),
                       "values": pairs})
    return {"version": REPLAY_VERSION, "dataset": "male-cns:v1.0",
            "source": {"kind": kind, "name": name,
                       "normalization": f"spike count per step x {fps:g} steps/s, divided by "
                                        f"{rate_max:g} Hz, clipped to [0, 1]"},
            "frames": frames}


def validate_replay(replay: dict, known_ids=None, *, max_frames: int = MAX_FRAMES,
                    n_bytes: int | None = None) -> list[str]:
    """Problems with a replay dict (empty list: fine). ``known_ids``: the atlas or
    artifact ids every value must refer to; ``n_bytes``: the file's size if known."""
    problems: list[str] = []
    if n_bytes is not None and n_bytes > MAX_BYTES:
        problems.append(f"file is {n_bytes / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB")
    if replay.get("version") != REPLAY_VERSION:
        problems.append(f"version must be {REPLAY_VERSION}, got {replay.get('version')!r}")
    src = replay.get("source") or {}
    if src.get("kind") not in ("synthetic", "predicted", "measured"):
        problems.append("source.kind must be synthetic, predicted or measured")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not 2 <= len(frames) <= max_frames:
        problems.append(f"frames must be a list of 2 to {max_frames} entries")
        return problems
    known = None if known_ids is None else set(int(i) for i in np.asarray(known_ids).ravel())
    last = None
    for k, fr in enumerate(frames):
        t = fr.get("time")
        if not isinstance(t, (int, float)):
            problems.append(f"frame {k}: time missing")
            continue
        if k == 0 and t != 0:
            problems.append("the first frame must be at time 0")
        if last is not None and t <= last:
            problems.append(f"frame {k}: time {t} does not increase")
        last = t
        seen = set()
        for pair in fr.get("values", []):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                problems.append(f"frame {k}: a value is not [id, value]")
                break
            i, v = pair
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                problems.append(f"frame {k}: value {v!r} for id {i} is outside [0, 1]")
                break
            if i in seen:
                problems.append(f"frame {k}: id {i} repeated")
                break
            seen.add(i)
            if known is not None and int(i) not in known:
                problems.append(f"frame {k}: id {i} is not in the atlas")
                break
        if len(problems) > 20:
            problems.append("...")
            break
    return problems


def load_ids(path: str) -> np.ndarray:
    """The ids of an atlas directory (``ids.bin``) or an artifact (``neurons/ids.bin``)."""
    for candidate in (os.path.join(path, "ids.bin"), os.path.join(path, "neurons", "ids.bin")):
        if os.path.exists(candidate):
            return np.fromfile(candidate, "<i8")
    raise FileNotFoundError(f"no ids.bin under {path} (an atlas or an artifact directory)")


def frame_at(replay: dict, time_s: float) -> dict | None:
    """The last frame at or before ``time_s`` (None before the first)."""
    frames = replay["frames"]
    lo, hi = 0, len(frames)
    while lo < hi:
        mid = (lo + hi) // 2
        if frames[mid]["time"] <= time_s:
            lo = mid + 1
        else:
            hi = mid
    return frames[lo - 1] if lo else None


def save(replay: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(replay, f)
    return path
