"""Reading things off the screen for reward: template matching and numbers.

    TemplatePresence(template_png, region)   reward = how well the template matches
    read_number(frame, region)               an integer read by OCR (needs pytesseract)
    NumberOnScreen(region)                   reward = change of that number since the last step

Template matching is normalised cross-correlation on grayscale, in numpy, over a region
of the frame given as fractions (left, top, width, height). OCR uses the ``pytesseract``
package and a Tesseract install when present; without them ``read_number`` returns None.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from neurofly_training.pc.task import Task


def _region(frame: np.ndarray, region) -> np.ndarray:
    if region is None:
        return frame
    h, w = frame.shape[:2]
    x0, y0, rw, rh = region
    x0, y0 = int(x0 * w), int(y0 * h)
    x1, y1 = max(x0 + 1, int((x0 / w + rw) * w)), max(y0 + 1, int((y0 / h + rh) * h))
    return frame[y0:y1, x0:x1]


def _gray(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32).mean(axis=2)
    return a.astype(np.float32)


def match_template(frame: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Best normalised cross-correlation score in [-1, 1] and its (row, col) in ``frame``."""
    f, t = _gray(frame), _gray(template)
    th, tw = t.shape
    fh, fw = f.shape
    if th > fh or tw > fw:
        return -1.0, (0, 0)
    t_mean = t.mean()
    t = t - t_mean
    tn = np.sqrt((t * t).sum())
    # sliding windows via stride tricks; frames are small (320x240), templates smaller
    win = np.lib.stride_tricks.sliding_window_view(f, (th, tw))
    means = win.mean(axis=(2, 3), keepdims=True)
    centred = win - means
    if tn < 1e-3:
        # a flat template: match on brightness and flatness instead of correlation
        flat = np.sqrt((centred * centred).mean(axis=(2, 3)))
        score = 1.0 - (np.abs(means[..., 0, 0] - t_mean) + flat) / 255.0
    else:
        num = (centred * t).sum(axis=(2, 3))
        den = np.sqrt((centred * centred).sum(axis=(2, 3))) * tn + 1e-6
        score = num / den
    idx = np.unravel_index(int(score.argmax()), score.shape)
    return float(score[idx]), (int(idx[0]), int(idx[1]))


def load_template(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class TemplatePresence(Task):
    """Reward = the template's best match score in ``region``; the episode ends when the
    match falls below ``end_below`` (an icon disappears) if that is set."""

    def __init__(self, template, region=None, end_below: float | None = None, scale: float = 1.0):
        self.template = (load_template(template) if isinstance(template, str)
                         else np.asarray(template))
        self.region, self.end_below, self.scale = region, end_below, scale

    def score(self, frame) -> float:
        return match_template(_region(frame, self.region), self.template)[0]

    def reward(self, frame, audio, state, info) -> float:
        s = self.score(frame)
        info["template_score"] = s
        return s * self.scale

    def done(self, frame, audio, info) -> bool:
        if self.end_below is None:
            return False
        return info.get("template_score", self.score(frame)) < self.end_below


def read_number(frame: np.ndarray, region=None, scale: int = 3) -> int | None:
    """An integer read by OCR from ``region`` of the frame, or None if unreadable or
    pytesseract is not installed."""
    try:
        import pytesseract
    except ImportError:
        return None
    patch = _gray(_region(frame, region))
    img = Image.fromarray(patch.astype(np.uint8))
    img = img.resize((patch.shape[1] * scale, patch.shape[0] * scale))
    text = pytesseract.image_to_string(img, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    digits = "".join(c for c in text if c.isdigit())
    return int(digits) if digits else None


class NumberOnScreen(Task):
    """Reward = change of a number on screen since the last step (a score, a counter),
    scaled; unreadable frames give 0. Needs pytesseract and Tesseract."""

    def __init__(self, region, scale: float = 1.0, end_below: int | None = None):
        self.region, self.scale, self.end_below = region, scale, end_below
        self._last = None

    def reset(self, controls, video, audio) -> None:
        self._last = None

    def reward(self, frame, audio, state, info) -> float:
        n = read_number(frame, self.region)
        info["number"] = n
        if n is None or self._last is None:
            self._last = n if n is not None else self._last
            return 0.0
        r = (n - self._last) * self.scale
        self._last = n
        return float(r)

    def done(self, frame, audio, info) -> bool:
        n = info.get("number")
        return self.end_below is not None and n is not None and n < self.end_below
