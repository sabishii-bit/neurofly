"""Video in: a live window, screen region or monitor, a video file, or synthetic frames.

Frames are RGB uint8 arrays of shape (height, width, 3). Live sources set
``live = True``: ``read()`` returns what is on the screen right now and the
environment paces itself to a frame rate. File sources return the next frame
and ``None`` at the end.
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".gif")


class VideoSource:
    live: bool = False
    fps: float | None = None

    def reset(self) -> np.ndarray | None:
        return self.read()

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def resize(frame: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """``size`` is (width, height); None keeps the frame as is."""
    if size is None:
        return np.ascontiguousarray(frame)
    w, h = size
    return np.asarray(Image.fromarray(frame).resize((w, h), Image.BOX))


def find_window(title: str) -> tuple[int, int, int, int, str]:
    """Client area (left, top, width, height) and full title of the first visible
    top-level window whose title contains ``title`` (case-insensitive). Windows only."""
    if sys.platform != "win32":
        raise RuntimeError("window lookup by title is Windows-only; pass a screen region instead")
    import ctypes
    import ctypes.wintypes as wt

    user32 = ctypes.windll.user32
    try:  # physical pixels, so that coordinates match what the capture sees
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    found: list[tuple[int, str]] = []
    proc_t = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if title.lower() in buf.value.lower():
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(proc_t(cb), 0)
    if not found:
        raise RuntimeError(f"no visible window with {title!r} in its title")
    hwnd, name = found[0]
    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return int(pt.x), int(pt.y), int(rect.right - rect.left), int(rect.bottom - rect.top), name


class ScreenCapture(VideoSource):
    """Grab a window (by title), a screen region (left, top, width, height) or a whole monitor."""
    live = True

    def __init__(self, window: str | None = None, region=None, monitor: int = 1,
                 size: tuple[int, int] | None = (320, 240), fps: float = 10.0):
        import mss
        self._sct = (getattr(mss, "MSS", None) or mss.mss)()
        self.window_name = None
        if window:
            x, y, w, h, self.window_name = find_window(window)
            self.region = (x, y, w, h)
        elif region is not None:
            self.region = tuple(int(v) for v in region)
        else:
            m = self._sct.monitors[monitor]
            self.region = (m["left"], m["top"], m["width"], m["height"])
        if self.region[2] <= 0 or self.region[3] <= 0:
            raise RuntimeError(f"empty capture region {self.region}; is the window minimised?")
        self.size = size
        self.fps = fps

    def describe(self) -> str:
        return self.window_name or f"region {self.region}"

    def read(self) -> np.ndarray:
        x, y, w, h = self.region
        img = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        frame = np.asarray(img)[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        return resize(frame, self.size)

    def close(self) -> None:
        self._sct.close()


class VideoFile(VideoSource):
    """Frames of a video file in order; ``read`` returns None at the end unless ``loop``."""

    def __init__(self, path: str, loop: bool = False, size: tuple[int, int] | None = None):
        import imageio.v2 as imageio
        self.path, self.loop, self.size = path, loop, size
        self._reader = imageio.get_reader(path)
        meta = self._reader.get_meta_data()
        self.fps = float(meta["fps"]) if meta.get("fps") else None
        self._it = None

    def reset(self) -> np.ndarray | None:
        self._it = self._reader.iter_data()
        return self.read()

    def read(self) -> np.ndarray | None:
        if self._it is None:
            self._it = self._reader.iter_data()
        try:
            frame = next(self._it)
        except StopIteration:
            if not self.loop:
                return None
            self._it = self._reader.iter_data()
            frame = next(self._it)
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return resize(frame, self.size)

    def close(self) -> None:
        self._reader.close()


class SyntheticVideo(VideoSource):
    """Frames from a function of the frame index; for tests, dry runs and toy tasks."""

    def __init__(self, frame_fn, n_frames: int | None = None, fps: float = 10.0):
        self.frame_fn, self.n_frames, self.fps = frame_fn, n_frames, fps
        self.t = 0

    def reset(self) -> np.ndarray:
        self.t = 0
        return self.read()

    def read(self) -> np.ndarray | None:
        if self.n_frames is not None and self.t >= self.n_frames:
            return None
        frame = np.asarray(self.frame_fn(self.t), dtype=np.uint8)
        self.t += 1
        return frame
