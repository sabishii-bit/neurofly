"""Safety around a brain that sends real input.

``FocusGuard`` remembers which window had the keyboard focus when the loop
started and reports when that changes, so the brain cannot type into whatever
you alt-tabbed to. ``Watchdog`` releases every held key and button if the loop
stops calling ``heartbeat`` (a stall in the brain, a hung capture, a crash
that skipped cleanup). Both are no-ops off Windows for the focus part.
"""
from __future__ import annotations

import sys
import threading
import time


def foreground_window() -> int:
    if sys.platform != "win32":
        return 0
    import ctypes
    return int(ctypes.windll.user32.GetForegroundWindow())


def window_title(hwnd: int) -> str:
    if sys.platform != "win32" or not hwnd:
        return ""
    import ctypes
    n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


class FocusGuard:
    """``arm()`` when the target window has focus; ``ok()`` is False once focus moved."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and sys.platform == "win32"
        self.hwnd = 0
        self.title = ""

    def arm(self) -> str:
        self.hwnd = foreground_window()
        self.title = window_title(self.hwnd)
        return self.title

    def ok(self) -> bool:
        if not self.enabled or not self.hwnd:
            return True
        return foreground_window() == self.hwnd

    def describe(self) -> str:
        if not self.enabled:
            return "focus guard off"
        return f"focus guard armed on {self.title!r}"


class Watchdog:
    """Calls ``controls.release_all()`` if ``heartbeat()`` is not called within ``timeout``."""

    def __init__(self, controls, timeout: float = 1.0):
        self.controls, self.timeout = controls, float(timeout)
        self._last = time.monotonic()
        self._stop = threading.Event()
        self.fired = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def heartbeat(self) -> None:
        self._last = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(self.timeout / 4):
            if time.monotonic() - self._last > self.timeout and not self.fired:
                self.fired = True
                try:
                    self.controls.release_all()
                except Exception:
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
