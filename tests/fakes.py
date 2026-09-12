"""Stand-ins for hardware and device libraries, so the code around them can be tested."""
from __future__ import annotations

import types

import numpy as np

from neurofly_core.controls import ControlState


# --- pynput --------------------------------------------------------------------------------

def fake_pynput(log: list):
    class Key:
        pass

    for n in ["ctrl", "ctrl_l", "ctrl_r", "shift", "shift_l", "alt", "alt_l", "cmd", "space",
              "enter", "esc", "tab", "backspace", "up", "down", "left", "right", "f1"]:
        k = Key()
        k.name = n
        setattr(Key, n, k)

    class KeyCode:
        def __init__(self, char=None, vk=None):
            self.char, self.vk = char, vk

        @classmethod
        def from_char(cls, c):
            return cls(char=c)

        def __eq__(self, other):
            return isinstance(other, KeyCode) and other.char == self.char

        def __hash__(self):
            return hash(self.char)

    class KController:
        def press(self, k):
            log.append(("kdown", getattr(k, "char", None) or k.name))

        def release(self, k):
            log.append(("kup", getattr(k, "char", None) or k.name))

    class Button:
        pass

    for n in ("left", "right", "middle"):
        b = Button()
        b.name = n
        setattr(Button, n, b)

    class MController:
        def press(self, b):
            log.append(("mdown", b.name))

        def release(self, b):
            log.append(("mup", b.name))

        def move(self, dx, dy):
            log.append(("move", dx, dy))

        def scroll(self, dx, dy):
            log.append(("scroll", dy))

    listeners = []

    class Listener:
        def __init__(self, **callbacks):
            self.callbacks = callbacks
            self.daemon = False
            self.started = False
            listeners.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

    keyboard = types.SimpleNamespace(Key=Key, KeyCode=KeyCode, Controller=KController,
                                     Listener=Listener)
    mouse = types.SimpleNamespace(Button=Button, Controller=MController, Listener=Listener)
    pynput = types.ModuleType("pynput")
    pynput.keyboard, pynput.mouse = keyboard, mouse
    pynput.listeners = listeners
    return pynput, keyboard, mouse


def install_fake_pynput(monkeypatch, log: list):
    import sys
    pynput, keyboard, mouse = fake_pynput(log)
    monkeypatch.setitem(sys.modules, "pynput", pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard)
    monkeypatch.setitem(sys.modules, "pynput.mouse", mouse)
    return pynput


# --- vgamepad -----------------------------------------------------------------------------

def install_fake_vgamepad(monkeypatch, log: list):
    import sys

    class Pad:
        def press_button(self, button):
            log.append(("pad_down", button))

        def release_button(self, button):
            log.append(("pad_up", button))

        def left_joystick_float(self, x_value_float, y_value_float):
            log.append(("ls", x_value_float, y_value_float))

        def right_joystick_float(self, x_value_float, y_value_float):
            log.append(("rs", x_value_float, y_value_float))

        def left_trigger_float(self, value_float):
            log.append(("lt", value_float))

        def right_trigger_float(self, value_float):
            log.append(("rt", value_float))

        def update(self):
            log.append(("update",))

        def reset(self):
            log.append(("reset",))

    names = ["A", "B", "X", "Y", "LEFT_SHOULDER", "RIGHT_SHOULDER", "START", "BACK", "LEFT_THUMB",
             "RIGHT_THUMB", "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"]
    buttons = types.SimpleNamespace(**{f"XUSB_GAMEPAD_{n}": n for n in names})
    vg = types.ModuleType("vgamepad")
    vg.VX360Gamepad, vg.XUSB_BUTTON = Pad, buttons
    monkeypatch.setitem(sys.modules, "vgamepad", vg)
    return vg


# --- sounddevice / soundcard -------------------------------------------------------------------

def install_fake_sounddevice(monkeypatch):
    import sys
    streams = []

    class InputStream:
        def __init__(self, device=None, samplerate=16000, channels=1, blocksize=1024,
                     callback=None):
            self.callback, self.channels = callback, channels
            self.running = False
            streams.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

        def close(self):
            pass

        def push(self, n):
            self.callback(np.full((n, self.channels), 0.25, np.float32), n, None, None)

    def query_devices(device=None, kind=None):
        devs = [{"name": "Fake Mic", "max_input_channels": 1, "hostapi": 0,
                 "default_samplerate": 44100.0},
                {"name": "Fake Out", "max_input_channels": 0, "hostapi": 0,
                 "default_samplerate": 44100.0}]
        if device is None:
            return devs
        return devs[int(device) if not isinstance(device, str) else 0]

    sd = types.ModuleType("sounddevice")
    sd.InputStream, sd.query_devices = InputStream, query_devices
    sd.query_hostapis = lambda i=None: {"name": "Fake API"}
    sd.streams = streams
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def install_fake_soundcard(monkeypatch):
    import sys

    class Recorder:
        def __init__(self, samplerate, channels):
            self.samplerate, self.channels = samplerate, channels

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def record(self, numframes):
            return np.full((numframes, self.channels), 0.1, np.float32)

    class Mic:
        def __init__(self, name, loop):
            self.name, self.isloopback = name, loop

        def recorder(self, samplerate, channels):
            return Recorder(samplerate, channels)

    sc = types.ModuleType("soundcard")
    sc.default_speaker = lambda: types.SimpleNamespace(name="Fake Speakers")
    sc.all_microphones = lambda include_loopback=False: [Mic("Fake Mic", False),
                                                          Mic("Fake Speakers loopback", True)]
    monkeypatch.setitem(sys.modules, "soundcard", sc)
    return sc


# --- screen, input recorder, panic key -----------------------------------------------------

class FakeScreen:
    live = True

    def __init__(self, window=None, region=None, monitor=1, size=(320, 240), fps=10.0):
        self.window_name = None
        self.region = tuple(region) if region else (0, 0, 64, 48)
        self.size, self.fps = size, fps
        self.rng = np.random.default_rng(0)
        self.closed = False

    def describe(self):
        return f"fake screen {self.region}"

    def reset(self):
        return self.read()

    def read(self):
        return self.rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class FakePanic:
    def __init__(self, key="esc"):
        self.key, self.stopped = key, False

    def close(self):
        pass


class FakeInputRecorder:
    """Human active on every other sample: holds the first key of the layout."""

    def __init__(self, layout, panic="esc", ignore_injected=False, stop_after=None):
        self.layout, self.stopped = layout, False
        self.keys, self.buttons = set(), set()
        self.n, self.stop_after = 0, stop_after

    def sample(self):
        self.n += 1
        if self.stop_after and self.n >= self.stop_after:
            self.stopped = True
        if self.n % 2 == 0 and self.layout.keys:
            return ControlState(frozenset({self.layout.keys[0]}), dx=3.0)
        return ControlState()

    def close(self):
        pass
