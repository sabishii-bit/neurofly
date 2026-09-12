"""Keyboard, mouse and gamepad out, and watching a human's inputs.

``Controls`` applies a ``ControlState`` to devices: ``PCControls`` (keyboard and
mouse through pynput), ``GamepadControls`` (a virtual Xbox pad through ViGEm),
``CompositeControls`` (several at once), ``LoggingControls`` (prints) and
``NullControls`` (discards). The base class tracks what is held and presses or
releases only what changed. ``InputRecorder`` watches the real keyboard, mouse
and (on Windows) an XInput controller and produces the same ``ControlState``
per frame, for imitation. ``PanicKey`` stops everything on one key.
"""
from __future__ import annotations

import sys

from neurofly_core.controls import MOUSE_BUTTONS, ControlLayout, ControlState, canonical


def pynput_key(name: str):
    from pynput.keyboard import Key, KeyCode
    name = canonical(name)
    if len(name) == 1:
        return KeyCode.from_char(name)
    try:
        return getattr(Key, name)
    except AttributeError as e:
        raise ValueError(f"unknown key name {name!r}") from e


def key_name(key) -> str | None:
    """pynput key event -> canonical name (left/right modifiers folded together)."""
    from pynput.keyboard import Key, KeyCode
    if isinstance(key, KeyCode):
        return key.char if key.char and len(key.char) == 1 else None
    if isinstance(key, Key):
        n = key.name
        for mod in ("ctrl", "shift", "alt", "cmd"):
            if n.startswith(mod + "_"):
                return mod
        return n
    return None


class Controls:
    """Tracks what is held; ``apply`` presses and releases only what changed."""

    def __init__(self):
        self.state = ControlState()

    def apply(self, state: ControlState) -> None:
        for k in state.keys - self.state.keys:
            self._press_key(k)
        for k in self.state.keys - state.keys:
            self._release_key(k)
        for b in state.buttons - self.state.buttons:
            self._press_button(b)
        for b in self.state.buttons - state.buttons:
            self._release_button(b)
        for b in state.pad_buttons - self.state.pad_buttons:
            self._press_pad(b)
        for b in self.state.pad_buttons - state.pad_buttons:
            self._release_pad(b)
        if state.dx or state.dy:
            self._move(state.dx, state.dy)
        if state.scroll:
            self._scroll(state.scroll)
        if state.axes != self.state.axes:
            axes = {a: 0.0 for a, _ in self.state.axes}
            axes.update(state.axes)
            self._axes(axes)
        self.state = state

    def release_all(self) -> None:
        self.apply(ControlState())

    def close(self) -> None:
        self.release_all()

    def _press_key(self, name):
        pass

    def _release_key(self, name):
        pass

    def _press_button(self, name):
        pass

    def _release_button(self, name):
        pass

    def _press_pad(self, name):
        pass

    def _release_pad(self, name):
        pass

    def _move(self, dx, dy):
        pass

    def _scroll(self, n):
        pass

    def _axes(self, values: dict):
        pass


class NullControls(Controls):
    pass


class LoggingControls(Controls):
    def _press_key(self, name):
        print(f"  press   {name}")

    def _release_key(self, name):
        print(f"  release {name}")

    def _press_button(self, name):
        print(f"  mouse   {name} down")

    def _release_button(self, name):
        print(f"  mouse   {name} up")

    def _press_pad(self, name):
        print(f"  pad     {name} down")

    def _release_pad(self, name):
        print(f"  pad     {name} up")

    def _move(self, dx, dy):
        print(f"  mouse   move {dx:+.0f}, {dy:+.0f}")

    def _scroll(self, n):
        print(f"  scroll  {n:+.1f}")

    def _axes(self, values):
        print("  axes    " + " ".join(f"{a}={v:+.2f}" for a, v in sorted(values.items())))


class PCControls(Controls):
    """The real keyboard and mouse, through pynput. Gamepad entries are ignored here;
    pair with ``GamepadControls`` through ``CompositeControls``."""

    def __init__(self):
        super().__init__()
        from pynput import keyboard, mouse
        self._kb = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._buttons = {b: getattr(mouse.Button, b) for b in MOUSE_BUTTONS}
        self._scroll_carry = 0.0

    def _press_key(self, name):
        self._kb.press(pynput_key(name))

    def _release_key(self, name):
        self._kb.release(pynput_key(name))

    def _press_button(self, name):
        self._mouse.press(self._buttons[name])

    def _release_button(self, name):
        self._mouse.release(self._buttons[name])

    def _move(self, dx, dy):
        self._mouse.move(int(round(dx)), int(round(dy)))

    def _scroll(self, n):
        self._scroll_carry += n
        clicks = int(self._scroll_carry)
        if clicks:
            self._mouse.scroll(0, clicks)
            self._scroll_carry -= clicks


class GamepadControls(Controls):
    """A virtual Xbox 360 controller through ViGEm (the ``vgamepad`` package; needs the
    ViGEmBus driver on Windows). Keyboard and mouse entries are ignored here."""

    def __init__(self):
        super().__init__()
        try:
            import vgamepad as vg
        except ImportError as e:
            raise RuntimeError("gamepad output needs the vgamepad package "
                               "(pip install vgamepad) and the ViGEmBus driver") from e
        self._pad = vg.VX360Gamepad()
        B = vg.XUSB_BUTTON
        self._btn = {"a": B.XUSB_GAMEPAD_A, "b": B.XUSB_GAMEPAD_B, "x": B.XUSB_GAMEPAD_X,
                     "y": B.XUSB_GAMEPAD_Y, "lb": B.XUSB_GAMEPAD_LEFT_SHOULDER,
                     "rb": B.XUSB_GAMEPAD_RIGHT_SHOULDER, "start": B.XUSB_GAMEPAD_START,
                     "back": B.XUSB_GAMEPAD_BACK, "ls": B.XUSB_GAMEPAD_LEFT_THUMB,
                     "rs": B.XUSB_GAMEPAD_RIGHT_THUMB, "dup": B.XUSB_GAMEPAD_DPAD_UP,
                     "ddown": B.XUSB_GAMEPAD_DPAD_DOWN, "dleft": B.XUSB_GAMEPAD_DPAD_LEFT,
                     "dright": B.XUSB_GAMEPAD_DPAD_RIGHT}

    def _press_pad(self, name):
        self._pad.press_button(button=self._btn[name])
        self._pad.update()

    def _release_pad(self, name):
        self._pad.release_button(button=self._btn[name])
        self._pad.update()

    def _axes(self, values):
        g = lambda k: float(values.get(k, 0.0))  # noqa: E731
        self._pad.left_joystick_float(x_value_float=g("lx"), y_value_float=g("ly"))
        self._pad.right_joystick_float(x_value_float=g("rx"), y_value_float=g("ry"))
        self._pad.left_trigger_float(value_float=max(0.0, g("lt")))
        self._pad.right_trigger_float(value_float=max(0.0, g("rt")))
        self._pad.update()

    def close(self):
        super().close()
        self._pad.reset()
        self._pad.update()


class CompositeControls(Controls):
    """Fan a state out to several devices (keyboard and mouse plus a gamepad)."""

    def __init__(self, children):
        super().__init__()
        self.children = list(children)

    def apply(self, state):
        for c in self.children:
            c.apply(state)
        self.state = state

    def close(self):
        for c in self.children:
            c.close()


def make_controls(kind: str = "pc", layout: ControlLayout | None = None) -> Controls:
    """'pc' (the real devices; a gamepad too when the layout has pad entries), 'log'
    (print), 'none' (discard)."""
    if kind == "log":
        return LoggingControls()
    if kind == "none":
        return NullControls()
    if kind != "pc":
        raise ValueError(f"unknown controls kind {kind!r}")
    if layout is not None and layout.has_pad:
        if layout.keys or layout.buttons or layout.mouse or layout.scroll:
            return CompositeControls([PCControls(), GamepadControls()])
        return GamepadControls()
    return PCControls()


class PanicKey:
    """Background listener; ``stopped`` becomes True once ``key`` is pressed."""

    def __init__(self, key: str = "esc"):
        from pynput import keyboard
        self.key = canonical(key)
        self.stopped = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if key_name(key) == self.key:
            self.stopped = True

    def close(self) -> None:
        self._listener.stop()


class XInputPad:
    """Reads an Xbox-style controller through XInput (Windows only)."""
    _BITS = {"dup": 0x0001, "ddown": 0x0002, "dleft": 0x0004, "dright": 0x0008, "start": 0x0010,
             "back": 0x0020, "ls": 0x0040, "rs": 0x0080, "lb": 0x0100, "rb": 0x0200,
             "a": 0x1000, "b": 0x2000, "x": 0x4000, "y": 0x8000}

    def __init__(self, index: int = 0):
        if sys.platform != "win32":
            raise RuntimeError("controller input is read through XInput, Windows only")
        import ctypes

        class Gamepad(ctypes.Structure):
            _fields_ = [("buttons", ctypes.c_ushort),
                        ("lt", ctypes.c_ubyte), ("rt", ctypes.c_ubyte),
                        ("lx", ctypes.c_short), ("ly", ctypes.c_short),
                        ("rx", ctypes.c_short), ("ry", ctypes.c_short)]

        class State(ctypes.Structure):
            _fields_ = [("packet", ctypes.c_ulong), ("pad", Gamepad)]

        self._State = State
        self._dll = None
        for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                self._dll = ctypes.windll.LoadLibrary(name)
                break
            except OSError:
                continue
        if self._dll is None:
            raise RuntimeError("no XInput library found")
        self.index = index

    def read(self) -> tuple[frozenset, tuple]:
        """(pad buttons held, axes) or empty values when no controller is connected."""
        import ctypes
        st = self._State()
        if self._dll.XInputGetState(self.index, ctypes.byref(st)) != 0:
            return frozenset(), ()
        p = st.pad
        buttons = frozenset(n for n, bit in self._BITS.items() if p.buttons & bit)
        axes = (("lx", p.lx / 32767.0), ("ly", p.ly / 32767.0), ("rx", p.rx / 32767.0),
                ("ry", p.ry / 32767.0), ("lt", p.lt / 255.0), ("rt", p.rt / 255.0))
        return buttons, axes


class InputRecorder(PanicKey):
    """Watches the real keyboard, mouse and controller. ``sample()`` returns the
    ControlState since the previous sample: keys, buttons and pad buttons held now,
    mouse motion and scroll accumulated, stick and trigger positions. Mouse motion
    comes from cursor position, so a program that hides and re-centres the cursor
    will not report it."""

    def __init__(self, layout: ControlLayout, panic: str = "esc", ignore_injected: bool = False):
        """``ignore_injected`` drops events this process sent itself (Windows), so a
        recorder running next to the fly's own controls sees only the human."""
        from pynput import keyboard, mouse
        self.layout = layout
        self.key = canonical(panic)
        self.stopped = False
        self.keys: set[str] = set()
        self.buttons: set[str] = set()
        self._dx = self._dy = self._scroll = 0.0
        self._pos = None
        kw_k, kw_m = {}, {}
        if ignore_injected and sys.platform == "win32":
            injected_k, injected_m = 0x10, 0x01      # LLKHF_INJECTED, LLMHF_INJECTED
            kw_k["win32_event_filter"] = lambda msg, data: not (data.flags & injected_k)
            kw_m["win32_event_filter"] = lambda msg, data: not (data.flags & injected_m)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release,
                                           **kw_k)
        self._listener.daemon = True
        self._listener.start()
        self._mouse = mouse.Listener(on_move=self._on_move, on_click=self._on_click,
                                     on_scroll=self._on_scroll, **kw_m)
        self._mouse.daemon = True
        self._mouse.start()
        self._pad = XInputPad() if layout.has_pad else None

    def _on_press(self, key):
        n = key_name(key)
        if n == self.key:
            self.stopped = True
        if n in self.layout.keys:
            self.keys.add(n)

    def _on_release(self, key):
        self.keys.discard(key_name(key))

    def _on_move(self, x, y):
        if self._pos is not None:
            self._dx += x - self._pos[0]
            self._dy += y - self._pos[1]
        self._pos = (x, y)

    def _on_click(self, x, y, button, pressed):
        name = button.name
        if name in self.layout.buttons:
            (self.buttons.add if pressed else self.buttons.discard)(name)

    def _on_scroll(self, x, y, dx, dy):
        self._scroll += dy

    def sample(self) -> ControlState:
        pad_buttons, axes = frozenset(), ()
        if self._pad is not None:
            pad_buttons, axes = self._pad.read()
        pad_buttons = frozenset(b for b in pad_buttons if b in self.layout.pad_buttons)
        axes = tuple((a, v) for a, v in axes if a in self.layout.axes)
        s = ControlState(frozenset(self.keys), frozenset(self.buttons),
                         self._dx, self._dy, self._scroll, pad_buttons, axes)
        self._dx = self._dy = self._scroll = 0.0
        return s

    def close(self) -> None:
        self._listener.stop()
        self._mouse.stop()
