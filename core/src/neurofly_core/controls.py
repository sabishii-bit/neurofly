"""What the brain may touch, and how that packs into one action vector.

``ControlLayout`` lists the keys, mouse buttons, gamepad buttons and gamepad
axes the brain may use and whether it may move the mouse or scroll, and packs
them into one action vector in [-1, 1]:

    [keys..., buttons..., pad_buttons..., dx, dy (if mouse), scroll (if scroll), axes...]

Keys and buttons are held while their entry is positive. ``dx``, ``dy`` are a
fraction of ``mouse_speed`` pixels per step, scroll a fraction of
``scroll_speed`` clicks. Axes are the entry itself: sticks in [-1, 1], triggers
mapped to [0, 1]. ``ControlState`` is the unpacked form. Both are plain data;
sending them to real devices is ``neurofly_core.io``.

Key names are single characters ('a', '1') or these words: ctrl, shift, alt,
space, enter, esc, tab, backspace, up, down, left, right, f1 ... f12. Mouse
buttons are left, right, middle. Gamepad buttons: a, b, x, y, lb, rb, start,
back, ls, rs, dup, ddown, dleft, dright. Axes: lx, ly, rx, ry (sticks), lt, rt
(triggers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

MOUSE_BUTTONS = ("left", "right", "middle")
PAD_BUTTONS = ("a", "b", "x", "y", "lb", "rb", "start", "back", "ls", "rs",
               "dup", "ddown", "dleft", "dright")
AXES = ("lx", "ly", "rx", "ry", "lt", "rt")
TRIGGERS = ("lt", "rt")
_ALIASES = {"control": "ctrl", "escape": "esc", "return": "enter", "spacebar": "space",
            "win": "cmd", "super": "cmd"}


def canonical(name: str) -> str:
    name = name.strip()
    if len(name) == 1:
        return name
    name = name.lower()
    return _ALIASES.get(name, name)


def parse_names(spec: str | Iterable[str] | None) -> list[str]:
    """'a,d,ctrl' or ['a', 'd', 'ctrl'] -> ['a', 'd', 'ctrl']."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = spec.split(",")
    return [canonical(k) for k in spec if str(k).strip()]


@dataclass(frozen=True)
class ControlState:
    keys: frozenset = frozenset()
    buttons: frozenset = frozenset()
    dx: float = 0.0        # pixels this step
    dy: float = 0.0
    scroll: float = 0.0    # wheel clicks this step
    pad_buttons: frozenset = frozenset()
    axes: tuple = ()       # ((name, value), ...) with sticks in [-1, 1], triggers in [0, 1]

    @property
    def held(self) -> list[str]:
        return (sorted(self.keys) + [f"mouse:{b}" for b in sorted(self.buttons)]
                + [f"pad:{b}" for b in sorted(self.pad_buttons)])

    @property
    def axis(self) -> dict:
        return dict(self.axes)

    def to_dict(self) -> dict:
        d = {"keys": sorted(self.keys), "buttons": sorted(self.buttons),
             "dx": self.dx, "dy": self.dy, "scroll": self.scroll}
        if self.pad_buttons or self.axes:
            d["pad_buttons"] = sorted(self.pad_buttons)
            d["axes"] = dict(self.axes)
        return d


@dataclass
class ControlLayout:
    keys: list[str] = field(default_factory=list)
    buttons: list[str] = field(default_factory=list)
    mouse: bool = False
    scroll: bool = False
    mouse_speed: float = 50.0   # pixels per step at full deflection
    scroll_speed: float = 3.0   # wheel clicks per step at full deflection
    pad_buttons: list[str] = field(default_factory=list)
    axes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.keys = parse_names(self.keys)
        self.buttons = parse_names(self.buttons)
        self.pad_buttons = parse_names(self.pad_buttons)
        self.axes = parse_names(self.axes)
        bad = [b for b in self.buttons if b not in MOUSE_BUTTONS]
        if bad:
            raise ValueError(f"unknown mouse buttons {bad}; choose from {MOUSE_BUTTONS}")
        bad = [b for b in self.pad_buttons if b not in PAD_BUTTONS]
        if bad:
            raise ValueError(f"unknown gamepad buttons {bad}; choose from {PAD_BUTTONS}")
        bad = [a for a in self.axes if a not in AXES]
        if bad:
            raise ValueError(f"unknown axes {bad}; choose from {AXES}")
        if self.n == 0:
            raise ValueError("the layout has no controls: give keys, buttons, mouse, scroll, "
                             "pad_buttons or axes")

    @property
    def names(self) -> list[str]:
        out = ([f"key:{k}" for k in self.keys] + [f"button:{b}" for b in self.buttons]
               + [f"pad:{b}" for b in self.pad_buttons])
        if self.mouse:
            out += ["mouse:dx", "mouse:dy"]
        if self.scroll:
            out.append("scroll")
        out += [f"axis:{a}" for a in self.axes]
        return out

    @property
    def n(self) -> int:
        return (len(self.keys) + len(self.buttons) + len(self.pad_buttons)
                + 2 * self.mouse + self.scroll + len(self.axes))

    @property
    def n_binary(self) -> int:
        return len(self.keys) + len(self.buttons) + len(self.pad_buttons)

    @property
    def binary_mask(self) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[:self.n_binary] = True
        return m

    @property
    def has_pad(self) -> bool:
        return bool(self.pad_buttons or self.axes)

    def decode(self, vec) -> ControlState:
        v = np.clip(np.asarray(vec, dtype=np.float64).ravel(), -1.0, 1.0)
        if len(v) != self.n:
            raise ValueError(f"action has {len(v)} entries, layout needs {self.n}")
        i = 0
        keys = frozenset(k for k, x in zip(self.keys, v[i:i + len(self.keys)]) if x > 0)
        i += len(self.keys)
        buttons = frozenset(b for b, x in zip(self.buttons, v[i:i + len(self.buttons)]) if x > 0)
        i += len(self.buttons)
        pad_vals = v[i:i + len(self.pad_buttons)]
        pad = frozenset(b for b, x in zip(self.pad_buttons, pad_vals) if x > 0)
        i += len(self.pad_buttons)
        dx = dy = scroll = 0.0
        if self.mouse:
            dx, dy = v[i] * self.mouse_speed, v[i + 1] * self.mouse_speed
            i += 2
        if self.scroll:
            scroll = v[i] * self.scroll_speed
            i += 1
        axes = []
        for a, x in zip(self.axes, v[i:i + len(self.axes)]):
            axes.append((a, float((x + 1) / 2) if a in TRIGGERS else float(x)))
        return ControlState(keys, buttons, float(dx), float(dy), float(scroll), pad, tuple(axes))

    def encode(self, state: ControlState) -> np.ndarray:
        v = np.full(self.n, -1.0, dtype=np.float32)
        i = 0
        for k in self.keys:
            v[i] = 1.0 if k in state.keys else -1.0
            i += 1
        for b in self.buttons:
            v[i] = 1.0 if b in state.buttons else -1.0
            i += 1
        for b in self.pad_buttons:
            v[i] = 1.0 if b in state.pad_buttons else -1.0
            i += 1
        if self.mouse:
            v[i] = np.clip(state.dx / self.mouse_speed, -1, 1)
            v[i + 1] = np.clip(state.dy / self.mouse_speed, -1, 1)
            i += 2
        if self.scroll:
            v[i] = np.clip(state.scroll / self.scroll_speed, -1, 1)
            i += 1
        axis = state.axis
        for a in self.axes:
            x = float(axis.get(a, 0.0))
            v[i] = np.clip(x * 2 - 1 if a in TRIGGERS else x, -1, 1)
            i += 1
        return v

    def to_dict(self) -> dict:
        return {"keys": list(self.keys), "buttons": list(self.buttons), "mouse": self.mouse,
                "scroll": self.scroll, "mouse_speed": self.mouse_speed,
                "scroll_speed": self.scroll_speed, "pad_buttons": list(self.pad_buttons),
                "axes": list(self.axes)}

    @classmethod
    def from_dict(cls, d: dict) -> "ControlLayout":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
