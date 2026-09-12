"""Correcting the fly while it plays: your inputs win, and become labels.

While ``neurofly play --correct-out DIR`` runs, an ``InputRecorder`` watches
the real keyboard, mouse and controller (ignoring the fly's own injected
events). Whenever you touch a control in the layout, the fly's controls are
released and the frame plus your action are appended to a recording; when you
let go, the fly resumes. Training on the correction recording afterwards
(``imitate`` or ``surrogate``, together with the original recordings) is the
DAgger loop: the policy learns from what you did in the states it reached.
"""
from __future__ import annotations

import json
import os

import numpy as np

from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.io.controls import InputRecorder


def human_active(state: ControlState, move_px: float = 2.0) -> bool:
    return bool(state.keys or state.buttons or state.pad_buttons
                or abs(state.dx) > move_px or abs(state.dy) > move_px or state.scroll
                or any(abs(v) > 0.2 for _, v in state.axes))


class CorrectionRecorder:
    def __init__(self, layout: ControlLayout, out_dir: str, fps: float, panic: str = "esc",
                 size=None):
        import imageio
        self.layout, self.out_dir, self.fps = layout, out_dir, fps
        os.makedirs(out_dir, exist_ok=True)
        self.recorder = InputRecorder(layout, panic=panic, ignore_injected=True)
        self._writer = imageio.get_writer(os.path.join(out_dir, "video.mp4"), fps=fps,
                                          macro_block_size=1)
        self.actions: list[np.ndarray] = []
        self.size = size
        self.taken_over = 0

    @property
    def stopped(self) -> bool:
        return self.recorder.stopped

    def step(self, frame: np.ndarray) -> ControlState | None:
        """Call once per step with the current frame. Returns the human's state while the
        human is active (the fly should stand down), else None."""
        state = self.recorder.sample()
        if not human_active(state):
            return None
        self._writer.append_data(frame)
        self.actions.append(self.layout.encode(state))
        self.taken_over += 1
        return state

    def close(self) -> str:
        self.recorder.close()
        self._writer.close()
        actions = np.asarray(self.actions, np.float32).reshape(-1, self.layout.n)
        np.save(os.path.join(self.out_dir, "actions.npy"), actions)
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump({"layout": self.layout.to_dict(), "fps": self.fps, "n_frames": len(actions),
                       "corrections": True}, f, indent=2)
        return self.out_dir
