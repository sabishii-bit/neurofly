"""Write a recording from inside the runtime, so a session driven by another program
(a browser game, a Node trainer, anything over the protocol) becomes the same kind of
directory ``neurofly record`` makes: ``video.mp4``, ``actions.npy``, ``meta.json`` and,
when there is sound, ``audio.wav``. The Python trainers (``imitate``, ``surrogate``,
``detect-label``, ``eval``) then apply to it unchanged.

    rec = Recorder("data/recordings/web1", layout, fps=10)
    rec.add(frame, chunk, action)       # every step; action is what was actually done
    rec.close()

Needs the ``video`` extra (imageio) for the frames; sound needs ``soundfile`` (the ``pc``
extra) and is skipped with a note otherwise.
"""
from __future__ import annotations

import json
import os

import numpy as np


class Recorder:
    def __init__(self, path: str, layout, fps: float = 10.0, sample_rate: int = 16000,
                 source: str = "server"):
        import imageio
        self.path, self.layout, self.fps = path, layout, float(fps)
        self.sample_rate, self.source = int(sample_rate), source
        os.makedirs(path, exist_ok=True)
        self._writer = imageio.get_writer(os.path.join(path, "video.mp4"), fps=self.fps,
                                          macro_block_size=1)
        self.actions: list[np.ndarray] = []
        self.chunks: list[np.ndarray] = []
        self.size: tuple[int, int] | None = None
        self.closed = False

    @property
    def n_frames(self) -> int:
        return len(self.actions)

    def add(self, frame: np.ndarray, chunk: np.ndarray | None, action) -> None:
        frame = np.asarray(frame, np.uint8)
        h, w = frame.shape[:2]
        h, w = h - h % 2, w - w % 2                       # mp4 wants even dimensions
        if self.size is None:
            self.size = (w, h)
        if (w, h) != self.size:                           # keep one size for the whole file
            from PIL import Image
            frame = np.asarray(Image.fromarray(frame).resize(self.size))
        self._writer.append_data(np.ascontiguousarray(frame[:self.size[1], :self.size[0]]))
        if chunk is not None:
            self.chunks.append(np.asarray(chunk, np.float32).reshape(len(chunk), -1))
        a = np.zeros(self.layout.n, np.float32) if action is None else \
            np.asarray(action, np.float32).reshape(-1)[:self.layout.n]
        self.actions.append(a)

    def close(self) -> dict:
        if self.closed:
            return self._final
        self.closed = True
        self._writer.close()
        actions = np.asarray(self.actions, np.float32).reshape(-1, self.layout.n)
        np.save(os.path.join(self.path, "actions.npy"), actions)
        meta = self.meta()
        if self.chunks:
            try:
                import soundfile as sf
                data = np.concatenate(self.chunks)
                sf.write(os.path.join(self.path, "audio.wav"), data, self.sample_rate)
                meta["audio"] = {"name": self.source, "sample_rate": self.sample_rate,
                                 "seconds": len(data) / self.sample_rate}
            except ImportError:
                meta["audio_note"] = "sound was sent but soundfile is not installed"
        with open(os.path.join(self.path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._final = meta
        return meta

    def meta(self) -> dict:
        return {"layout": self.layout.to_dict(), "fps": self.fps,
                "size": list(self.size) if self.size else None, "n_frames": self.n_frames,
                "source": self.source, "audio": None}
