"""Sound in: a capture device, what the PC is playing, an audio file, or synthetic sound.

``read()`` returns float32 samples of shape (n, channels) in [-1, 1]: for live
sources everything captured since the last read, for file and synthetic
sources the next ``1 / fps`` seconds (an empty array at the end of a file).
"""
from __future__ import annotations

import queue
import threading

import numpy as np


class AudioSource:
    live: bool = False
    sample_rate: int = 16000
    channels: int = 1

    def reset(self) -> np.ndarray:
        return self.read()

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SilentAudio(AudioSource):
    """Zeros, one chunk per frame; for recordings without sound."""

    def __init__(self, fps: float = 10.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.block = int(round(sample_rate / fps))

    def read(self) -> np.ndarray:
        return np.zeros((self.block, 1), np.float32)


class SyntheticAudio(AudioSource):
    """Chunks from a function ``chunk_fn(t, n_samples, sample_rate) -> (n,) or (n, ch)``."""

    def __init__(self, chunk_fn, fps: float = 10.0, sample_rate: int = 16000):
        self.chunk_fn, self.sample_rate = chunk_fn, sample_rate
        self.block = int(round(sample_rate / fps))
        self.t = 0

    def reset(self) -> np.ndarray:
        self.t = 0
        return self.read()

    def read(self) -> np.ndarray:
        x = np.asarray(self.chunk_fn(self.t, self.block, self.sample_rate), np.float32)
        self.t += 1
        return x.reshape(len(x), -1)


class AudioFile(AudioSource):
    """Audio of a recording, ``1 / fps`` seconds per read, in step with its video."""

    def __init__(self, path: str, fps: float = 10.0):
        import soundfile as sf
        self.path = path
        self._file = sf.SoundFile(path)
        self.sample_rate = int(self._file.samplerate)
        self.channels = int(self._file.channels)
        self.block = int(round(self.sample_rate / fps))

    def reset(self) -> np.ndarray:
        self._file.seek(0)
        return self.read()

    def read(self) -> np.ndarray:
        x = self._file.read(self.block, dtype="float32", always_2d=True)
        return np.asarray(x, np.float32)

    def close(self) -> None:
        self._file.close()


class AudioCapture(AudioSource):
    """Live sound. ``device`` is a capture device name or index (microphone, line in);
    ``loopback=True`` captures what the PC is playing on its default output instead."""
    live = True

    def __init__(self, device=None, sample_rate: int = 16000, channels: int = 1,
                 loopback: bool = False, block: int = 1024):
        self.sample_rate, self.channels = int(sample_rate), int(channels)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        if loopback:
            import soundcard as sc
            spk = sc.default_speaker()
            mics = [m for m in sc.all_microphones(include_loopback=True)
                    if m.isloopback and spk.name in m.name]
            if not mics:
                raise RuntimeError(f"no loopback capture for output {spk.name!r}")
            self.name = f"loopback of {spk.name}"
            rec = mics[0].recorder(samplerate=self.sample_rate, channels=self.channels)

            def run():
                with rec as r:
                    while not self._stop.is_set():
                        self._q.put(np.asarray(r.record(numframes=block), np.float32))

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        else:
            import sounddevice as sd
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            info = sd.query_devices(device, "input")
            self.name = info["name"]

            def callback(indata, frames, time_info, status):
                self._q.put(np.array(indata, np.float32, copy=True))

            self._stream = sd.InputStream(device=device, samplerate=self.sample_rate,
                                          channels=self.channels, blocksize=block,
                                          callback=callback)
            self._stream.start()

    def read(self) -> np.ndarray:
        chunks = []
        while True:
            try:
                chunks.append(self._q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros((0, self.channels), np.float32)
        return np.concatenate(chunks).reshape(-1, self.channels)

    def close(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def list_audio_devices() -> str:
    """Capture devices sounddevice can open, plus the loopback of the default output."""
    import sounddevice as sd
    lines = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            lines.append(f"  {i:3d}  {d['name']}  ({api}, {d['default_samplerate']:.0f} Hz)")
    try:
        import soundcard as sc
        lines.append(f"  loopback  what the PC plays on {sc.default_speaker().name!r}")
    except Exception as e:  # pragma: no cover
        lines.append(f"  loopback  unavailable ({e})")
    return "\n".join(lines)


def make_audio(spec, fps: float = 10.0, sample_rate: int = 16000) -> AudioSource | None:
    """``None`` / 'none' -> no audio; 'loopback' -> what the PC plays; anything else -> a
    capture device name or index. 'file' (a recording's own sound) means no live audio."""
    if spec is None or str(spec).lower() in ("none", "file"):
        return None
    if str(spec).lower() == "loopback":
        return AudioCapture(loopback=True, sample_rate=sample_rate)
    return AudioCapture(device=spec, sample_rate=sample_rate)
