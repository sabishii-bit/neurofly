import sys

import numpy as np
import pytest

from neurofly_core.controls import ControlLayout, ControlState
from tests.fakes import (install_fake_pynput, install_fake_soundcard, install_fake_sounddevice,
                         install_fake_vgamepad)


def test_pc_controls_and_key_helpers(monkeypatch):
    log = []
    install_fake_pynput(monkeypatch, log)
    from neurofly_core.io import controls as C
    c = C.PCControls()
    c.apply(ControlState(frozenset({"w", "ctrl"}), frozenset({"left"}), dx=3.2, dy=-1.0,
                         scroll=1.5))
    c.apply(ControlState(frozenset({"w"}), scroll=0.7))     # scroll carry: 1.5 -> 1, then 1.2 -> 1
    c.close()
    kinds = [e[0] for e in log]
    assert kinds.count("kdown") == 2 and kinds.count("kup") == 2
    assert ("mdown", "left") in log and ("mup", "left") in log and ("move", 3, -1) in log
    assert [e for e in log if e[0] == "scroll"] == [("scroll", 1), ("scroll", 1)]
    with pytest.raises(ValueError):
        C.pynput_key("nosuchkey")
    kb = sys.modules["pynput"].keyboard
    assert C.key_name(kb.Key.ctrl_l) == "ctrl" and C.key_name(kb.Key.esc) == "esc"
    assert C.key_name(kb.KeyCode.from_char("a")) == "a" and C.key_name(object()) is None
    assert C.make_controls("none").__class__.__name__ == "NullControls"
    with pytest.raises(ValueError):
        C.make_controls("weird")


def test_panic_key_and_input_recorder(monkeypatch):
    log = []
    pynput = install_fake_pynput(monkeypatch, log)
    from neurofly_core.io import controls as C
    kb, ms = pynput.keyboard, pynput.mouse
    panic = C.PanicKey("esc")
    assert not panic.stopped
    panic._on_press(kb.Key.esc)
    assert panic.stopped
    panic.close()

    layout = ControlLayout(keys=["w", "a"], buttons=["left"], mouse=True, scroll=True)
    rec = C.InputRecorder(layout, panic="f1", ignore_injected=False)
    rec._on_press(kb.KeyCode.from_char("w"))
    rec._on_press(kb.KeyCode.from_char("z"))                 # not in the layout
    rec._on_move(10, 10)
    rec._on_move(13, 8)
    rec._on_click(0, 0, ms.Button.left, True)
    rec._on_click(0, 0, ms.Button.right, True)               # not in the layout
    rec._on_scroll(0, 0, 0, 2)
    s = rec.sample()
    assert s.keys == {"w"} and s.buttons == {"left"} and (s.dx, s.dy, s.scroll) == (3, -2, 2)
    s2 = rec.sample()
    assert s2.dx == 0 and s2.keys == {"w"}                   # motion is per sample, keys are held
    rec._on_release(kb.KeyCode.from_char("w"))
    rec._on_click(0, 0, ms.Button.left, False)
    assert rec.sample().held == []
    rec._on_press(kb.Key.f1)
    assert rec.stopped
    rec.close()
    assert all(not L.started for L in pynput.listeners[-2:])


def test_input_recorder_injected_filter_on_windows(monkeypatch):
    log = []
    pynput = install_fake_pynput(monkeypatch, log)
    monkeypatch.setattr(sys, "platform", "win32")
    from neurofly_core.io import controls as C
    monkeypatch.setattr(C, "XInputPad", lambda index=0: None)
    rec = C.InputRecorder(ControlLayout(keys=["w"]), ignore_injected=True)
    filt = pynput.listeners[-2].callbacks["win32_event_filter"]
    injected = type("D", (), {"flags": 0x10})()
    real = type("D", (), {"flags": 0})()
    assert filt(0, injected) is False and filt(0, real) is True
    rec.close()


@pytest.mark.skipif(sys.platform != "win32", reason="XInput is Windows-only")
def test_xinput_pad_reads_without_a_controller():
    from neurofly_core.io.controls import XInputPad
    pad = XInputPad(index=3)                 # almost certainly no controller here
    buttons, axes = pad.read()
    assert isinstance(buttons, frozenset) and isinstance(axes, tuple)


def test_gamepad_and_composite_controls(monkeypatch):
    log = []
    install_fake_vgamepad(monkeypatch, log)
    install_fake_pynput(monkeypatch, log)
    from neurofly_core.io import controls as C
    layout = ControlLayout(keys=["w"], pad_buttons=["a", "rb"], axes=["lx", "rt"])
    c = C.make_controls("pc", layout)
    assert isinstance(c, C.CompositeControls) and len(c.children) == 2
    c.apply(ControlState(frozenset({"w"}), pad_buttons=frozenset({"a"}),
                         axes=(("lx", 0.5), ("rt", 0.75))))
    c.close()
    assert ("pad_down", "A") in log and ("pad_up", "A") in log
    assert ("ls", 0.5, 0.0) in log and ("rt", 0.75) in log and ("reset",) in log
    only_pad = C.make_controls("pc", ControlLayout(pad_buttons=["b"]))
    assert isinstance(only_pad, C.GamepadControls)
    monkeypatch.setitem(sys.modules, "vgamepad", None)      # import fails
    with pytest.raises(RuntimeError):
        C.GamepadControls()


def test_audio_capture_device_and_loopback(monkeypatch):
    sd = install_fake_sounddevice(monkeypatch)
    install_fake_soundcard(monkeypatch)
    from neurofly_core.io import audio as A
    cap = A.AudioCapture(device="0", sample_rate=16000)
    assert cap.name == "Fake Mic" and cap.read().shape == (0, 1)
    sd.streams[-1].push(100)
    sd.streams[-1].push(50)
    chunk = cap.read()
    assert chunk.shape == (150, 1) and chunk.dtype == np.float32
    cap.close()
    loop = A.AudioCapture(loopback=True, sample_rate=8000, block=64)
    import time
    time.sleep(0.05)
    got = loop.read()
    loop.close()
    assert got.shape[1] == 1 and got.shape[0] >= 64 and "loopback of" in loop.name
    assert A.make_audio("none") is None and A.make_audio("file") is None
    for spec in ("loopback", "0"):
        src = A.make_audio(spec, sample_rate=8000)
        assert isinstance(src, A.AudioCapture)
        src.close()                                    # a live source must not outlive the test
    listing = A.list_audio_devices()
    assert "Fake Mic" in listing and "loopback" in listing


def test_audio_loopback_without_endpoint(monkeypatch):
    import types
    sc = types.ModuleType("soundcard")
    sc.default_speaker = lambda: types.SimpleNamespace(name="Nothing")
    sc.all_microphones = lambda include_loopback=False: []
    monkeypatch.setitem(sys.modules, "soundcard", sc)
    from neurofly_core.io.audio import AudioCapture
    with pytest.raises(RuntimeError):
        AudioCapture(loopback=True)


def test_video_sources(tmp_path):
    from neurofly_core.io.video import ScreenCapture, SyntheticVideo, VideoFile, find_window, resize
    v = SyntheticVideo(lambda t: np.full((4, 6, 3), t, np.uint8), n_frames=2)
    assert v.reset()[0, 0, 0] == 0 and v.read()[0, 0, 0] == 1 and v.read() is None
    assert resize(np.zeros((4, 6, 3), np.uint8), (3, 2)).shape == (2, 3, 3)
    import imageio.v2 as imageio
    with imageio.get_writer(str(tmp_path / "v.mp4"), fps=5, macro_block_size=1) as w:
        for t in range(3):
            w.append_data(np.full((8, 8, 3), 40 * t, np.uint8))
    f = VideoFile(str(tmp_path / "v.mp4"), loop=True, size=(4, 4))
    first = f.reset()
    assert first.shape == (4, 4, 3)
    f.read()
    f.read()
    assert f.read() is not None          # looped
    f.close()
    if sys.platform == "win32":
        with pytest.raises(RuntimeError):
            find_window("no window has this title 1f8a2c")
        cap = ScreenCapture(region=(0, 0, 32, 16), size=(8, 4))
        assert cap.read().shape == (4, 8, 3) and "region" in cap.describe()
        cap.close()
        with pytest.raises(RuntimeError):
            ScreenCapture(region=(0, 0, 0, 0))
    else:
        with pytest.raises(RuntimeError):
            find_window("x")


def test_focus_guard_on_this_platform():
    from neurofly_core.io.guard import FocusGuard, foreground_window, window_title
    g = FocusGuard(enabled=True)
    title = g.arm()
    assert g.ok() and (sys.platform != "win32" or "armed" in g.describe())
    assert isinstance(title, str) and window_title(0) == ""
    assert isinstance(foreground_window(), int)
