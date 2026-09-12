import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from neurofly_core.controls import ControlLayout, ControlState
from neurofly_core.decode.linear import ControlDecoder
from neurofly_core.encode.audition import AuditionEncoder
from neurofly_core.io.audio import SilentAudio, SyntheticAudio
from neurofly_core.io.controls import Controls, NullControls
from neurofly_core.io.video import SyntheticVideo, VideoFile
from neurofly_training.build import build_audition, build_model
from neurofly_training.data.populations import Populations
from neurofly_training.envs import make_env, make_pc_env
from neurofly_training.pc.env import PCEnv
from neurofly_training.pc.imitation import collect_features, evaluate, fit_control_decoder
from neurofly_training.pc.task import PatchBrightness, Task, load_task

LAYOUT = ControlLayout(keys=["a", "d"], buttons=["left"], mouse=True, scroll=True)


def noise_frames(t, rng=np.random.default_rng(0)):
    return rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)


def test_layout_roundtrip():
    assert LAYOUT.names == ["key:a", "key:d", "button:left", "mouse:dx", "mouse:dy", "scroll"]
    assert LAYOUT.n == 6 and LAYOUT.n_binary == 3
    s = ControlState(frozenset({"a"}), frozenset({"left"}), dx=25.0, dy=-50.0, scroll=1.5)
    v = LAYOUT.encode(s)
    assert v.tolist() == [1.0, -1.0, 1.0, 0.5, -1.0, 0.5]
    back = LAYOUT.decode(v)
    assert back == s and back.held == ["a", "mouse:left"]
    assert ControlLayout.from_dict(LAYOUT.to_dict()).names == LAYOUT.names
    with pytest.raises(ValueError):
        ControlLayout()
    with pytest.raises(ValueError):
        ControlLayout(buttons=["side"])


def test_controls_apply_only_changes():
    log = []

    class Rec(Controls):
        def _press_key(self, n):
            log.append(("+k", n))

        def _release_key(self, n):
            log.append(("-k", n))

        def _press_button(self, n):
            log.append(("+b", n))

        def _release_button(self, n):
            log.append(("-b", n))

        def _move(self, dx, dy):
            log.append(("mv", dx, dy))

        def _scroll(self, n):
            log.append(("sc", n))

    c = Rec()
    c.apply(ControlState(frozenset({"a"}), frozenset({"left"}), dx=3, dy=0))
    c.apply(ControlState(frozenset({"a"}), frozenset(), scroll=1))
    c.close()
    assert log == [("+k", "a"), ("+b", "left"), ("mv", 3, 0), ("-b", "left"), ("sc", 1),
                   ("-k", "a")]


def test_pc_env_video_audio_task(synthetic_cx):
    class HoldA(Task):
        def reward(self, frame, audio, state, info):
            return float("a" in state.keys)

        def done(self, frame, audio, info):
            return info["t"] >= 8

    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=10, audio=True,
                        dopamine_punish=10.0, plasticity=True, include_frame=True,
                        frame_grid=(6, 8), include_audio=True)
    tone = SyntheticAudio(lambda t, n, sr: 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sr))
    env = PCEnv(model, video=SyntheticVideo(noise_frames, fps=10), audio=tone, task=HoldA())
    assert not env.realtime and model.substeps == 20
    assert model.retina.mode == "hex" and model.audition.n_driven > 0
    assert env.observation_space.shape == (len(model.readout_idx) + 48 + 16,)
    assert env.action_space.shape == (6,)
    obs, _ = env.reset()
    assert obs.dtype == np.float32 and obs[-16:].max() > 0  # the tone shows in the audio bands
    press_a = LAYOUT.encode(ControlState(frozenset({"a"})))
    ret, spikes, done = 0.0, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(press_a)
        ret += r
        spikes += info["brain_spikes"]
        done = term or trunc
    assert ret == 8.0 and info["held"] == ["a"] and term
    assert spikes > 0
    assert env.render().shape == (60, 80, 3)
    check_env(env, warn=False)
    env.close()


def test_video_ends_episode(synthetic_cx):
    model = build_model(synthetic_cx, LAYOUT)
    env = PCEnv(model, video=SyntheticVideo(noise_frames, n_frames=5))
    env.reset()
    n, trunc = 0, False
    while not trunc:
        _, _, _, trunc, _ = env.step(np.zeros(6, np.float32))
        n += 1
    assert n == 5
    env.close()


def test_audition_encoder_hears_tones(synthetic_cx):
    pops = Populations(synthetic_cx)
    assert len(pops.auditory) > 0
    enc = build_audition(pops, synthetic_cx.n, sample_rate=16000, n_bands=8)
    assert isinstance(enc, AuditionEncoder)
    n = 1600
    silence = np.zeros((n, 1), np.float32)
    assert enc(silence).sum() == 0
    low = 0.3 * np.sin(2 * np.pi * 100 * np.arange(n) / 16000)
    high = 0.3 * np.sin(2 * np.pi * 4000 * np.arange(n) / 16000)
    assert enc.bands(low).argmax() < enc.bands(high).argmax()
    d = enc(high).numpy()
    assert d.sum() > 0 and np.all(d[np.setdiff1d(np.arange(synthetic_cx.n), pops.auditory)] == 0)


def _write_bars_video(path, T=60):
    import imageio.v2 as imageio
    layout = ControlLayout(keys=["a", "d"], mouse=True)
    actions = np.zeros((T, layout.n), np.float32)
    with imageio.get_writer(str(path), fps=10, macro_block_size=1) as w:
        for t in range(T):
            left = (t // 10) % 2 == 0
            f = np.zeros((60, 80, 3), np.uint8)
            f[:, :40] = 255 if left else 0
            f[:, 40:] = 0 if left else 255
            s = ControlState(frozenset({"a" if left else "d"}), dx=-30.0 if left else 30.0)
            actions[t] = layout.encode(s)
            w.append_data(f)
    return layout, actions


def test_video_file_and_imitation(synthetic_cx, tmp_path):
    video = tmp_path / "video.mp4"
    layout, actions = _write_bars_video(video)
    src = VideoFile(str(video))
    assert src.fps == 10 and src.reset().shape == (60, 80, 3)
    # read the retina neurons directly so the decoder problem is separable
    ret = Populations(synthetic_cx).retina()
    readout = np.concatenate([ret["L"]["idx"], ret["R"]["idx"]])
    model = build_model(synthetic_cx, layout, readout=readout, brain_ms=50, audio=True)
    env = PCEnv(model, video=src, audio=SilentAudio(fps=10))
    X, Y = collect_features(env, actions)
    assert X.shape == (60, len(readout)) and Y.shape == (60, 4)
    dec = fit_control_decoder(X, Y, layout, epochs=200)
    ev = evaluate(dec, X, Y)
    assert ev["key:a"]["f1"] > 0.9 and ev["key:d"]["f1"] > 0.9, ev
    assert ev["mouse:dx"]["corr"] > 0.8, ev
    assert isinstance(dec, ControlDecoder)
    env.close()


def test_patch_brightness_task_and_loader(tmp_path):
    frame = np.zeros((10, 20, 3), np.uint8)
    frame[:, 10:] = 200
    t = PatchBrightness(patch=(0.5, 0.0, 0.5, 1.0), end_below=0.1)
    info = {}
    assert abs(t.reward(frame, None, ControlState(), info) - 200 / 255) < 1e-6
    assert not t.done(frame, None, info) and t.done(np.zeros_like(frame), None, info)
    mod = tmp_path / "my_task.py"
    mod.write_text("from neurofly_training.pc.task import Task\n"
                   "class Always(Task):\n"
                   "    def reward(self, frame, audio, state, info): return 1.0\n")
    assert load_task(f"{mod}:Always").reward(frame, None, ControlState(), {}) == 1.0
    assert load_task(None).reward(frame, None, ControlState(), {}) == 0.0


def test_make_env_pc_paths(synthetic_cx, tmp_path):
    with pytest.raises(ValueError):
        make_env("pc", brain="none", keys="a")
    video = tmp_path / "video.mp4"
    layout, _ = _write_bars_video(video, T=6)
    env = make_env(str(video), brain="synthetic", synthetic_n=1200, keys="a,d", mouse=True,
                   audio="file", max_steps=3)
    assert isinstance(env, PCEnv) and env.layout.names == layout.names
    assert isinstance(env.controls, NullControls) and isinstance(env.audio, SilentAudio)
    assert env.model.audition is not None
    env.reset()
    _, _, _, trunc, _ = env.step(env.action_space.sample())
    assert not trunc
    env.close()
    env = make_pc_env(str(video), brain="synthetic", synthetic_n=1200, layout=layout,
                      task_obj=PatchBrightness())
    obs, _ = env.reset()
    _, r, _, _, info = env.step(np.zeros(4, np.float32))
    assert 0 <= r <= 1 and "brightness" in info
    env.close()
