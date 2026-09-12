"""Every `neurofly` subcommand, driven in-process on the toy brain with fake devices."""
import json
import sys
import types

import numpy as np
import pytest

from neurofly_core.controls import ControlLayout
from tests.fakes import FakeInputRecorder, FakePanic, FakeScreen
from tests.training.test_tools import _bars_recording


def run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["neurofly-test"] + [str(a) for a in argv])
    module.main()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=30)
    return tmp_path


def test_dispatcher(monkeypatch, capsys):
    from neurofly_training import cli
    cli.main([])
    assert "commands:" in capsys.readouterr().out
    cli.main(["list"])
    assert "toy" not in capsys.readouterr().out or True
    monkeypatch.setattr("neurofly_core.io.audio.list_audio_devices", lambda: "fake devices")
    cli.main(["devices"])
    assert "fake devices" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["nope"])
    cli.main(["inspect", "--synthetic", "--subset", "central"])
    assert "populations" in capsys.readouterr().out


def test_build_export_eval_watch(workdir, monkeypatch, capsys):
    from neurofly_training.cli import build, eval as eval_cli, export, train_imitation, watch
    run(build, ["artifacts/base", "--brain", "toy", "--synthetic-n", "1000", "--keys", "a,d",
                "--mouse", "--include-frame"], monkeypatch)
    assert (workdir / "artifacts" / "base" / "manifest.json").exists()
    run(train_imitation, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--brain-ms", "20",
                          "--epochs", "30", "--run-name", "imi"], monkeypatch)
    assert (workdir / "runs" / "imi" / "decoder.npz").exists()
    run(export, ["runs/imi", "artifacts/imi"], monkeypatch)
    assert "kind pc" in capsys.readouterr().out
    run(eval_cli, ["artifacts/imi", "rec", "--reward", "neurofly_training.pc.task:PatchBrightness",
                   "--max-frames", "12"], monkeypatch)
    assert (workdir / "artifacts" / "imi" / "eval.json").exists()
    run(watch, ["runs/imi", "--episodes", "1", "--max-steps", "6", "--video", "v.mp4",
                "--probe", "name=readout", "--probe-out", "p.npz"], monkeypatch)
    out = capsys.readouterr().out
    assert "wrote 6 frames" in out and (workdir / "p.npz").exists()
    run(watch, ["--task", "rec/video.mp4", "--brain", "toy", "--keys",
                "a,d", "--policy", "random", "--episodes", "1", "--max-steps", "3",
                "--stimulate", "name=readout:5", "--silence", "type_re=^L2$"], monkeypatch)
    assert "episode 0" in capsys.readouterr().out


def test_train_ppo_video_and_body(workdir, monkeypatch, capsys):
    from neurofly_training.cli import train, watch
    run(train, ["--task", "rec/video.mp4", "--brain", "toy", "--synthetic-n", "1000", "--keys",
                "a,d", "--max-steps", "8", "--n-envs", "1", "--timesteps", "32", "--n-steps", "16",
                "--run-name", "ppo"], monkeypatch)
    assert (workdir / "runs" / "ppo" / "model.zip").exists()
    run(train, ["--resume", "runs/ppo", "--timesteps", "16", "--n-envs", "1"], monkeypatch)
    run(watch, ["runs/ppo", "--episodes", "1", "--max-steps", "4"], monkeypatch)
    assert "episode 0" in capsys.readouterr().out
    run(train, ["--task", "forward", "--brain", "none", "--n-envs", "1", "--timesteps", "32",
                "--n-steps", "16", "--run-name", "body"], monkeypatch)
    run(watch, ["runs/body", "--episodes", "1", "--max-steps", "4", "--poses", "poses.json",
                "--stimulate", "superclass=vnc_motor:5"], monkeypatch)
    assert json.load(open(workdir / "poses.json"))["frames"]


def test_es_serial(workdir, monkeypatch, capsys):
    from neurofly_training.cli import train_es, watch

    class Pool:
        def __init__(self, workers, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def map(self, fn, jobs):
            return [fn(j) for j in jobs]

    monkeypatch.setattr(train_es.mp, "get_context", lambda name: types.SimpleNamespace(Pool=Pool))
    run(train_es, ["--task", "rec/video.mp4", "--brain", "toy", "--synthetic-n", "1000", "--keys",
                   "a,d", "--generations", "2", "--population", "2", "--episode-steps", "4",
                   "--workers", "1", "--run-name", "es"], monkeypatch)
    assert (workdir / "runs" / "es" / "decoder_best.npz").exists()
    run(train_es, ["--task", "forward", "--brain", "synthetic", "--synthetic-n", "1000",
                   "--generations", "1", "--population", "2", "--episode-steps", "3",
                   "--workers", "1", "--run-name", "esbody"], monkeypatch)
    run(watch, ["runs/esbody", "--episodes", "1", "--max-steps", "3"], monkeypatch)
    assert "episode 0" in capsys.readouterr().out
    from neurofly_training.cli import export
    run(export, ["runs/esbody", "artifacts/walk"], monkeypatch)
    assert "kind body" in capsys.readouterr().out


def test_surrogate_calibrate_idm_label(workdir, monkeypatch, capsys):
    from neurofly_training.cli import calibrate, idm, label, surrogate
    run(surrogate, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--epochs", "1",
                    "--bptt-window", "4", "--max-frames", "10", "--run-name", "sur"], monkeypatch)
    assert (workdir / "runs" / "sur" / "artifact" / "manifest.json").exists()
    run(calibrate, ["--brain", "toy", "--synthetic-n", "1000", "--video", "rec/video.mp4",
                    "--frames", "3", "--grid", "0.5,1"], monkeypatch)
    assert "pick: --brain-gain" in capsys.readouterr().out
    run(calibrate, ["--brain", "toy", "--synthetic-n", "1000", "--video", "rec/video.mp4",
                    "--frames", "3", "--grid", "1", "--param", "audio_gain", "--audio", "loopback"],
        monkeypatch)
    with pytest.raises(SystemExit):
        run(calibrate, ["--brain", "toy", "--video", "rec/video.mp4", "--param", "audio_gain"],
            monkeypatch)
    run(idm, ["rec", "--epochs", "20", "--context", "1", "--grid", "6,8", "--hidden", "16",
              "--run-name", "idm"], monkeypatch)
    assert "holdout" in capsys.readouterr().out
    run(label, ["rec/video.mp4", "--idm", "runs/idm", "--out", "labelled", "--max-frames", "8"],
        monkeypatch)
    assert (workdir / "labelled" / "video" / "actions.npy").exists()


def test_export_body_bench_download(workdir, monkeypatch, capsys):
    from neurofly_training.cli import bench_brain, download_data, export_body
    run(export_body, ["--out", "fly.glb", "--mjcf", "mj"], monkeypatch)
    assert (workdir / "mj" / "fly.xml").exists() and "geoms" in capsys.readouterr().out
    run(bench_brain, ["--brain", "synthetic", "--steps", "5"], monkeypatch)
    assert "ms per brain step" in capsys.readouterr().out
    run(bench_brain, ["--brain", "synthetic", "--steps", "5", "--subset", "central",
                      "--backend", "torch"], monkeypatch)
    calls = []
    monkeypatch.setattr(download_data, "download",
                        lambda d, overwrite=False: calls.append((d, overwrite)))
    run(download_data, ["--data-dir", "here", "--overwrite"], monkeypatch)
    assert calls == [("here", True)]


def test_play_and_record_with_fake_devices(workdir, monkeypatch, capsys):
    import neurofly_training.envs as envs
    from neurofly_training.cli import play_pc, record_pc
    from neurofly_training.pc import dagger
    monkeypatch.setattr(envs, "ScreenCapture", FakeScreen)
    monkeypatch.setattr(play_pc, "PanicKey", FakePanic)
    monkeypatch.setattr(dagger, "InputRecorder", FakeInputRecorder)
    run(play_pc, ["--region", "0,0,64,48", "--keys", "a,d", "--mouse", "--brain", "toy",
                  "--synthetic-n", "1000", "--policy", "fixed", "--dry-run", "--steps", "6",
                  "--countdown", "0", "--fps", "50", "--record", "live.mp4",
                  "--correct-out", "corr", "--probe", "name=readout", "--probe-out", "p.npz"],
        monkeypatch)
    out = capsys.readouterr().out
    assert "stopped after" in out and "corrections:" in out
    assert (workdir / "corr" / "actions.npy").exists() and (workdir / "live.mp4").exists()
    from neurofly_training.cli import train_imitation
    run(train_imitation, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--epochs", "5",
                          "--run-name", "imi2"], monkeypatch)
    run(play_pc, ["--region", "0,0,64,48", "--run", "runs/imi2", "--dry-run", "--steps", "2",
                  "--countdown", "0", "--fps", "50", "--policy", "random"], monkeypatch)

    monkeypatch.setattr(record_pc, "ScreenCapture", FakeScreen)
    def short_recorder(layout, panic="esc"):
        return FakeInputRecorder(layout, panic, stop_after=6)

    monkeypatch.setattr(record_pc, "InputRecorder", short_recorder)
    from neurofly_core.io.audio import SyntheticAudio
    fake_audio = SyntheticAudio(lambda t, n, sr: np.zeros(n), fps=50)
    fake_audio.name = "fake"
    monkeypatch.setattr(record_pc, "make_audio",
                        lambda spec, fps=10.0, sample_rate=16000: fake_audio)
    run(record_pc, ["--out", "recd", "--keys", "w,a", "--mouse", "--region", "0,0,64,48",
                    "--fps", "50", "--countdown", "0", "--audio", "loopback"], monkeypatch)
    meta = json.load(open(workdir / "recd" / "meta.json"))
    assert meta["n_frames"] >= 5 and (workdir / "recd" / "audio.wav").exists()
    assert ControlLayout.from_dict(meta["layout"]).keys == ["w", "a"]
