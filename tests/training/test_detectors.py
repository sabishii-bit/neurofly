"""Detector backends (open-vocabulary mocked, torchvision and ONNX real), datasets, the
env with a detector, and the two CLI commands."""
import json
import os
import sys
import types

import numpy as np
import pytest

from neurofly_core.controls import ControlLayout, ControlState
from neurofly_training import envs
from neurofly_training.pc import detect as D
from tests.training.test_tools import _bars_recording

CLASSES = ["bar"]


def _dets_for_bars(frame):
    """A perfect detector for the bars recording: the bright half is the object."""
    left = frame[:, :40].mean() > frame[:, 40:].mean()
    return [{"class": 0, "box": [0.0, 0.0, 0.5, 1.0] if left else [0.5, 0.0, 1.0, 1.0],
             "score": 0.9}]


class PerfectDetector(D.Detector):
    name = "perfect"
    classes = CLASSES

    def detect(self, frame):
        return _dets_for_bars(frame)


def test_specs_and_classes(tmp_path):
    assert D.parse_spec("owl:enemy, health pack") == ("owl", "enemy, health pack")
    assert D.classes_for("owl:enemy, health pack;door") == ["enemy", "health pack", "door"]
    assert D.classes_for(None) == [] and D.classes_for("none") == []
    run = tmp_path / "det"
    run.mkdir()
    json.dump({"backend": "torchvision", "classes": ["a", "b"]}, open(run / "detector.json", "w"))
    assert D.parse_spec(str(run)) == ("torchvision", str(run))
    assert D.classes_for(str(run)) == ["a", "b"] and D.classes_for(f"onnx:{run}") == ["a", "b"]
    with pytest.raises(ValueError):
        D.parse_spec("nothing here")
    with pytest.raises(ValueError):
        D.parse_spec(str(tmp_path))
    assert D.make_detector(None) is None
    with pytest.raises(ValueError):
        D.OpenVocabDetector([])


def test_open_vocab_with_fake_transformers(monkeypatch):
    import torch

    class Proc:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, text, images, return_tensors):
            return {"pixel_values": torch.zeros(1)}

        def post_process_grounded_object_detection(self, outputs, threshold, target_sizes):
            h, w = [int(v) for v in target_sizes[0]]
            return [{"boxes": torch.tensor([[0.0, 0.0, w / 2, h], [w / 2, 0, w, h]]),
                     "scores": torch.tensor([0.9, 0.05]), "labels": torch.tensor([0, 1])}]

    class Mod(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def forward(self, **kw):
            return {}

    fake = types.ModuleType("transformers")
    fake.OwlViTProcessor, fake.OwlViTForObjectDetection = Proc, Mod
    fake.Owlv2Processor, fake.Owlv2ForObjectDetection = Proc, Mod
    monkeypatch.setitem(sys.modules, "transformers", fake)
    det = D.make_detector("owl:enemy,door", threshold=0.1)
    out = det.detect(np.zeros((48, 64, 3), np.uint8))
    assert len(out) == 1 and out[0]["label"] == "enemy" and out[0]["box"] == [0, 0, 0.5, 1.0]
    v2 = D.make_detector("owl2:x")
    assert isinstance(v2, D.OpenVocabDetector) and v2.v2
    out = v2.detect(np.zeros((48, 64, 3), np.uint8))       # boxes come back in the padded square
    assert out[0]["box"] == [0, 0, 0.5, 64 / 48]


def test_ultralytics_with_fake_package(monkeypatch):
    import torch

    class Boxes:
        xyxyn = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
        conf = torch.tensor([0.7])
        cls = torch.tensor([1.0])

        def __len__(self):
            return 1

    class YOLO:
        names = {0: "a", 1: "b"}

        def __init__(self, w):
            pass

        def predict(self, img, conf, device, verbose):
            return [types.SimpleNamespace(boxes=Boxes())]

    fake = types.ModuleType("ultralytics")
    fake.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    det = D.make_detector("yolo:whatever.pt")
    assert det.classes == ["a", "b"]
    out = det.detect(np.zeros((8, 8, 3), np.uint8))
    assert out[0]["label"] == "b" and abs(out[0]["box"][2] - 0.4) < 1e-6


def test_cached_and_fixed_detectors(tmp_path):
    calls = []

    class Counting(D.Detector):
        name = "count"
        classes = CLASSES

        def detect(self, frame):
            calls.append(1)
            return [{"class": 0, "box": [0, 0, 1, 1], "score": 1.0}]

    path = str(tmp_path / "v.mp4")
    c = D.cached(Counting(), path)
    frame = np.zeros((4, 4, 3), np.uint8)
    c.reset()
    assert c.detect(frame) and c.detect(frame) and len(calls) == 2
    c.reset()
    c.detect(frame)
    assert len(calls) == 2                        # served from memory
    c.close()
    assert os.path.exists(c.path)
    again = D.cached(Counting(), path)
    again.reset()
    assert again.detect(frame)[0]["class"] == 0 and len(calls) == 2   # served from the file
    assert D.cached(None, path) is None and D.cached(Counting(), None).name == "count"
    fixed = D.FixedDetections(CLASSES, [[{"class": 0, "box": [0, 0, 1, 1]}], []])
    assert fixed.detect(frame) and fixed.detect(frame) == [] and fixed.detect(frame)


def test_draw_and_dataset_roundtrip(tmp_path):
    frame = np.zeros((20, 30, 3), np.uint8)
    dets = [{"class": 0, "box": [0.1, 0.2, 0.5, 0.8], "score": 0.9}]
    drawn = D.draw(frame, dets, CLASSES)
    assert drawn.shape == frame.shape and drawn.sum() > 0
    n = D.write_yolo_dataset(str(tmp_path / "ds"), [frame, frame], [dets, []], CLASSES)
    n += D.write_yolo_dataset(str(tmp_path / "ds"), [frame], [dets], CLASSES, prefix="g", start=2)
    assert n == 3 and (tmp_path / "ds" / "data.yaml").exists()
    classes, items = D.read_yolo_dataset(str(tmp_path / "ds"))
    assert classes == CLASSES and len(items) == 3
    _, boxes, labels = items[0]
    assert boxes.shape == (1, 4) and np.allclose(boxes[0], [0.1, 0.2, 0.5, 0.8], atol=1e-5)
    assert labels.tolist() == [0] and items[1][1].shape == (0, 4)


def test_env_with_detector_and_model_classes_check(tmp_path):
    _bars_recording(tmp_path / "rec", T=12)
    video = str(tmp_path / "rec" / "video.mp4")
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a,d", detect=CLASSES,
                           detector=PerfectDetector(), include_detections=True,
                           detection_grid=("2,2"))
    obs, _ = env.reset()
    assert env.model.detection.classes == CLASSES and obs[-4:].sum() > 0
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert info["detections"][0]["class"] == 0
    env.close()
    with pytest.raises(ValueError):
        envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a", detect=["x", "y"],
                         detector=PerfectDetector())
    with pytest.raises(ValueError):
        envs.make_pc_env(video, brain="toy", synthetic_n=1000, keys="a", detect=CLASSES)


def test_imitation_learns_from_detections(tmp_path):
    """With a perfect detector the readout separates left from right on the toy brain."""
    from neurofly_training.pc.imitation import collect_features, evaluate, fit_control_decoder
    layout, frames, actions = _bars_recording(tmp_path / "rec", T=40)
    video = str(tmp_path / "rec" / "video.mp4")
    env = envs.make_pc_env(video, brain="toy", synthetic_n=1000, layout=layout, detect=CLASSES,
                           detector=PerfectDetector(), brain_ms=20, include_detections=True,
                           detection_grid="1,2")
    X, Y = collect_features(env, actions)
    env.close()
    dec = fit_control_decoder(X, Y, layout, epochs=200)
    scores = evaluate(dec, X, Y)
    assert scores["key:a"]["f1"] > 0.9 and scores["key:d"]["f1"] > 0.9


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["neurofly-test"] + [str(a) for a in argv])
    module.main()


def test_label_train_onnx_and_cli(tmp_path, monkeypatch, capsys):
    """detect-label with a stand-in detector, detect-train on it (tiny SSDLite, no
    download), then the torchvision and ONNX backends detect on the same frames."""
    from neurofly_training.cli import detect_label, detect_train
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=16)
    monkeypatch.setattr(D, "make_detector", lambda spec, device="cpu", threshold=None:
                        PerfectDetector())
    monkeypatch.setattr(detect_label, "make_detector", D.make_detector)
    _run(detect_label, ["rec", "--detect", "owl:bar", "--out", "ds", "--every", "2",
                        "--preview", "prev.mp4"], monkeypatch)
    out = capsys.readouterr().out
    assert "8 frames, 8 boxes" in out and (tmp_path / "prev.mp4").exists()
    assert len(os.listdir(tmp_path / "ds" / "images")) == 8
    with pytest.raises(SystemExit):
        _run(detect_label, ["rec", "--detect", "owl:bar"], monkeypatch)
    _run(detect_train, ["ds", "--out", "det1", "--epochs", "2", "--size", "64", "--batch", "4",
                        "--no-pretrained", "--holdout", "0.25"], monkeypatch)
    out = capsys.readouterr().out
    assert "detector -> det1" in out
    meta = json.load(open(tmp_path / "det1" / "detector.json"))
    assert meta["classes"] == CLASSES and (tmp_path / "det1" / "detector.pt").exists()
    hist = json.load(open(tmp_path / "det1" / "history.json"))
    assert len(hist["loss"]) == 2 and hist["n_train"] == 6 and hist["holdout"] is not None
    monkeypatch.undo()
    tv = D.make_detector(str(tmp_path / "det1"), threshold=0.0)
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[:, :40] = 255
    dets = tv.detect(frame)
    assert isinstance(dets, list) and all(0 <= d["box"][0] <= 1 for d in dets)
    assert D.classes_for(str(tmp_path / "det1")) == CLASSES
    if meta["onnx"]:
        ox = D.make_detector(f"onnx:{tmp_path / 'det1'}", threshold=0.0)
        assert ox.classes == CLASSES and isinstance(ox.detect(frame), list)
    else:
        pytest.skip(f"ONNX export unavailable here: {hist.get('onnx_error')}")


def test_eval_and_surrogate_with_detections(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import build, eval as eval_cli, surrogate
    monkeypatch.chdir(tmp_path)
    _bars_recording(tmp_path / "rec", T=16)
    monkeypatch.setattr(D, "make_detector", lambda spec, device="cpu", threshold=None:
                        PerfectDetector())
    _run(build, ["art", "--brain", "toy", "--synthetic-n", "1000", "--keys", "a,d",
                 "--detect", "owl:bar", "--detection-grid", "2,2"], monkeypatch)
    assert "detection: bar" in capsys.readouterr().out
    _run(eval_cli, ["art", "rec", "--max-frames", "6"], monkeypatch)
    assert (tmp_path / "art" / "eval.json").exists()
    _run(surrogate, ["rec", "--brain", "toy", "--synthetic-n", "1000", "--epochs", "1",
                     "--bptt-window", "4", "--max-frames", "8", "--run-name", "sur",
                     "--detect", "owl:bar", "--detection-grid", "2,2"], monkeypatch)
    out = capsys.readouterr().out
    assert "detections from perfect" in out and "kind pc" in out
    assert "detection: bar" in out


def test_perfect_detector_matches_layout():
    frame = np.zeros((60, 80, 3), np.uint8)
    frame[:, 40:] = 255
    assert _dets_for_bars(frame)[0]["box"][0] == 0.5
    assert ControlLayout(keys=["a"]).encode(ControlState(frozenset({"a"})))[0] == 1
