"""The detector backend registry, installing, and the new backends with fake packages."""
import json
import sys
import types

import numpy as np
import pytest

from neurofly_training.pc import backends as B
from neurofly_training.pc import detect as D


def test_registry_and_describe():
    names = set(B.BACKENDS)
    assert {"owl2", "owl", "gdino", "yolo-world", "yolo", "rtdetr", "dfine", "ssdlite",
            "onnx"} <= names
    assert all(b.factory for b in B.BACKENDS.values())
    assert all(b.trainer for b in B.BACKENDS.values() if b.trains)
    assert B.get("torchvision").name == "ssdlite" and B.get("owlv2").name == "owl2"
    with pytest.raises(ValueError):
        B.get("nope")
    text = B.describe()
    assert "AGPL" in text and "dfine" in text and "detect-install" in text
    for b in B.BACKENDS.values():
        assert B.resolve(b.factory) is not None
        if b.trains:
            assert callable(B.resolve(b.trainer))


def test_install_and_require(monkeypatch):
    calls = []
    monkeypatch.setattr(B.subprocess, "check_call", lambda cmd: calls.append(cmd))
    assert B.install(["yolo", "yolo-world", "dfine"], dry_run=True) == [
        "ultralytics", "clip @ git+https://github.com/ultralytics/CLIP.git", "transformers"]
    assert calls == []
    B.install(["ssdlite"], upgrade=True)
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"] and "--upgrade" in calls[0]
    assert B.install([]) == []
    monkeypatch.setattr(B.importlib.util, "find_spec", lambda m: None)
    assert not B.get("yolo").installed and B.missing("yolo") == ["ultralytics"]
    with pytest.raises(RuntimeError, match="detect-install yolo"):
        B.require("yolo")
    with pytest.raises(RuntimeError):
        D.make_detector("yolo:x.pt")


def test_specs_with_registry(tmp_path):
    assert D.parse_spec("dfine") == ("dfine", "") and D.parse_spec("dfine:") == ("dfine", "")
    assert D.parse_spec("yolo-world:enemy, door") == ("yolo-world", "enemy, door")
    assert D.parse_spec("torchvision:runs/x") == ("ssdlite", "runs/x")
    assert D.classes_for("gdino:enemy;door") == ["enemy", "door"]
    run = tmp_path / "det"
    run.mkdir()
    json.dump({"backend": "rtdetr", "classes": ["a"]}, open(run / "detector.json", "w"))
    assert D.parse_spec(str(run)) == ("rtdetr", str(run)) and D.classes_for(str(run)) == ["a"]
    assert D.classes_for(f"rtdetr:{run}") == ["a"]


def test_cli_list_and_install(monkeypatch, capsys):
    from neurofly_training import cli
    from neurofly_training.cli import detect_backends
    calls = []
    monkeypatch.setattr(B, "install", lambda names, upgrade=False, dry_run=False:
                        calls.append((tuple(names), dry_run)) or ["pkg"])
    cli.main(["detect-list"])
    assert "backend" in capsys.readouterr().out
    cli.main(["detect-install", "yolo", "--dry-run"])
    out = capsys.readouterr().out
    assert calls == [(("yolo",), True)] and "would install: pkg" in out and "AGPL" in out
    monkeypatch.setattr(sys, "argv", ["neurofly detect-install"])
    detect_backends.main()
    assert "backend" in capsys.readouterr().out                     # no names: the list


def _fake_transformers(monkeypatch, with_grounding=True):
    import torch

    class Proc:
        @classmethod
        def from_pretrained(cls, name, **kw):
            return cls()

        def __call__(self, images=None, text=None, annotations=None, return_tensors="pt"):
            out = {"pixel_values": torch.zeros(1, 3, 8, 8)}
            if text is not None:
                out["input_ids"] = torch.zeros(1, 4, dtype=torch.long)
            if annotations is not None:
                out["labels"] = [{"class_labels": torch.zeros(1, dtype=torch.long),
                                  "boxes": torch.zeros(1, 4)} for _ in annotations]
            return out

        def post_process_object_detection(self, outputs, threshold, target_sizes):
            h, w = target_sizes[0]
            return [{"boxes": torch.tensor([[0.0, 0.0, w / 2, h]]), "scores": torch.tensor([0.9]),
                     "labels": torch.tensor([1])}]

        def post_process_grounded_object_detection(self, outputs, input_ids, threshold,
                                                   text_threshold, target_sizes):
            h, w = target_sizes[0]
            return [{"boxes": torch.tensor([[0.0, 0.0, w / 2, h], [0, 0, w, h]]),
                     "scores": torch.tensor([0.8, 0.7]),
                     "text_labels": ["health pack", "nothing"]}]

        def save_pretrained(self, out):
            pass

    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)
            self.config = types.SimpleNamespace(id2label={0: "person", 1: "car"})

        @classmethod
        def from_pretrained(cls, name, **kw):
            m = cls()
            if "id2label" in kw:
                m.config.id2label = kw["id2label"]
            return m

        def forward(self, pixel_values=None, labels=None, **kw):
            loss = self.lin(torch.ones(1, 1)).sum() * 0 + torch.tensor(0.5, requires_grad=True)
            return types.SimpleNamespace(loss=loss)

        def save_pretrained(self, out):
            import os
            os.makedirs(out, exist_ok=True)

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = fake.AutoProcessor = Proc
    fake.AutoModelForObjectDetection = fake.AutoModelForZeroShotObjectDetection = Mod
    fake.OwlViTProcessor = fake.Owlv2Processor = Proc
    fake.OwlViTForObjectDetection = fake.Owlv2ForObjectDetection = Mod
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def test_transformers_and_gdino_backends(monkeypatch, tmp_path):
    _fake_transformers(monkeypatch)
    frame = np.zeros((48, 64, 3), np.uint8)
    det = D.make_detector("rtdetr:", threshold=0.5)
    assert det.name == "rtdetr" and det.classes == ["person", "car"]
    out = det.detect(frame)
    assert out[0]["label"] == "car" and out[0]["box"] == [0, 0, 0.5, 1.0]
    assert D.make_detector("dfine").classes == ["person", "car"]
    g = D.make_detector("gdino:enemy,health pack", threshold=0.3)
    out = g.detect(frame)
    assert len(out) == 1 and out[0]["label"] == "health pack" and out[0]["class"] == 1
    assert g._class_of("a health pack") == 1 and g._class_of("dog") is None
    with pytest.raises(ValueError):
        D.GroundingDinoDetector([])
    # fine-tuning through the same fake: writes a run directory that is a spec on its own
    frames = [np.zeros((20, 30, 3), np.uint8)] * 4
    dets = [[{"class": 0, "box": [0.1, 0.1, 0.5, 0.5], "score": 1.0}]] * 3 + [[]]
    D.write_yolo_dataset(str(tmp_path / "ds"), frames, dets, ["thing"])
    hist = D.train_dfine(str(tmp_path / "ds"), str(tmp_path / "run"), epochs=2, batch=2,
                         holdout=0.25, size=32)
    assert len(hist["loss"]) == 2 and hist["holdout"] is not None
    meta = json.load(open(tmp_path / "run" / "detector.json"))
    assert meta["backend"] == "dfine" and meta["classes"] == ["thing"]
    assert D.parse_spec(str(tmp_path / "run")) == ("dfine", str(tmp_path / "run"))
    loaded = D.make_detector(str(tmp_path / "run"))
    assert loaded.classes == ["thing"] and loaded.threshold == 0.3
    (tmp_path / "empty" / "images").mkdir(parents=True)
    json.dump(["thing"], open(tmp_path / "empty" / "classes.json", "w"))
    with pytest.raises(ValueError):
        D.train_rtdetr(str(tmp_path / "empty"), str(tmp_path / "r2"), epochs=1)


def test_yolo_world_with_fake_package(monkeypatch):
    import torch

    class Boxes:
        xyxyn = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        conf = torch.tensor([0.6])
        cls = torch.tensor([0.0])

        def __len__(self):
            return 1

    class YOLOWorld:
        def __init__(self, w):
            self.names = {}

        def set_classes(self, names):
            self.names = dict(enumerate(names))

        def predict(self, img, conf, device, verbose):
            return [types.SimpleNamespace(boxes=Boxes())]

    fake = types.ModuleType("ultralytics")
    fake.YOLOWorld = fake.YOLO = YOLOWorld
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setitem(sys.modules, "clip", types.ModuleType("clip"))   # the backend needs both
    det = D.make_detector("yolo-world:enemy,door")
    assert det.classes == ["enemy", "door"] and det.detect(np.zeros((8, 8, 3), np.uint8))[0][
        "label"] == "enemy"
    with pytest.raises(ValueError):
        D.YOLOWorldDetector([])
    assert D.make_detector("yolo").classes == []          # the fake YOLO has no names yet


def test_detect_train_cli_dispatch(monkeypatch, tmp_path, capsys):
    from neurofly_training.cli import detect_train
    seen = {}

    def trainer(dataset, out, **kw):
        seen.update(kw, dataset=dataset, out=out)
        import os
        os.makedirs(out, exist_ok=True)
        json.dump({"backend": "dfine", "classes": ["x"], "onnx": False},
                  open(f"{out}/detector.json", "w"))
        return {"loss": [1.0, 0.5], "holdout": 0.4, "n_train": 3}

    monkeypatch.setattr(B, "resolve", lambda dotted: trainer)
    monkeypatch.setattr(B, "require", lambda name: B.BACKENDS[name])
    monkeypatch.setattr(sys, "argv", ["neurofly-test", str(tmp_path / "ds"), "--out",
                                      str(tmp_path / "run"), "--backend", "dfine", "--epochs",
                                      "2", "--model", "some/model"])
    detect_train.main()
    out = capsys.readouterr().out
    assert seen["model"] == "some/model" and seen["epochs"] == 2 and "holdout 0.4000" in out
    assert "dfine" in out
