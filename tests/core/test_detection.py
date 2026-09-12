"""The detection encoder, the model with detections, the artifact and the protocols."""
import base64
import json

import numpy as np
import pytest

from neurofly_core.artifact import describe, load_model, save_model, validate
from neurofly_core.controls import ControlLayout
from neurofly_core.encode.detection import DetectionEncoder, as_array, paint
from neurofly_core.server import Session
from neurofly_training.build import build_detection, build_model
from neurofly_training.data.connectome import Connectome
from neurofly_training.data.populations import Populations

CLASSES = ["enemy", "health"]
LAYOUT = ControlLayout(keys=["w"], mouse=True)


@pytest.fixture(scope="module")
def cx():
    return Connectome.toy(1200, seed=0).subset("central")


@pytest.fixture(scope="module")
def model(cx):
    return build_model(cx, LAYOUT, brain_ms=5, detect_classes=CLASSES, detection_grid=(2, 4),
                       include_detections=True)


def test_as_array_and_paint():
    dets = [{"class": "enemy", "box": [0.0, 0.0, 0.5, 1.0], "score": 0.8},
            {"class": 1, "box": [0.5, 0.5, 1.0, 1.0]},
            {"class": "nope", "box": [0, 0, 1, 1]},                  # unknown name: dropped
            {"class": 0, "box": [0.9, 0.9, 0.1, 0.1]}]               # inverted: ignored
    arr = as_array(dets, CLASSES)
    assert arr.shape == (3, 6) and arr[0, 0] == 0 and arr[1, 5] == 1.0
    assert abs(arr[0, 5] - 0.8) < 1e-6
    g = paint(arr, 2, (2, 4))
    assert g.shape == (2, 2, 4)
    assert np.allclose(g[0, :, :2], 0.8) and np.allclose(g[0, :, 2:], 0.0)   # left half, enemy
    assert np.allclose(g[1, 1, 2:], 1.0) and g[1, 0].sum() == 0              # bottom right, health
    assert as_array(None, CLASSES).shape == (0, 6)
    assert as_array(np.array([[0, 0, 0, 1, 1, 1.0]]), CLASSES).shape == (1, 6)
    assert as_array([], CLASSES).shape == (0, 6)
    # a half-covered cell scores half
    half = paint(np.array([[0, 0.0, 0.0, 0.125, 0.5, 1.0]], np.float32), 1, (2, 4))
    assert abs(half[0, 0, 0] - 0.5) < 1e-6 and half[0, 1, 0] == 0


def test_encoder_drive_and_tables(cx):
    pops = Populations(cx)
    enc = build_detection(pops, cx.n, CLASSES, grid=(2, 4), gain=10.0)
    assert enc.n_inputs == 16 and enc.n_classes == 2 and enc.targets.size > 0
    zero = enc(None)
    assert zero.sum() == 0 and enc.levels.sum() == 0
    drive = enc([{"class": "enemy", "box": [0, 0, 1, 1], "score": 1.0}])
    assert drive.max() > 0 and drive.min() >= 0 and np.count_nonzero(enc.levels) == 8
    assert set(np.flatnonzero(drive.cpu().numpy())) <= set(enc.targets.tolist())
    enc.reset()
    assert enc.levels.sum() == 0
    back = DetectionEncoder.from_tables(cx.n, enc.params(), enc.tables())
    assert back.classes == CLASSES and back.grid == (2, 4)
    assert np.allclose(back([{"class": 0, "box": [0, 0, 1, 1]}]).cpu(), drive.cpu())
    with pytest.raises(ValueError):
        DetectionEncoder(cx.n, classes=CLASSES, matrix=enc.matrix[:, :3], targets=enc.targets,
                         grid=(2, 4))


def test_model_uses_detections(model):
    frame = np.full((6, 8, 3), 100, np.uint8)
    assert model.n_features == len(model.readout_idx) + 16
    assert "detection" in model.populations()
    model.reset()
    f0 = model.observe(frame)
    assert f0[-16:].sum() == 0
    model.reset()
    f1 = model.observe(frame, detections=[{"class": "enemy", "box": [0, 0, 1, 1]}])
    assert np.count_nonzero(f1[-16:]) == 8
    assert model.last_spikes > 0
    state, info = model.step(frame, detections=[{"class": 1, "box": [0, 0, 0.5, 0.5]}]) \
        if model.policy is not None else (None, None)
    model.reset()
    assert model.detection.levels.sum() == 0


def test_artifact_and_session_with_detections(model, tmp_path):
    path = save_model(model, str(tmp_path / "det"))
    assert validate(path) == [] and "detection: enemy, health" in describe(path)
    loaded = load_model(path)
    assert loaded.detection.classes == CLASSES and loaded.include_detections
    s = Session(loaded)
    assert s.info()["detection_classes"] == CLASSES
    frame = base64.b64encode(np.zeros((6, 8, 3), np.uint8).tobytes()).decode()
    plain = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6})
    with_det = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6,
                         "detections": [{"class": "health", "box": [0.5, 0, 1, 1], "score": 0.5}]})
    assert plain["ok"] and with_det["ok"]
    assert sum(plain["features"][-16:]) == 0 and sum(with_det["features"][-16:]) > 0
    rows = s.handle({"op": "observe", "frame": frame, "width": 8, "height": 6,
                     "detections": [[0, 0, 0, 1, 1, 1.0]]})
    assert rows["ok"] and sum(rows["features"][-16:-8]) > 0
    m = json.load(open(tmp_path / "det" / "manifest.json"))
    m["detection"]["params"]["classes"] = []
    json.dump(m, open(tmp_path / "det" / "manifest.json", "w"))
    assert any("classes" in p for p in validate(path))


def test_grpc_detections(model):
    grpc = pytest.importorskip("grpc")
    from neurofly_core.rpc import neurofly_pb2 as pb
    from neurofly_core.rpc import neurofly_pb2_grpc as rpc
    from neurofly_core.rpc.server import make_server
    server = make_server(model, "127.0.0.1:0")
    try:
        stub = rpc.NeuroFlyStub(grpc.insecure_channel(f"127.0.0.1:{server.bound_port}"))
        assert list(stub.Info(pb.Empty()).detection_classes) == CLASSES
        frame = np.zeros((6, 8, 3), np.uint8).tobytes()
        r = stub.Observe(pb.StepRequest(frame=frame, width=8, height=6,
                                        detections=[pb.Detection(class_id=0, x0=0, y0=0,
                                                                 x1=1, y1=1)]))
        assert r.ok and sum(r.features[-16:-8]) > 0 and sum(r.features[-8:]) == 0
    finally:
        server.stop(0)
