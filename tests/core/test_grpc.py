import numpy as np
import pytest

from neurofly_core.controls import ControlLayout
from neurofly_core.decode.linear import ControlDecoder
from neurofly_training.build import build_model

grpc = pytest.importorskip("grpc")

from neurofly_core.rpc import neurofly_pb2 as pb  # noqa: E402
from neurofly_core.rpc import neurofly_pb2_grpc as rpc  # noqa: E402
from neurofly_core.rpc.server import make_server  # noqa: E402

LAYOUT = ControlLayout(keys=["w", "a"], mouse=True, pad_buttons=["a"], axes=["lx"])


def _frame(seed):
    return np.random.default_rng(seed).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_grpc_roundtrip(synthetic_cx, tmp_path):
    model = build_model(synthetic_cx.subset("brain"), LAYOUT, brain_ms=5, audio=True)
    server = make_server(model, "127.0.0.1:0")
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
        stub = rpc.NeuroFlyStub(channel)
        info = stub.Info(pb.Empty())
        assert info.n_actions == LAYOUT.n and list(info.controls) == LAYOUT.names
        assert info.has_audition and info.sample_rate == 16000 and info.has_annotations
        audio = (0.1 * np.sin(np.arange(800))).astype("<f4").tobytes()
        req = pb.StepRequest(frame=_frame(0).tobytes(), width=64, height=48, audio=audio,
                             sample_rate=16000, channels=1)
        r = stub.Observe(req)
        assert r.ok and len(r.features) == model.n_features and r.t == 1
        r = stub.Step(req)                      # no policy yet: features again
        assert r.ok and len(r.features) == model.n_features
        n = model.n_features
        W = np.zeros((LAYOUT.n, n))
        b = [1.0, -1.0, 0.0, 0.0, -1.0, 0.5]     # hold w, pad a released, lx = tanh(0.5)
        layer = pb.Layer(rows=LAYOUT.n, cols=n, w=W.ravel().tolist(), b=b)
        ack = stub.SetPolicy(pb.Policy(type="linear", layers=[layer]))
        assert ack.ok
        assert stub.Probe(pb.Selection(name="readout")).n == len(model.readout_idx)
        assert stub.Stimulate(pb.Selection(superclass="descending_neuron", mv=10.0)).n > 0
        r = stub.Step(req)
        assert r.ok and list(r.keys) == ["w"] and abs(r.axes["lx"] - np.tanh(0.5)) < 1e-5
        assert len(r.probe.spikes) == len(model.readout_idx)
        replies = list(stub.Stream(iter([req, req, req])))
        assert [x.t for x in replies] == [r.t + 1, r.t + 2, r.t + 3]
        assert stub.Select(pb.Selection(type_re="^L1$")).indices
        assert stub.Clear(pb.Empty()).ok and model.manipulations == []
        assert stub.Reset(pb.Empty()).ok and model.t == 0
        bad = stub.Step(pb.StepRequest(frame=b"\x00", width=3, height=3))
        assert not bad.ok and "expected" in bad.error
        saved = stub.Save(pb.SaveRequest(path=str(tmp_path / "grpc_art"), name="via_grpc"))
        assert (tmp_path / "grpc_art" / "manifest.json").exists() and saved.path
        channel.close()
    finally:
        server.stop(0)
    assert isinstance(model.policy, ControlDecoder)
