"""The gRPC service: a thin translation between protobuf messages and ``Session``."""
from __future__ import annotations

import json
import sys
from concurrent import futures

import numpy as np

from neurofly_core.model import Model
from neurofly_core.rpc import neurofly_pb2 as pb
from neurofly_core.rpc import neurofly_pb2_grpc as rpc
from neurofly_core.server import Session, audio_from_bytes, frame_from_bytes


def _selection(sel: pb.Selection) -> dict:
    if sel.indices:
        return {"indices": list(sel.indices)}
    if sel.ids:
        return {"ids": list(sel.ids)}
    if sel.type_re:
        return {"type_re": sel.type_re}
    if sel.superclass:
        return {"superclass": sel.superclass}
    if sel.name:
        return {"name": sel.name}
    return {}


def _layer(L: pb.Layer) -> dict:
    W = np.asarray(L.w, np.float64).reshape(L.rows, L.cols)
    return {"W": W.tolist(), "b": list(L.b)}


def _fill_activity(msg: pb.ActivityReply, a: dict) -> None:
    msg.t = a["t"]
    msg.indices.extend(a["indices"])
    msg.counts.extend(a["counts"])
    for step in a.get("steps", []):
        msg.steps.add().indices.extend(step)


class Service(rpc.NeuroFlyServicer):
    def __init__(self, session: Session):
        self.session = session

    def _step(self, req: pb.StepRequest, observe: bool) -> pb.StepReply:
        try:
            frame = frame_from_bytes(req.frame, req.format or "rgb", req.width, req.height)
            audio = audio_from_bytes(req.audio, req.channels or 1)
            dets = ([[d.class_id, d.x0, d.y0, d.x1, d.y1, d.score if d.score else 1.0]
                     for d in req.detections] or None)
            r = self.session.step_arrays(frame, audio, req.reward,
                                         observe_only=observe or req.observe_only,
                                         detections=dets,
                                         odours=list(req.odours) if req.odours else None,
                                         tastes=list(req.tastes) if req.tastes else None,
                                         thermo=list(req.thermo) if req.thermo else None)
        except Exception as e:
            return pb.StepReply(ok=False, error=f"{type(e).__name__}: {e}")
        reply = pb.StepReply(ok=True, t=r["t"], spikes=r["spikes"])
        if "features" in r:
            reply.features.extend(r["features"])
        else:
            reply.action.extend(r["action"])
            reply.held.extend(r["held"])
            reply.keys.extend(r["keys"])
            reply.buttons.extend(r["buttons"])
            reply.dx, reply.dy, reply.scroll = r["dx"], r["dy"], r["scroll"]
            reply.pad_buttons.extend(r.get("pad_buttons") or [])
            for k, v in r.get("axes", {}).items():
                reply.axes[k] = v
        if "probe" in r:
            reply.probe.spikes.extend(r["probe"]["spikes"])
            reply.probe.rates.extend(r["probe"]["rates"])
        if "activity" in r:
            _fill_activity(reply.activity, r["activity"])
        return reply

    def _body(self, req: pb.BodyRequest, observe: bool) -> pb.BodyReply:
        try:
            r = self.session.body_step_arrays(np.asarray(req.obs, np.float32), req.reward,
                                              observe_only=observe or req.observe_only)
        except Exception as e:
            return pb.BodyReply(ok=False, error=f"{type(e).__name__}: {e}")
        reply = pb.BodyReply(ok=True, t=r["t"], spikes=r["spikes"])
        if "features" in r:
            reply.features.extend(r["features"])
        else:
            reply.action.extend(r["action"])
        if "probe" in r:
            reply.probe.spikes.extend(r["probe"]["spikes"])
            reply.probe.rates.extend(r["probe"]["rates"])
        if "activity" in r:
            _fill_activity(reply.activity, r["activity"])
        return reply

    def BodyStep(self, request, context):
        return self._body(request, observe=False)

    def BodyStream(self, request_iterator, context):
        for req in request_iterator:
            yield self._body(req, observe=False)

    def Info(self, request, context):
        i = self.session.info()
        return pb.InfoReply(name=i["name"], n_neurons=i["n_neurons"], n_features=i["n_features"],
                            n_actions=i["n_actions"], controls=i["controls"],
                            brain_ms=i["brain_ms"],
                            has_policy=i["has_policy"], has_audition=i["has_audition"],
                            sample_rate=i["sample_rate"] or 0, retina_grid=i["retina_grid"],
                            has_annotations=i["has_annotations"],
                            layout_json=json.dumps(i["layout"]), kind=i["kind"],
                            n_obs=i.get("n_obs", 0), obs_json=json.dumps(i.get("obs", {})),
                            detection_classes=i.get("detection_classes") or [],
                            odour_channels=i.get("odour_channels") or [],
                            taste_channels=i.get("taste_channels") or [],
                            thermo_channels=i.get("thermo_channels") or [])

    def Reset(self, request, context):
        self.session.model.reset()
        return pb.Ack(ok=True)

    def Step(self, request, context):
        return self._step(request, observe=False)

    def Observe(self, request, context):
        return self._step(request, observe=True)

    def Stream(self, request_iterator, context):
        for req in request_iterator:
            yield self._step(req, observe=False)

    def SetPolicy(self, request, context):
        try:
            if request.type == "linear":
                L = _layer(request.layers[0])
                self.session.model.policy = self.session.policy_from({"type": "linear", **L})
            else:
                req = {"type": "mlp", "layers": [_layer(L) for L in request.layers],
                       "activation": request.activation or "tanh",
                       "obs_clip": request.obs_clip or 10.0, "obs_eps": request.obs_eps or 1e-8}
                if request.obs_mean:
                    req["obs_mean"], req["obs_var"] = list(request.obs_mean), list(request.obs_var)
                self.session.model.policy = self.session.policy_from(req)
            return pb.Ack(ok=True)
        except Exception as e:
            return pb.Ack(ok=False, error=f"{type(e).__name__}: {e}")

    def Save(self, request, context):
        return pb.SaveReply(path=self.session.save(request.path, request.name or None))

    def Stimulate(self, request, context):
        return pb.Count(n=self.session.model.stimulate(_selection(request), request.mv))

    def Silence(self, request, context):
        return pb.Count(n=self.session.model.silence(_selection(request)))

    def Probe(self, request, context):
        sel = _selection(request)
        return pb.Count(n=self.session.model.probe(sel or None))

    def Clear(self, request, context):
        self.session.model.clear()
        return pb.Ack(ok=True)

    def Activity(self, request, context):
        return pb.Count(n=self.session.model.watch_activity(request.on, request.substeps))

    def Positions(self, request, context):
        m = self.session.handle({"op": "positions"})
        reply = pb.PositionsReply(n=m["n"], unit=m["unit"])
        if m["positions"] is not None:
            reply.positions.extend(v for row in m["positions"] for v in row)
            reply.known.extend(m["known"])
        reply.superclass.extend(m["superclass"] or [])
        for k, v in m["populations"].items():
            reply.populations[k].indices.extend(v)
        return reply

    def Select(self, request, context):
        return pb.Indices(indices=self.session.model.select(_selection(request)).tolist())


def make_server(model: Model, address: str = "127.0.0.1:50051", workers: int = 4,
                after_step=()):
    """A started gRPC server; call ``.wait_for_termination()`` or ``.stop(0)``."""
    import grpc
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers),
                         options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)])
    rpc.add_NeuroFlyServicer_to_server(Service(Session(model, after_step)), server)
    server.bound_port = server.add_insecure_port(address)   # the real port for ":0"
    server.start()
    return server


def serve_grpc(model: Model, address: str = "127.0.0.1:50051", after_step=()) -> None:
    server = make_server(model, address, after_step=after_step)
    print(f"neurofly-core listening on grpc://{address}", file=sys.stderr, flush=True)
    server.wait_for_termination()
