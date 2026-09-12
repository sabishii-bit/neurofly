"""Object detectors: what is on the screen, as boxes with class names.

The brain cannot learn what an enemy is from pixels; a detector can say where the
enemies are, and the ``DetectionEncoder`` in the core turns that into drive on central
brain neurons. Every backend sits behind one interface and one spec grammar,
``backend:arg`` (``neurofly detect-list`` shows them all with their licences):

    owl2:enemy,health pack,door      OWLv2, open vocabulary: name the objects, no training
    owl:...   gdino:...              OWL-ViT (faster, weaker); Grounding DINO (phrases)
    yolo-world:enemy,health pack     YOLO-World: open vocabulary in real time
    yolo:                            Ultralytics YOLO11 on its COCO classes (yolo:yolo11m.pt)
    yolo:runs/det1                   a YOLO you fine-tuned with `neurofly detect-train`
    rtdetr: / dfine:                 RT-DETRv2 / D-FINE (Apache) pretrained, or fine-tuned dirs
    runs/det1                        any detect-train run directory (detector.json says which)
    onnx:runs/det1                   the exported detector.onnx through ONNX Runtime

Every backend returns, per frame, a list of ``{"class", "box", "score"}`` with the box in
fractions of the frame, and carries ``classes`` (the names, in id order). ``classes_for``
gives the names without loading weights where it can, so a model can be built before a
detector runs. Backends that are not installed say what to run.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

DETECTOR_JSON = "detector.json"


class Detector:
    classes: list[str] = []
    name: str = "none"

    def detect(self, frame: np.ndarray) -> list[dict]:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _rows_to_dicts(boxes, scores, labels, classes, width, height, threshold) -> list[dict]:
    out = []
    for (x0, y0, x1, y1), s, c in zip(boxes, scores, labels):
        c, s = int(c), float(s)
        if s < threshold or not 0 <= c < len(classes):
            continue
        out.append({"class": c, "label": classes[c],
                    "box": [float(x0) / width, float(y0) / height, float(x1) / width,
                            float(y1) / height], "score": s})
    return out


class OpenVocabDetector(Detector):
    """OWL-ViT (Apache-2.0, through the ``transformers`` package): the classes are text
    prompts. About 1 to 3 frames per second on a CPU, faster on a GPU."""
    name = "owl"

    def __init__(self, prompts, model_name: str = "google/owlvit-base-patch32",
                 threshold: float = 0.1, device: str = "cpu"):
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        self.classes = [str(p).strip() for p in prompts if str(p).strip()]
        if not self.classes:
            raise ValueError("an open-vocabulary detector needs at least one prompt")
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.v2 = "owlv2" in model_name.lower()
        if self.v2:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            Proc, Mod = Owlv2Processor, Owlv2ForObjectDetection
        else:
            Proc, Mod = OwlViTProcessor, OwlViTForObjectDetection
        self.processor = Proc.from_pretrained(model_name)
        self.model = Mod.from_pretrained(model_name).to(self.device).eval()
        self._torch = torch

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image
        img = Image.fromarray(np.asarray(frame))
        inputs = self.processor(text=[self.classes], images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        S = max(img.height, img.width) if self.v2 else None       # OWLv2 pads to a square
        sizes = self._torch.tensor([[S, S] if S else [img.height, img.width]], device=self.device)
        post = getattr(self.processor, "post_process_grounded_object_detection", None)
        if post is None:                                     # renamed in transformers 4.5x
            post = self.processor.post_process_object_detection
        res = post(outputs, threshold=self.threshold, target_sizes=sizes)[0]
        return _rows_to_dicts(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                              res["labels"].cpu().numpy(), self.classes, img.width, img.height,
                              self.threshold)


class TorchvisionDetector(Detector):
    """A detector fine-tuned by ``train_torchvision`` (SSDLite, BSD-licensed torchvision)."""
    name = "torchvision"

    def __init__(self, run_dir: str, threshold: float | None = None, device: str = "cpu"):
        import torch
        with open(os.path.join(run_dir, DETECTOR_JSON)) as f:
            meta = json.load(f)
        self.classes = list(meta["classes"])
        self.size = int(meta.get("size", 320))
        self.threshold = float(meta.get("threshold", 0.3) if threshold is None else threshold)
        self.device = torch.device(device)
        self.model = ssdlite(len(self.classes), pretrained_backbone=False, size=self.size)
        state = torch.load(os.path.join(run_dir, "detector.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self._torch = torch

    def detect(self, frame: np.ndarray) -> list[dict]:
        x = self._torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        with self._torch.no_grad():
            out = self.model([x.to(self.device)])[0]
        h, w = frame.shape[:2]
        return _rows_to_dicts(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(),
                              out["labels"].cpu().numpy() - 1, self.classes, w, h, self.threshold)


class OnnxDetector(Detector):
    """The exported ``detector.onnx`` through ONNX Runtime: the same file a consumer in any
    language runs. Input ``image`` float32 [1, 3, size, size] in [0, 1]; outputs boxes
    (pixels in that square), scores and labels (1-based, 0 is background)."""
    name = "onnx"

    def __init__(self, run_dir: str, threshold: float | None = None):
        import onnxruntime as ort
        with open(os.path.join(run_dir, DETECTOR_JSON)) as f:
            meta = json.load(f)
        self.classes = list(meta["classes"])
        self.size = int(meta.get("size", 320))
        self.threshold = float(meta.get("threshold", 0.3) if threshold is None else threshold)
        self.layout = meta.get("onnx_layout", "torchvision")
        self.session = ort.InferenceSession(os.path.join(run_dir, "detector.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image
        img = Image.fromarray(np.asarray(frame)).resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0
        outputs = self.session.run(None, {self.input_name: x})
        if self.layout == "ultralytics":
            boxes, scores, labels = decode_ultralytics(outputs[0], self.threshold)
        else:                          # torchvision: boxes, scores, 1-based labels
            boxes, scores, labels = outputs[0], outputs[1], np.asarray(outputs[2]).ravel() - 1
        return _rows_to_dicts(np.asarray(boxes).reshape(-1, 4), np.asarray(scores).ravel(),
                              np.asarray(labels).ravel(), self.classes, self.size, self.size,
                              self.threshold)


def decode_ultralytics(raw: np.ndarray, threshold: float, iou: float = 0.5, top: int = 100):
    """Ultralytics' ONNX output ``[1, 4 + n_classes, n_anchors]`` (cx, cy, w, h in pixels,
    then class scores) -> boxes xyxy, scores, 0-based labels after a plain NMS."""
    p = np.asarray(raw)[0].T                                  # (anchors, 4 + classes)
    xywh, cls = p[:, :4], p[:, 4:]
    labels = cls.argmax(axis=1)
    scores = cls[np.arange(len(cls)), labels]
    keep = scores >= threshold
    xywh, scores, labels = xywh[keep], scores[keep], labels[keep]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1)
    order = np.argsort(-scores)
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= top:
            break
        ok = True
        for j in chosen:
            if labels[j] != labels[i]:
                continue
            ix0, iy0 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            ix1, iy1 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            b = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            if inter / (a + b - inter + 1e-9) > iou:
                ok = False
                break
        if ok:
            chosen.append(int(i))
    idx = np.asarray(chosen, dtype=np.int64)
    return boxes[idx], scores[idx], labels[idx]


def _ultralytics_no_autoinstall() -> None:
    """Ultralytics pip-installs missing requirements on its own (and once replaced this
    environment's numpy with an incompatible one). Off: `neurofly detect-install yolo`
    installs what it needs, including CLIP for YOLO-World."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")


class UltralyticsDetector(Detector):
    """An Ultralytics YOLO model. Their package is AGPL-3.0: installing it (the ``yolo``
    extra) puts your project under that licence's terms; the rest of neurofly does not."""
    name = "yolo"

    def __init__(self, weights: str = "yolo11n.pt", threshold: float = 0.25,
                 device: str = "cpu"):
        _ultralytics_no_autoinstall()
        from ultralytics import YOLO
        if os.path.isdir(weights):
            weights = os.path.join(weights, "detector.pt")
        self.model = YOLO(weights)
        names = self.model.names
        self.classes = ([names[i] for i in range(len(names))] if isinstance(names, dict)
                        else list(names))
        self.threshold, self.device = float(threshold), device

    def detect(self, frame: np.ndarray) -> list[dict]:
        res = self.model.predict(np.asarray(frame)[:, :, ::-1], conf=self.threshold,
                                 device=self.device, verbose=False)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return []
        boxes = b.xyxyn.cpu().numpy()
        return _rows_to_dicts(boxes, b.conf.cpu().numpy(), b.cls.cpu().numpy(), self.classes,
                              1.0, 1.0, self.threshold)


class CachedDetector(Detector):
    """Detections of a fixed video, computed once and kept in a file next to it. Frames
    are identified by their order since ``reset``."""
    name = "cached"

    def __init__(self, inner: Detector, path: str):
        self.inner, self.path = inner, path
        self.classes = list(inner.classes)
        self.rows: dict[int, list[dict]] = {}
        self._dirty = False
        self.t = 0
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            if saved.get("classes") == self.classes and saved.get("detector") == inner.name:
                self.rows = {int(k): v for k, v in saved["frames"].items()}

    def reset(self) -> None:
        self.t = 0
        self.inner.reset()

    def detect(self, frame: np.ndarray) -> list[dict]:
        t = self.t
        self.t += 1
        if t not in self.rows:
            self.rows[t] = self.inner.detect(frame)
            self._dirty = True
        return self.rows[t]

    def save(self) -> None:
        if self._dirty:
            with open(self.path, "w") as f:
                json.dump({"classes": self.classes, "detector": self.inner.name,
                           "frames": {str(k): v for k, v in self.rows.items()}}, f)
            self._dirty = False

    def close(self) -> None:
        self.save()
        self.inner.close()


class FixedDetections(Detector):
    """Detections given up front (tests, replays)."""
    name = "fixed"

    def __init__(self, classes, per_frame):
        self.classes, self.per_frame, self.t = list(classes), list(per_frame), 0

    def reset(self) -> None:
        self.t = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        d = self.per_frame[self.t % len(self.per_frame)] if self.per_frame else []
        self.t += 1
        return d


class YOLOWorldDetector(UltralyticsDetector):
    """YOLO-World v2 through Ultralytics: text prompts, real time. Setting the classes
    downloads a CLIP text encoder the first time."""
    name = "yolo-world"

    def __init__(self, prompts, weights: str = "yolov8s-worldv2.pt", threshold: float = 0.1,
                 device: str = "cpu"):
        _ultralytics_no_autoinstall()
        from ultralytics import YOLOWorld
        self.classes = [str(p).strip() for p in prompts if str(p).strip()]
        if not self.classes:
            raise ValueError("yolo-world needs at least one prompt")
        self.model = YOLOWorld(weights)
        self.model.set_classes(self.classes)
        self.threshold, self.device = float(threshold), device


class GroundingDinoDetector(Detector):
    """Grounding DINO (Apache-2.0, through ``transformers``): phrase-grounded detection.
    The prompts become one caption; a box's phrase is matched back to a prompt."""
    name = "gdino"

    def __init__(self, prompts, model_name: str = "IDEA-Research/grounding-dino-tiny",
                 threshold: float = 0.25, device: str = "cpu"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.classes = [str(p).strip().lower() for p in prompts if str(p).strip()]
        if not self.classes:
            raise ValueError("gdino needs at least one prompt")
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name) \
            .to(self.device).eval()
        self.text = ". ".join(self.classes) + "."
        self._torch = torch

    def _class_of(self, phrase) -> int | None:
        phrase = str(phrase).strip().lower()
        for i, c in enumerate(self.classes):
            if phrase == c:
                return i
        for i, c in enumerate(self.classes):
            if c in phrase or phrase in c:
                return i
        return None

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image
        img = Image.fromarray(np.asarray(frame))
        inputs = self.processor(images=img, text=self.text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.threshold, text_threshold=self.threshold,
            target_sizes=[(img.height, img.width)])[0]
        phrases = res.get("text_labels", res.get("labels"))
        out = []
        for box, score, phrase in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                                      phrases):
            c = self._class_of(phrase)
            if c is None:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            out.append({"class": c, "label": self.classes[c],
                        "box": [x0 / img.width, y0 / img.height, x1 / img.width, y1 / img.height],
                        "score": float(score)})
        return out


class TransformersDetector(Detector):
    """A detector from the ``transformers`` model hub or a ``train_transformers`` run:
    RT-DETRv2, D-FINE and any other ``AutoModelForObjectDetection``."""
    name = "transformers"

    def __init__(self, model_id: str, threshold: float | None = None, device: str = "cpu",
                 name: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.name = name or self.name
        meta = {}
        if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, DETECTOR_JSON)):
            with open(os.path.join(model_id, DETECTOR_JSON)) as f:
                meta = json.load(f)
        self.threshold = float(meta.get("threshold", 0.3) if threshold is None else threshold)
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        id2label = self.model.config.id2label
        self.classes = list(meta.get("classes") or
                            [id2label[i] for i in sorted(id2label)])
        self._torch = torch

    def detect(self, frame: np.ndarray) -> list[dict]:
        from PIL import Image
        img = Image.fromarray(np.asarray(frame))
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(img.height, img.width)])[0]
        return _rows_to_dicts(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                              res["labels"].cpu().numpy(), self.classes, img.width, img.height,
                              self.threshold)


# --- factories, one per backend (see neurofly_training.pc.backends) ------------------------------

def _thr(threshold):
    return {} if threshold is None else {"threshold": threshold}


def make_owl2(arg, device="cpu", threshold=None):
    return OpenVocabDetector(_split_prompts(arg), model_name="google/owlv2-base-patch16-ensemble",
                             device=device, **_thr(threshold))


def make_owl(arg, device="cpu", threshold=None):
    return OpenVocabDetector(_split_prompts(arg), device=device, **_thr(threshold))


def make_gdino(arg, device="cpu", threshold=None):
    return GroundingDinoDetector(_split_prompts(arg), device=device, **_thr(threshold))


def make_yolo_world(arg, device="cpu", threshold=None):
    return YOLOWorldDetector(_split_prompts(arg), device=device, **_thr(threshold))


def make_yolo(arg, device="cpu", threshold=None):
    return UltralyticsDetector(arg or "yolo11n.pt", device=device, **_thr(threshold))


def make_rtdetr(arg, device="cpu", threshold=None):
    return TransformersDetector(arg or "PekingU/rtdetr_v2_r18vd", threshold=threshold,
                                device=device, name="rtdetr")


def make_dfine(arg, device="cpu", threshold=None):
    return TransformersDetector(arg or "ustc-community/dfine-small-coco", threshold=threshold,
                                device=device, name="dfine")


def make_ssdlite(arg, device="cpu", threshold=None):
    return TorchvisionDetector(arg, threshold=threshold, device=device)


def make_onnx(arg, device="cpu", threshold=None):
    return OnnxDetector(arg, threshold=threshold)


# --- specs ------------------------------------------------------------------------------------

def _split_prompts(text: str) -> list[str]:
    """``'a, b; c . d .'`` -> ``['a', 'b', 'c', 'd']``: commas, semicolons, or the
    Grounding DINO habit of full stops between phrases."""
    return [p.strip() for p in re.split(r"[,;]|\s\.\s|\.\s*$", text) if p.strip()]


def parse_spec(spec: str) -> tuple[str, str]:
    """``'owl2:a,b'`` -> ('owl2', 'a,b'); a run directory -> (its backend, the directory).
    A backend name alone (``'dfine'`` or ``'dfine:'``) means its default weights."""
    from neurofly_training.pc import backends
    spec = str(spec).strip()
    head, sep, tail = spec.partition(":")
    key = backends.ALIASES.get(head, head)
    if key in backends.BACKENDS and not (len(head) == 1 and tail.startswith(("\\", "/"))):
        return key, tail.strip()
    if os.path.isdir(spec) and os.path.exists(os.path.join(spec, DETECTOR_JSON)):
        with open(os.path.join(spec, DETECTOR_JSON)) as f:
            b = json.load(f).get("backend", "ssdlite")
        return backends.ALIASES.get(b, b), spec
    raise ValueError(f"unknown detector {spec!r}: use backend:arg with a backend from "
                     f"`neurofly detect-list`, or a directory written by `neurofly detect-train`")


def classes_for(spec) -> list[str]:
    """The class names a spec will produce, without loading weights where possible."""
    from neurofly_training.pc import backends
    if spec is None or str(spec).lower() in ("", "none"):
        return []
    backend, arg = parse_spec(spec)
    if backends.get(backend).kind == "open-vocab":
        return _split_prompts(arg)
    if os.path.isdir(arg) and os.path.exists(os.path.join(arg, DETECTOR_JSON)):
        with open(os.path.join(arg, DETECTOR_JSON)) as f:
            return list(json.load(f)["classes"])
    return make_detector(spec).classes


def make_detector(spec, device: str = "cpu", threshold: float | None = None) -> Detector | None:
    """A ``Detector`` for a spec, or None for no spec. Raises RuntimeError with the install
    command when the backend's packages are missing."""
    from neurofly_training.pc import backends
    if spec is None or str(spec).lower() in ("", "none"):
        return None
    backend, arg = parse_spec(spec)
    b = backends.require(backend)
    return backends.resolve(b.factory)(arg, device=device, threshold=threshold)


def cached(detector: Detector | None, video_path: str | None) -> Detector | None:
    """Wrap a detector so a video file's detections are computed once."""
    if detector is None or not video_path:
        return detector
    return CachedDetector(detector, video_path + f".{detector.name}.detections.json")


# --- drawing and datasets ----------------------------------------------------------------------

def draw(frame: np.ndarray, detections: list[dict], classes=None) -> np.ndarray:
    """The frame with boxes and labels drawn on it."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(frame)).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(80, 255, 120), width=2)
        label = det.get("label") or (classes[int(det["class"])] if classes else str(det["class"]))
        d.text((x0 * w + 2, y0 * h + 1), f"{label} {det.get('score', 1.0):.2f}",
               fill=(80, 255, 120))
    return np.asarray(img)


def write_yolo_dataset(out: str, frames, detections, classes, prefix: str = "f",
                       start: int = 0) -> int:
    """Append frames and their boxes to a dataset in the YOLO layout (``images/*.png``,
    ``labels/*.txt`` with ``class cx cy w h`` in fractions, ``classes.json`` and
    ``data.yaml``), which every labelling tool can open for corrections. Returns how
    many frames were written."""
    from PIL import Image
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    n = 0
    for i, (frame, dets) in enumerate(zip(frames, detections)):
        stem = f"{prefix}_{start + i:06d}"
        Image.fromarray(np.asarray(frame)).save(os.path.join(out, "images", stem + ".png"))
        with open(os.path.join(out, "labels", stem + ".txt"), "w") as f:
            for d in dets:
                x0, y0, x1, y1 = d["box"]
                f.write(f"{int(d['class'])} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} "
                        f"{x1 - x0:.6f} {y1 - y0:.6f}\n")
        n += 1
    with open(os.path.join(out, "classes.json"), "w") as f:
        json.dump(list(classes), f)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images\nval: images\n"
                f"names: {json.dumps(list(classes))}\n")
    return n


def read_yolo_dataset(root: str):
    """``(classes, [(image path, boxes (k, 4) xyxy fractions, labels (k,))...])``."""
    with open(os.path.join(root, "classes.json")) as f:
        classes = json.load(f)
    items = []
    images = os.path.join(root, "images")
    for name in sorted(os.listdir(images)):
        stem = os.path.splitext(name)[0]
        boxes, labels = [], []
        lab = os.path.join(root, "labels", stem + ".txt")
        if os.path.exists(lab):
            for line in open(lab):
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, w, h = int(parts[0]), *[float(v) for v in parts[1:]]
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                labels.append(c)
        items.append((os.path.join(images, name), np.asarray(boxes, np.float32).reshape(-1, 4),
                      np.asarray(labels, np.int64)))
    return classes, items


# --- fine-tuning ------------------------------------------------------------------------------

def ssdlite(n_classes: int, pretrained_backbone: bool = True, size: int = 320):
    """SSDLite320 with a MobileNetV3 backbone and a fresh head for ``n_classes``
    (+ background). Real time on a CPU; BSD-licensed."""
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    kw = {"weights": None, "num_classes": n_classes + 1}
    kw["weights_backbone"] = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone \
        else None
    model = ssdlite320_mobilenet_v3_large(**kw)
    model.transform.min_size = (size,)
    model.transform.max_size = size
    model.transform.fixed_size = (size, size)
    return model


def train_torchvision(dataset: str, out: str, *, epochs: int = 20, size: int | None = 320,
                      batch: int = 8, lr: float = 1e-3, holdout: float = 0.1, seed: int = 0,
                      pretrained: bool = True, threshold: float = 0.3, device: str = "cpu",
                      export_onnx: bool = True, verbose: bool = False, **_ignored) -> dict:
    """Fine-tune SSDLite on a YOLO-layout dataset; writes ``detector.pt``, ``detector.json``
    and (when the export succeeds) ``detector.onnx`` to ``out``. Returns a history with
    the loss per epoch and the held-out loss."""
    import torch
    from PIL import Image
    torch.manual_seed(seed)
    size = int(size or 320)
    classes, items = read_yolo_dataset(dataset)
    if not items:
        raise ValueError(f"no images in {dataset}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_hold = int(len(items) * holdout)
    hold, train = [items[i] for i in order[:n_hold]], [items[i] for i in order[n_hold:]]
    if not train:
        train, hold = hold, []

    def load(item):
        path, boxes, labels = item
        img = Image.open(path).convert("RGB")
        w, h = img.size
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        b = torch.as_tensor(boxes, dtype=torch.float32) * torch.tensor([w, h, w, h])
        keep = ((b[:, 2] > b[:, 0] + 1) & (b[:, 3] > b[:, 1] + 1) if len(b)
                else torch.zeros(0, dtype=torch.bool))
        return x, {"boxes": b[keep].reshape(-1, 4),
                   "labels": torch.as_tensor(labels, dtype=torch.int64)[keep] + 1}

    dev = torch.device(device)
    model = ssdlite(len(classes), pretrained_backbone=pretrained, size=size).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "holdout": None, "classes": classes, "n_train": len(train),
               "n_holdout": len(hold)}
    for ep in range(epochs):
        model.train()
        rng.shuffle(train)
        total, nb = 0.0, 0
        for i in range(0, len(train), batch):
            xs, ts = zip(*[load(it) for it in train[i:i + batch]])
            losses = model([x.to(dev) for x in xs], [{k: v.to(dev) for k, v in t.items()}
                                                     for t in ts])
            loss = sum(losses.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        history["loss"].append(total / max(nb, 1))
        if verbose:
            print(f"epoch {ep + 1}/{epochs}  loss {history['loss'][-1]:.4f}")
    if hold:
        model.train()          # detection models only return losses in train mode
        with torch.no_grad():
            xs, ts = zip(*[load(it) for it in hold])
            losses = model([x.to(dev) for x in xs], [{k: v.to(dev) for k, v in t.items()}
                                                     for t in ts])
            history["holdout"] = float(sum(losses.values()))
    model.eval().cpu()
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "detector.pt"))
    meta = {"backend": "ssdlite", "classes": classes, "size": size, "threshold": threshold,
            "arch": "ssdlite320_mobilenet_v3_large", "onnx": False}
    if export_onnx:
        try:
            export_onnx_model(model, os.path.join(out, "detector.onnx"), size)
            meta["onnx"] = True
        except Exception as e:      # the torchvision detector still works without it
            history["onnx_error"] = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"ONNX export failed ({e}); detector.pt is still usable")
    with open(os.path.join(out, DETECTOR_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history


def export_onnx_model(model, path: str, size: int) -> None:
    """``detector.onnx`` taking ``image`` [1, 3, size, size] in [0, 1], returning boxes
    (pixels), scores and labels for that one image."""
    import torch
    model.eval()
    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(model, dummy, path, input_names=["image"],
                      output_names=["boxes", "scores", "labels"], opset_version=17,
                      dynamo=False)


def train_ultralytics(dataset: str, out: str, *, epochs: int = 20, size: int | None = 320,
                      batch: int = 8, model: str | None = None, device: str = "cpu",
                      threshold: float = 0.25, verbose: bool = False, **_ignored) -> dict:
    """Fine-tune an Ultralytics YOLO model (AGPL-3.0 package) on the same dataset and
    write ``detector.pt`` (their weights) plus ``detector.json`` and, when it succeeds,
    ``detector.onnx``."""
    import shutil
    _ultralytics_no_autoinstall()
    from ultralytics import YOLO
    classes, _ = read_yolo_dataset(dataset)
    size = int(size or 320)
    model = model or "yolo11n.pt"
    yolo = YOLO(model)
    out = os.path.abspath(out)        # a relative project would land in Ultralytics' runs dir
    os.makedirs(out, exist_ok=True)
    res = yolo.train(data=os.path.abspath(os.path.join(dataset, "data.yaml")), epochs=epochs,
                     imgsz=size, batch=batch, device=device, project=out, name="train",
                     exist_ok=True, verbose=verbose, plots=False)
    trainer = getattr(yolo, "trainer", None)
    candidates = [str(getattr(trainer, "best", "")), str(getattr(trainer, "last", "")),
                  os.path.join(out, "train", "weights", "best.pt"),
                  os.path.join(out, "train", "weights", "last.pt")]
    best = next((c for c in candidates if c and os.path.exists(c)), None)
    if best is None:
        raise RuntimeError(f"Ultralytics wrote no weights under {out}")
    shutil.copyfile(best, os.path.join(out, "detector.pt"))
    meta = {"backend": "yolo", "classes": classes, "size": size, "threshold": threshold,
            "arch": model, "onnx": False}
    try:
        exported = YOLO(best).export(format="onnx", imgsz=size, simplify=False)
        shutil.copyfile(exported, os.path.join(out, "detector.onnx"))
        meta["onnx"] = True
        meta["onnx_layout"] = "ultralytics"
    except Exception as e:
        meta["onnx_error"] = f"{type(e).__name__}: {e}"
    with open(os.path.join(out, DETECTOR_JSON), "w") as f:
        json.dump(meta, f, indent=2)
    return {"classes": classes, "results": str(getattr(res, "save_dir", out))}


def train_transformers(dataset: str, out: str, *, model_id: str, backend: str, epochs: int = 20,
                       size: int | None = 640, batch: int = 4, lr: float = 1e-4,
                       holdout: float = 0.1,
                       seed: int = 0, threshold: float = 0.3, device: str = "cpu",
                       verbose: bool = False, **_ignored) -> dict:
    """Fine-tune a ``transformers`` detector (RT-DETRv2, D-FINE) on a YOLO-layout dataset.
    Writes the model and processor to ``out`` plus ``detector.json``; the run directory is
    then a spec on its own. Returns a history with the loss per epoch."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    torch.manual_seed(seed)
    size = max(int(size or 640), 320)     # DETR-style decoders need enough tokens to pick from
    classes, items = read_yolo_dataset(dataset)
    if not items:
        raise ValueError(f"no images in {dataset}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    n_hold = int(len(items) * holdout)
    hold, train = [items[i] for i in order[:n_hold]], [items[i] for i in order[n_hold:]]
    if not train:
        train, hold = hold, []
    processor = AutoImageProcessor.from_pretrained(model_id, size={"height": size, "width": size},
                                                   do_pad=False)
    id2label = {i: c for i, c in enumerate(classes)}
    model = AutoModelForObjectDetection.from_pretrained(
        model_id, num_labels=len(classes), id2label=id2label,
        label2id={c: i for i, c in id2label.items()}, ignore_mismatched_sizes=True)
    dev = torch.device(device)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def batch_of(chunk):
        images, anns = [], []
        for k, (path, boxes, labels) in enumerate(chunk):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            images.append(img)
            anns.append({"image_id": k, "annotations": [
                {"bbox": [float(x0 * w), float(y0 * h), float((x1 - x0) * w),
                          float((y1 - y0) * h)],
                 "category_id": int(c), "area": float((x1 - x0) * w * (y1 - y0) * h),
                 "iscrowd": 0}
                for (x0, y0, x1, y1), c in zip(boxes, labels)]})
        enc = processor(images=images, annotations=anns, return_tensors="pt")
        return enc["pixel_values"].to(dev), [{k: v.to(dev) for k, v in t.items()}
                                             for t in enc["labels"]]

    history = {"loss": [], "holdout": None, "classes": classes, "n_train": len(train),
               "n_holdout": len(hold), "model_id": model_id}
    for ep in range(epochs):
        model.train()
        rng.shuffle(train)
        total, nb = 0.0, 0
        for i in range(0, len(train), batch):
            pixels, labels = batch_of(train[i:i + batch])
            loss = model(pixel_values=pixels, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        history["loss"].append(total / max(nb, 1))
        if verbose:
            print(f"epoch {ep + 1}/{epochs}  loss {history['loss'][-1]:.4f}")
    if hold:
        model.eval()
        with torch.no_grad():
            pixels, labels = batch_of(hold)
            history["holdout"] = float(model(pixel_values=pixels, labels=labels).loss)
    os.makedirs(out, exist_ok=True)
    model.cpu().save_pretrained(out)
    processor.save_pretrained(out)
    with open(os.path.join(out, DETECTOR_JSON), "w") as f:
        json.dump({"backend": backend, "classes": classes, "size": size, "threshold": threshold,
                   "arch": model_id, "onnx": False}, f, indent=2)
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history


def train_rtdetr(dataset: str, out: str, model: str | None = None, **kw) -> dict:
    return train_transformers(dataset, out, model_id=model or "PekingU/rtdetr_v2_r18vd",
                              backend="rtdetr", **kw)


def train_dfine(dataset: str, out: str, model: str | None = None, **kw) -> dict:
    return train_transformers(dataset, out, model_id=model or "ustc-community/dfine-small-coco",
                              backend="dfine", **kw)
