"""The detector backends: what exists, what it needs, what it costs in licence terms.

One table drives the spec grammar (``name:arg``), ``neurofly detect-list`` and
``neurofly detect-install``. Adding a backend is one entry here plus a factory (and,
for trainable ones, a trainer) in ``neurofly_training.pc.detect``.

    neurofly detect-list                    # every backend, installed or not, with its licence
    neurofly detect-install yolo gdino      # pip install what those need, into this Python
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str            # the spec prefix: "name:arg"
    kind: str            # "open-vocab" (text prompts), "trained" (weights), "runtime" (a file)
    summary: str
    licence: str         # of the package and its default weights
    packages: tuple      # pip packages it needs
    modules: tuple       # import names that prove they are installed
    default: str = ""    # default weights or model id when the arg is empty
    trains: bool = False  # `neurofly detect-train --backend name`
    factory: str = ""    # "module:callable(arg, device, threshold) -> Detector"
    trainer: str = ""    # "module:callable(dataset, out, **options) -> dict"
    speed: str = ""      # a rough word for a CPU

    @property
    def installed(self) -> bool:
        return all(_importable(m) for m in self.modules)


def _importable(module: str) -> bool:
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


D = "neurofly_training.pc.detect"
BACKENDS: dict[str, Backend] = {b.name: b for b in [
    Backend("owl2", "open-vocab", "OWLv2: name the objects, no training; best zero-shot quality",
            "Apache-2.0", ("transformers",), ("transformers",),
            default="google/owlv2-base-patch16-ensemble", factory=f"{D}:make_owl2",
            speed="slow (about 3 s per frame)"),
    Backend("owl", "open-vocab", "OWL-ViT: like owl2, three times faster and weaker",
            "Apache-2.0", ("transformers",), ("transformers",),
            default="google/owlvit-base-patch32", factory=f"{D}:make_owl", speed="1 s per frame"),
    Backend("gdino", "open-vocab", "Grounding DINO: phrase-grounded boxes, strong on scenes",
            "Apache-2.0", ("transformers",), ("transformers",),
            default="IDEA-Research/grounding-dino-tiny", factory=f"{D}:make_gdino",
            speed="slow (several seconds per frame)"),
    Backend("yolo-world", "open-vocab", "YOLO-World v2: text prompts in real time",
            "AGPL-3.0 (package); GPL-3.0 (weights)",
            ("ultralytics", "clip @ git+https://github.com/ultralytics/CLIP.git"),
            ("ultralytics", "clip"),
            default="yolov8s-worldv2.pt", factory=f"{D}:make_yolo_world",
            speed="real time (about 50 ms per frame)"),
    Backend("yolo", "trained", "Ultralytics YOLO11: the usual choice, best tooling",
            "AGPL-3.0", ("ultralytics",), ("ultralytics",), default="yolo11n.pt", trains=True,
            factory=f"{D}:make_yolo", trainer=f"{D}:train_ultralytics",
            speed="real time (20 to 60 ms per frame)"),
    Backend("rtdetr", "trained", "RT-DETRv2: transformer detector, beats same-size YOLO",
            "Apache-2.0", ("transformers",), ("transformers",),
            default="PekingU/rtdetr_v2_r18vd", trains=True, factory=f"{D}:make_rtdetr",
            trainer=f"{D}:train_rtdetr", speed="near real time (100 to 200 ms per frame)"),
    Backend("dfine", "trained", "D-FINE: the current accuracy leader per FLOP",
            "Apache-2.0", ("transformers",), ("transformers",),
            default="ustc-community/dfine-small-coco", trains=True, factory=f"{D}:make_dfine",
            trainer=f"{D}:train_dfine", speed="near real time (100 to 200 ms per frame)"),
    Backend("ssdlite", "trained", "SSDLite MobileNetV3: the small fast baseline",
            "BSD-3-Clause", ("torchvision",), ("torchvision",), trains=True,
            factory=f"{D}:make_ssdlite", trainer=f"{D}:train_torchvision",
            speed="real time (about 20 ms per frame)"),
    Backend("onnx", "runtime", "an exported detector.onnx through ONNX Runtime (any language)",
            "MIT (runtime); the model's own", ("onnxruntime",), ("onnxruntime",),
            factory=f"{D}:make_onnx", speed="as the model"),
]}
ALIASES = {"torchvision": "ssdlite", "owlv2": "owl2", "yoloworld": "yolo-world",
           "grounding-dino": "gdino"}


def get(name: str) -> Backend:
    key = ALIASES.get(name, name)
    if key not in BACKENDS:
        raise ValueError(f"unknown detector backend {name!r}; choose from {sorted(BACKENDS)} "
                         f"(neurofly detect-list)")
    return BACKENDS[key]


def resolve(dotted: str):
    module, _, attr = dotted.partition(":")
    return getattr(importlib.import_module(module), attr)


def missing(name: str) -> list[str]:
    """Packages a backend needs that are not installed."""
    b = get(name)
    return [] if b.installed else list(b.packages)


def require(name: str) -> Backend:
    b = get(name)
    if not b.installed:
        raise RuntimeError(f"detector backend {b.name!r} needs {', '.join(b.packages)}: "
                           f"run `neurofly detect-install {b.name}` (licence: {b.licence})")
    return b


def install(names, upgrade: bool = False, dry_run: bool = False) -> list[str]:
    """pip install what the named backends need, into the running Python. Returns the
    package list. With ``dry_run`` nothing is installed."""
    packages: list[str] = []
    for n in names:
        for p in get(n).packages:
            if p not in packages:
                packages.append(p)
    if not packages:
        return []
    cmd = [sys.executable, "-m", "pip", "install", *(["--upgrade"] if upgrade else []), *packages]
    if not dry_run:
        subprocess.check_call(cmd)
    return packages


def describe() -> str:
    rows = [("backend", "kind", "installed", "licence", "speed on a CPU", "what it is")]
    for b in BACKENDS.values():
        rows.append((b.name, b.kind, "yes" if b.installed else "no", b.licence, b.speed,
                     b.summary + (" (trainable)" if b.trains else "")))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = []
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r[:5], widths)) + "  " + r[5])
    lines.append("")
    lines.append("spec: backend:arg, e.g. owl2:enemy,health pack | yolo:yolo11n.pt | "
                 "yolo:runs/det1 | dfine: (default weights) | a detect-train run directory")
    lines.append("install: neurofly detect-install <backend...>")
    return "\n".join(lines)
