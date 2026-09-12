"""Fine-tune a fast object detector on a dataset from `neurofly detect-label`.

    neurofly detect-train data/objects --out runs/det1                   # SSDLite: BSD, fastest
    neurofly detect-train data/objects --out runs/det1 --backend dfine   # D-FINE: Apache, strongest
    neurofly detect-train data/objects --out runs/det1 --backend yolo    # Ultralytics: AGPL

Then the run directory is the detector anywhere: `--detect runs/det1`. `neurofly
detect-list` shows every trainable backend with its licence and speed.
"""
from __future__ import annotations

import argparse
import json
import os

from neurofly_training.pc import backends


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="directory written by neurofly detect-label")
    p.add_argument("--out", required=True, help="run directory to write the detector to")
    p.add_argument("--backend", default="ssdlite",
                   choices=[b.name for b in backends.BACKENDS.values() if b.trains])
    p.add_argument("--model", default=None,
                   help="weights or model id to start from (the backend's default otherwise)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=None,
                   help="input square, pixels (default: 320 for ssdlite and yolo, 640 for "
                        "rtdetr and dfine)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.3, help="confidence cut at use")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true",
                   help="torchvision: start the backbone from scratch (no download)")
    p.add_argument("--no-onnx", action="store_true", help="skip the ONNX export")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    b = backends.require(args.backend)
    train = backends.resolve(b.trainer)
    history = train(args.dataset, args.out, epochs=args.epochs, size=args.size, batch=args.batch,
                    lr=args.lr, holdout=args.holdout, seed=args.seed,
                    pretrained=not args.no_pretrained, threshold=args.threshold,
                    device=args.device, export_onnx=not args.no_onnx, model=args.model,
                    verbose=True)
    if history.get("loss"):
        loss = history["loss"]
        print(f"loss {loss[0]:.4f} -> {loss[-1]:.4f} on {history.get('n_train', '?')} images"
              + (f"; holdout {history['holdout']:.4f}" if history.get("holdout") is not None
                 else "")
              + (f"; ONNX export failed: {history['onnx_error']}"
                 if "onnx_error" in history else ""))
    with open(os.path.join(args.out, "detector.json")) as f:
        meta = json.load(f)
    print(f"detector -> {args.out} ({meta['backend']}, classes {meta['classes']}, "
          f"onnx {'yes' if meta.get('onnx') else 'no'}); use it with --detect {args.out}")


if __name__ == "__main__":
    main()
