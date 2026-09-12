"""Export the brain atlas (soma positions, ids, regions) as plain files with a manifest.

    neurofly export-atlas --out assets/brain-atlas                    # the whole CNS
    neurofly export-atlas --out assets/brain-atlas --subset brain     # brain only
    neurofly export-atlas --out assets/brain-atlas --check            # verify hashes

The atlas feeds examples/workbench.html and examples/brain_viewer.html, and any model
output in the replay format (neurofly-core replay-validate) is drawn on it by bodyId.
"""
from __future__ import annotations

import argparse

from neurofly_training.atlas import check_atlas, export_atlas
from neurofly_training.data.connectome import SUBSETS
from neurofly_training.envs import DEFAULT_DATA_DIR, load_connectome


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="assets/brain-atlas")
    p.add_argument("--brain", default="malecns", choices=["malecns", "synthetic", "toy"])
    p.add_argument("--subset", default="full", choices=sorted(SUBSETS))
    p.add_argument("--synthetic-n", type=int, default=3000)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--all-neurons", action="store_true",
                   help="include neurons without a soma in the volume, at placed positions")
    p.add_argument("--check", action="store_true", help="verify an existing atlas and stop")
    args = p.parse_args()
    if args.check:
        problems = check_atlas(args.out)
        if problems:
            raise SystemExit("\n".join(f"  {x}" for x in problems))
        print(f"ok: {args.out} matches its manifest")
        return
    cx = load_connectome(args.brain, subset=args.subset, data_dir=args.data_dir,
                         synthetic_n=args.synthetic_n)
    m = export_atlas(cx, args.out, known_only=not args.all_neurons)
    counts = ", ".join(f"{k} {v:,}" for k, v in m["groups"]["counts"].items())
    print(f"wrote {args.out}: {m['n']:,} neurons ({counts})")
    print(f"bounds {m['bounds']['min']} to {m['bounds']['max']} um; {m['licence']}")


if __name__ == "__main__":
    main()
