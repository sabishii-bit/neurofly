"""List the detector backends, or install what some of them need.

    neurofly detect-list
    neurofly detect-install yolo yolo-world     # pip install into this Python; note the licences
    neurofly detect-install dfine --dry-run     # only say what would be installed
"""
from __future__ import annotations

import argparse
import sys

from neurofly_training.pc import backends


def main():
    listing = "detect-list" in sys.argv[0] or (len(sys.argv) > 1 and sys.argv[1] == "--list")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="backends to install (see detect-list)")
    p.add_argument("--list", action="store_true", help="list the backends instead")
    p.add_argument("--upgrade", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the packages, install nothing")
    args = p.parse_args()
    if listing or args.list or not args.names:
        print(backends.describe())
        return
    for n in args.names:
        b = backends.get(n)
        print(f"{b.name}: {', '.join(b.packages)}  [{b.licence}]"
              + ("  (already installed)" if b.installed else ""))
    packages = backends.install(args.names, upgrade=args.upgrade, dry_run=args.dry_run)
    if args.dry_run:
        print("would install: " + " ".join(packages))
    else:
        print("installed: " + " ".join(packages))


if __name__ == "__main__":
    main()
