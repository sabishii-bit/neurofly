"""Print what is in the connectome and which populations the body interface finds.

    python scripts/inspect_connectome.py                # full CNS
    python scripts/inspect_connectome.py --subset vnc   # nerve cord subset
"""
import argparse
import time

from flybrain_body.data.connectome import SUBSETS, Connectome
from flybrain_body.envs import DEFAULT_DATA_DIR
from flybrain_body.interface.populations import Populations

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--subset", default="full", choices=sorted(SUBSETS))
    p.add_argument("--synthetic", action="store_true", help="use a random test connectome")
    a = p.parse_args()

    t0 = time.time()
    cx = Connectome.synthetic() if a.synthetic else Connectome.load(a.data_dir)
    cx = cx.subset(a.subset)
    print(f"loaded in {time.time() - t0:.1f} s")
    print(cx.summary())
    print("populations used by the body interface:")
    print(Populations(cx).summary())
