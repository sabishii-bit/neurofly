"""Print what is in the connectome and which populations the interfaces find.

    neurofly inspect                  # full CNS
    neurofly inspect --subset central # the PC's default brain
"""
import argparse
import time

from neurofly_training.data.connectome import SUBSETS, Connectome
from neurofly_training.data.populations import Populations
from neurofly_training.envs import DEFAULT_DATA_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--subset", default="full", choices=sorted(SUBSETS))
    p.add_argument("--synthetic", action="store_true", help="use a random test connectome")
    a = p.parse_args()

    t0 = time.time()
    cx = Connectome.synthetic() if a.synthetic else Connectome.load(a.data_dir)
    cx = cx.subset(a.subset)
    print(f"loaded in {time.time() - t0:.1f} s")
    print(cx.summary())
    print("populations used by the interfaces:")
    print(Populations(cx).summary())


if __name__ == "__main__":
    main()
