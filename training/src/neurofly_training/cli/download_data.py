"""Download the MaleCNS v1.0 flat connectome (about 570 MB, no login needed).

    neurofly download
    neurofly download --data-dir D:/connectomes/malecns
"""
import argparse

from neurofly_training.data.download import download
from neurofly_training.envs import DEFAULT_DATA_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    download(a.data_dir, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
