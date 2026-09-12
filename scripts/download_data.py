"""Download the MaleCNS v1.0 flat connectome (about 570 MB, no login needed)."""
import argparse

from flybrain_body.data.download import download
from flybrain_body.envs import DEFAULT_DATA_DIR

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    download(a.data_dir, overwrite=a.overwrite)
