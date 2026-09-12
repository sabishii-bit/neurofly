"""Download the MaleCNS v1.0 flat connectome from Janelia's public bucket.

No account is needed. Three files, about 570 MB total:
  * body annotations (14 MB): neuron classes, types, sides, neuromeres
  * body neurotransmitters (43 MB): predicted transmitter per neuron
  * connectome weights, traced-only (508 MB): synapse counts per neuron pair
"""
from __future__ import annotations

import os

import requests

BASE_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome/")

# local file name -> remote file name
FILES = {
    "body-annotations.feather":
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters.feather":
        "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-traced-only.feather":
        "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}


def download(data_dir: str, overwrite: bool = False, chunk_mb: int = 8) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for local, remote in FILES.items():
        dest = os.path.join(data_dir, local)
        if os.path.exists(dest) and not overwrite:
            print(f"exists, skipping: {dest}")
            continue
        url = BASE_URL + remote
        print(f"downloading {url}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(dest + ".part", "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"  {done / 1e6:8.0f} / {total / 1e6:.0f} MB", end="\r")
        os.replace(dest + ".part", dest)
        print(f"\n  saved {dest}")
