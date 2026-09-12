import os

import pytest

from neurofly_training.data.connectome import Connectome
from neurofly_training.envs import DEFAULT_DATA_DIR


@pytest.fixture(scope="session")
def synthetic_cx():
    return Connectome.synthetic(n=1500, seed=0)


def has_real_data() -> bool:
    return os.path.exists(os.path.join(DEFAULT_DATA_DIR, "connectome-weights-traced-only.feather"))


needs_data = pytest.mark.skipif(not has_real_data(), reason="MaleCNS data not downloaded")
