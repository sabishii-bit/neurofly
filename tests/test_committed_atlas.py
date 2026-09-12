"""The brain atlas committed under assets/brain-atlas: the one the workbench loads by default."""
import json
import os

import numpy as np
import pytest

ATLAS = os.path.join(os.path.dirname(__file__), "..", "assets", "brain-atlas")

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ATLAS, "manifest.json")),
                                reason="no committed atlas")


def test_committed_atlas_is_intact_and_brain_sized():
    from neurofly_training.atlas import check_atlas
    assert check_atlas(ATLAS) == []
    m = json.load(open(os.path.join(ATLAS, "manifest.json")))
    assert m["dataset"] == "male-cns:v1.0" and m["connectome"].endswith(":brain")
    ids = np.fromfile(os.path.join(ATLAS, "ids.bin"), "<i8")
    groups = np.fromfile(os.path.join(ATLAS, "groups.bin"), "u1")
    pos = np.fromfile(os.path.join(ATLAS, "positions.bin"), "<f4").reshape(-1, 3)
    assert 100_000 < len(ids) == len(groups) == len(pos) == m["n"] < 200_000
    assert len(np.unique(ids)) == len(ids)                              # bodyIds are unique
    assert (groups == 3).sum() == 0                                     # no VNC in the brain atlas
    assert np.isfinite(pos).all() and pos.min() > 0 and pos.max() < 1500   # micrometres
    assert os.path.getsize(os.path.join(ATLAS, "superclass.json")) < 3_000_000
    assert "CC BY 4.0" in open(os.path.join(ATLAS, "NOTICE.md"), encoding="utf-8").read()


def test_a_replay_against_the_committed_atlas_validates():
    from neurofly_core.replay import load_ids, validate_replay
    ids = load_ids(ATLAS)
    good = {"version": 1, "dataset": "male-cns:v1.0", "source": {"kind": "synthetic", "name": "t"},
            "frames": [{"time": 0, "values": [[int(ids[0]), 0.5]]},
                       {"time": 0.1, "values": [[int(ids[1]), 1.0]]}]}
    assert validate_replay(good, ids) == []
    bad = dict(good, frames=[{"time": 0, "values": [[123, 0.5]]}, {"time": 0.1, "values": []}])
    assert any("not in the atlas" in p for p in validate_replay(bad, ids))
