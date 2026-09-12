import numpy as np

from flybrain_body.data.connectome import Connectome
from flybrain_body.envs import DEFAULT_DATA_DIR
from flybrain_body.interface.populations import Populations
from tests.conftest import needs_data


def test_synthetic_structure(synthetic_cx):
    cx = synthetic_cx
    assert cx.W.shape == (cx.n, cx.n)
    assert cx.n_edges > 0
    # sign of every outgoing edge matches the presynaptic neuron's transmitter
    Wc = cx.W.tocsc()
    for pre in range(0, cx.n, 97):
        col = Wc[:, pre].data
        if len(col):
            assert np.all(np.sign(col) == cx.sign[pre])


def test_select_subgraph_neighborhood(synthetic_cx):
    cx = synthetic_cx
    motor = cx.select(superclass="vnc_motor")
    assert len(motor) > 0
    t1l = cx.select(superclass="vnc_motor", somaNeuromere="T1", somaSide="L")
    assert set(t1l) <= set(motor)
    assert len(cx.select(superclass_re="^vnc_")) >= len(motor)
    sub = cx.subgraph(motor)
    assert sub.n == len(motor)
    hood = cx.neighborhood(motor[:5], hops=1)
    assert set(motor[:5]) <= set(hood) and len(hood) > 5
    assert cx.subset("vnc").n < cx.n


def test_populations_find_every_leg(synthetic_cx):
    pops = Populations(synthetic_cx)
    for key, idx in pops.leg_motor.items():
        assert len(idx) > 0, key
    for key, idx in pops.leg_proprio.items():
        assert len(idx) > 0, key
    assert len(pops.descending) > 0
    assert len(pops.readout("motor+descending")) == len(pops.all_leg_motor) + len(pops.descending)


@needs_data
def test_real_connectome_loads_and_caches():
    cx = Connectome.load(DEFAULT_DATA_DIR, verbose=False)
    assert cx.n == 165122
    assert cx.n_edges > 25_000_000
    vnc = cx.subset("vnc")
    pops = Populations(vnc)
    assert all(len(v) > 50 for v in pops.leg_motor.values())   # 75 to 88 per leg
    assert len(pops.descending) > 1000
