import numpy as np
import torch

from flybrain_body.brain.lif import LIFBrain
from flybrain_body.brain.plasticity import DopamineHebbian


def test_silent_without_input(synthetic_cx):
    brain = LIFBrain(synthetic_cx.W, dt=0.5)
    counts = brain.run(100)
    assert counts.sum() == 0
    assert torch.allclose(brain.v, torch.full_like(brain.v, brain.v_rest))


def test_driven_neurons_fire_at_analytic_rate():
    import scipy.sparse as sp
    brain = LIFBrain(sp.csr_matrix((50, 50), dtype=np.float32), dt=0.1)  # no synapses
    idx = np.arange(10)
    drive = brain.drive(idx, 20.0)  # 20 mV above rest; threshold is 7 mV above rest
    counts = brain.run(int(500 / brain.dt), drive)  # 500 ms
    rates = counts[:10].cpu().numpy() / 0.5
    # analytic LIF rate: 1 / (t_ref + tau_m * ln(I / (I - (v_th - v_rest))))
    expected = 1000.0 / (brain.t_ref + brain.tau_m * np.log(20.0 / 13.0))
    assert np.all(np.abs(rates - expected) / expected < 0.1)
    assert counts[10:].sum() == 0
    assert brain.rates(idx).mean() > 0


def test_activity_propagates_through_synapses(synthetic_cx):
    cx = synthetic_cx
    brain = LIFBrain(cx.W, dt=0.5)
    # A handful of driven neurons deliver synaptic input to their targets but
    # cannot recruit them (a few synapses are far below the 7 mV threshold).
    idx = np.arange(10)
    brain.run(100, brain.drive(idx, 20.0))
    targets = np.setdiff1d(cx.neighborhood(idx, hops=1, direction="down"), idx)
    assert brain.i_syn[torch.as_tensor(targets)].abs().sum() > 0
    # The random test graph is ~10x sparser than the real nerve cord, so a
    # gain is needed for broad convergent drive to recruit downstream neurons.
    strong = LIFBrain(cx.W, dt=0.5, gain=10.0)
    driven = np.arange(cx.n // 3)
    counts = strong.run(400, strong.drive(driven, 20.0))  # 200 ms
    assert counts[len(driven):].sum() > 0, "downstream neurons should be recruited"


def test_reset_clears_state(synthetic_cx):
    brain = LIFBrain(synthetic_cx.W)
    brain.run(50, brain.drive([0, 1, 2], 30.0))
    assert brain.total_spikes > 0
    brain.reset()
    assert brain.total_spikes == 0 and brain.t == 0 and brain.rate.sum() == 0


def test_plasticity_changes_weights_only_with_dopamine(synthetic_cx):
    cx = synthetic_cx
    brain = LIFBrain(cx.W, dt=0.5)
    pre = np.arange(cx.n)
    post = cx.select(superclass="vnc_motor")
    rule = DopamineHebbian(brain, pre, post, lr=0.1)
    assert rule.n_edges > 0
    w0 = brain.vals.clone()
    drive = brain.drive(np.arange(200), 25.0)
    for _ in range(100):
        brain.step(drive)
        rule.step(0.0)
    assert torch.equal(brain.vals, w0)
    for _ in range(100):
        brain.step(drive)
        rule.step(1.0)
    assert not torch.equal(brain.vals, w0)
    # signs preserved, magnitudes bounded
    changed = brain.vals[rule.e]
    assert torch.all(torch.sign(changed) * rule.sign >= 0)
    assert torch.all(changed.abs() <= rule.w_max + 1e-5)
