import numpy as np

from neurofly_core.encode.vision import RetinaEncoder, hex_to_unit_square, luminance
from neurofly_training.build import build_retina
from neurofly_training.data.populations import Populations


def test_luminance_grid():
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, 80:] = 255
    lum = luminance(frame, (12, 16))
    assert lum.shape == (12, 16) and lum.dtype == np.float32
    assert lum[:, :8].max() == 0.0 and lum[:, 8:].min() == 1.0


def test_hex_to_unit_square_spans_unit_square():
    hexes = np.array([[p, q] for p in range(1, 9) for q in range(1, 9)], float)
    uv = hex_to_unit_square(hexes)
    assert uv.min() >= 0 and uv.max() <= 1
    assert np.allclose(uv.min(0), 0) and np.allclose(uv.max(0), 1)


def test_retina_is_lateralised_and_has_on_off_channels(synthetic_cx):
    pops = Populations(synthetic_cx)
    ret = pops.retina()
    assert len(ret["L"]["idx"]) > 0 and len(ret["R"]["idx"]) > 0
    enc = build_retina(pops, synthetic_cx.n, gain=10.0)
    assert enc.mode == "hex" and enc.n_driven == pops.n_retina
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:, :80] = 255  # left half bright -> left eye sees light, right eye sees dark
    d = enc(frame).numpy()
    for side, bright in (("L", True), ("R", False)):
        idx, on = ret[side]["idx"], ret[side]["on"]
        on_drive, off_drive = d[idx[on]].mean(), d[idx[~on]].mean()
        if bright:
            assert on_drive == 10.0 and off_drive == 0.0
        else:
            assert on_drive == 0.0 and off_drive == 10.0
    # only the ON cells of the left eye and the OFF cells of the right eye are driven
    assert np.count_nonzero(d) == ret["L"]["on"].sum() + (~ret["R"]["on"]).sum()
    # the tables round-trip
    again = RetinaEncoder.from_tables(synthetic_cx.n, enc.params(), enc.tables())
    assert np.array_equal(again(frame).numpy(), d)


def test_retina_temporal_mode_responds_to_change(synthetic_cx):
    pops = Populations(synthetic_cx)
    enc = build_retina(pops, synthetic_cx.n, temporal=1.0)
    frame = np.full((60, 80, 3), 128, np.uint8)
    enc(frame)
    assert enc(frame).sum() == 0  # nothing changed
    assert enc(np.full((60, 80, 3), 255, np.uint8)).sum() > 0


def test_projection_fallback_without_optic_lobe(synthetic_cx):
    cx = synthetic_cx.subset("central")
    pops = Populations(cx)
    assert pops.n_retina == 0 and len(pops.visual_projection) > 0
    enc = build_retina(pops, cx.n)
    assert enc.mode == "projection" and enc.n_driven == len(pops.visual_projection)
    d = enc(np.full((60, 80, 3), 255, np.uint8)).numpy()
    assert d.sum() > 0 and np.all(d[np.setdiff1d(np.arange(cx.n), pops.visual_projection)] == 0)
    again = RetinaEncoder.from_tables(cx.n, enc.params(), enc.tables())
    assert np.allclose(again(np.full((60, 80, 3), 255, np.uint8)).numpy(), d)
