"""The brain atlas export and the replay format."""
import json
import os
import sys

import numpy as np
import pytest

from neurofly_core import replay as R
from neurofly_training.atlas import GROUPS, check_atlas, export_atlas, group_of, load_atlas
from neurofly_training.data.connectome import Connectome


def test_group_of():
    assert [group_of(s) for s in ("ol_intrinsic", "cb_sensory", "descending_neuron",
                                  "vnc_motor", "weird", None)] == [0, 1, 2, 3, 4, 4]


def test_export_check_and_load(synthetic_cx, tmp_path):
    m = export_atlas(synthetic_cx, str(tmp_path / "atlas"))
    known = synthetic_cx.positions(fill=False)[1]
    assert m["n"] == int(known.sum()) and m["format"] == "neurofly-atlas"
    assert set(m["groups"]["counts"]) == set(GROUPS) and m["groups"]["counts"]["vnc"] > 0
    assert len(m["sha256"]) == 4 and "CC BY 4.0" in m["licence"]
    assert (tmp_path / "atlas" / "NOTICE.md").exists()
    assert check_atlas(str(tmp_path / "atlas")) == []
    a = load_atlas(str(tmp_path / "atlas"))
    assert a["positions"].shape == (m["n"], 3) and a["ids"].shape == (m["n"],)
    assert a["groups"].max() <= 4 and len(a["superclass"]) == m["n"]
    assert np.isfinite(a["positions"]).all()
    # every neuron, placed ones included and flagged
    m2 = export_atlas(synthetic_cx, str(tmp_path / "all"), known_only=False)
    assert m2["n"] == synthetic_cx.n and sum(m2["known"]) == m["n"]
    # tampering is caught
    with open(tmp_path / "atlas" / "groups.bin", "r+b") as f:
        f.write(b"\x04")
    problems = check_atlas(str(tmp_path / "atlas"))
    assert any("groups.bin" in p and "SHA-256" in p for p in problems)
    os.remove(tmp_path / "atlas" / "ids.bin")
    assert any("ids.bin: missing" in p for p in check_atlas(str(tmp_path / "atlas")))
    assert check_atlas(str(tmp_path / "nowhere"))[0].startswith("manifest")
    json.dump({"format": "x"}, open(tmp_path / "bad.json", "w"))
    (tmp_path / "bad").mkdir()
    json.dump({"format": "x"}, open(tmp_path / "bad" / "manifest.json", "w"))
    assert "not a neurofly-atlas" in check_atlas(str(tmp_path / "bad"))[0]


def test_export_atlas_cli(tmp_path, monkeypatch, capsys):
    from neurofly_training.cli import export_atlas as cli
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "--out", str(tmp_path / "a"), "--brain",
                                      "toy", "--synthetic-n", "800", "--subset", "central"])
    cli.main()
    out = capsys.readouterr().out
    assert "wrote" in out and "central" in out and "CC BY" in out
    monkeypatch.setattr(sys, "argv", ["neurofly-test", "--out", str(tmp_path / "a"), "--check"])
    cli.main()
    assert "ok:" in capsys.readouterr().out
    (tmp_path / "a" / "positions.bin").write_bytes(b"\x00")
    with pytest.raises(SystemExit):
        cli.main()


def _activity(model, steps=4):
    from neurofly_core.activity import ActivityLog
    model.watch_activity(True)
    log = ActivityLog(model, None, fps=10)
    model.reset()
    for k in range(steps):
        model.observe(np.full((6, 8, 3), 60 * (k % 2), np.uint8))
        log.record()
    from neurofly_core.activity import map_payload
    return {**map_payload(model, 10), "steps": log.steps}


def test_replay_from_activity_and_validation(tmp_path):
    from neurofly_core.controls import ControlLayout
    from neurofly_training.build import build_model
    cx = Connectome.toy(800, seed=0).subset("central")
    model = build_model(cx, ControlLayout(keys=["w"]), brain_ms=10)
    act = _activity(model)
    assert act["ids"] and len(act["ids"]) == cx.n
    rep = R.from_activity(act, name="toy")
    assert rep["version"] == 1 and len(rep["frames"]) == 4 and rep["frames"][0]["time"] == 0
    assert rep["frames"][1]["time"] == 0.1 and rep["source"]["kind"] == "predicted"
    ids = set(cx.neurons["bodyId"].values.tolist())
    for fr in rep["frames"]:
        assert all(i in ids and 0 < v <= 1 for i, v in fr["values"])
    assert R.validate_replay(rep, known_ids=cx.neurons["bodyId"].values) == []
    assert R.frame_at(rep, 0.25)["time"] == 0.2 and R.frame_at(rep, -1) is None
    path = R.save(rep, str(tmp_path / "r.json"))
    assert json.load(open(path))["frames"]
    # every rule of the validator
    bad = json.loads(json.dumps(rep))
    bad["version"] = 2
    bad["source"]["kind"] = "guess"
    assert len(R.validate_replay(bad)) == 2
    assert R.validate_replay({"version": 1, "source": {"kind": "synthetic"}, "frames": []})
    late = json.loads(json.dumps(rep))
    late["frames"][0]["time"] = 0.5
    late["frames"][2]["time"] = 0.05
    p = R.validate_replay(late)
    assert any("time 0" in x for x in p) and any("does not increase" in x for x in p)
    dup = json.loads(json.dumps(rep))
    first = dup["frames"][1]["values"][0][0]
    dup["frames"][1]["values"].append([first, 0.5])
    assert any("repeated" in x for x in R.validate_replay(dup))
    out = json.loads(json.dumps(rep))
    out["frames"][1]["values"][0][1] = 1.5
    assert any("outside" in x for x in R.validate_replay(out))
    unknown = json.loads(json.dumps(rep))
    unknown["frames"][1]["values"][0][0] = 999999999
    assert any("not in the atlas" in x for x in R.validate_replay(unknown, known_ids=[1, 2]))
    assert any("MB" in x for x in R.validate_replay(rep, n_bytes=R.MAX_BYTES + 1))
    shape = json.loads(json.dumps(rep))
    shape["frames"][1]["values"][0] = [1]
    assert any("not [id, value]" in x for x in R.validate_replay(shape))
    del act["ids"]
    with pytest.raises(ValueError):
        R.from_activity(act)


def test_replay_cli_and_ids(tmp_path, monkeypatch, capsys):
    from neurofly_core import cli as core_cli
    from neurofly_core.artifact import save_model
    from neurofly_core.controls import ControlLayout
    from neurofly_training.build import build_model
    cx = Connectome.toy(800, seed=0).subset("central")
    model = build_model(cx, ControlLayout(keys=["w"]), brain_ms=10)
    art = save_model(model, str(tmp_path / "art"))
    json.dump(_activity(model), open(tmp_path / "act.json", "w"))
    core_cli.main(["replay-export", str(tmp_path / "act.json"), "--out",
                   str(tmp_path / "rep.json"), "--name", "toy-run"])
    assert "4 frames" in capsys.readouterr().out
    core_cli.main(["replay-validate", str(tmp_path / "rep.json"), "--atlas", art])
    assert "ok" in capsys.readouterr().out
    assert len(R.load_ids(art)) == cx.n
    export_atlas(cx, str(tmp_path / "atlas"))
    core_cli.main(["replay-validate", str(tmp_path / "rep.json"), "--atlas",
                   str(tmp_path / "atlas")])
    assert "ok" in capsys.readouterr().out
    rep = json.load(open(tmp_path / "rep.json"))
    rep["frames"][0]["values"] = [[42424242, 0.5]]
    json.dump(rep, open(tmp_path / "rep.json", "w"))
    with pytest.raises(SystemExit):
        core_cli.main(["replay-validate", str(tmp_path / "rep.json"), "--atlas", art])
    with pytest.raises(FileNotFoundError):
        R.load_ids(str(tmp_path))
