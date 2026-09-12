"""The walking dataset download: streaming, resuming, figshare's 202, and the unpack."""
import io
import os
import zipfile

import pytest

from neurofly_training.cli import body_replay


class FakeResponse:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def iter_content(self, chunk):
        for i in range(0, len(self._body), chunk):
            yield self._body[i:i + chunk]

    def close(self):
        pass


def fake_requests(monkeypatch, responses):
    calls = []

    class R:
        @staticmethod
        def get(url, stream=True, headers=None, timeout=None):
            calls.append(dict(headers or {}))
            return responses.pop(0)
    monkeypatch.setitem(__import__("sys").modules, "requests", R)
    monkeypatch.setattr(body_replay.time, "sleep", lambda s: None)
    return calls


def test_download_retries_202_then_streams(tmp_path, monkeypatch):
    data = os.urandom(3000)
    calls = fake_requests(monkeypatch, [
        FakeResponse(202),
        FakeResponse(200, data, {"Content-Length": str(len(data))}),
    ])
    logs = []
    out = body_replay.download("http://x/f", str(tmp_path / "d" / "f.zip"), chunk=1024,
                               log=logs.append)
    assert open(out, "rb").read() == data and len(calls) == 2
    assert not os.path.exists(out + ".part")
    assert any("staging" in line for line in logs)


def test_download_resumes_partial_file(tmp_path, monkeypatch):
    data = os.urandom(5000)
    path = tmp_path / "f.zip"
    (tmp_path / "f.zip.part").write_bytes(data[:2000])
    calls = fake_requests(monkeypatch, [
        FakeResponse(206, data[2000:], {"Content-Length": str(3000)}),
    ])
    body_replay.download("http://x/f", str(path), chunk=1000, log=lambda s: None)
    assert path.read_bytes() == data and calls[0] == {"Range": "bytes=2000-"}


def test_download_fails_loudly(tmp_path, monkeypatch):
    fake_requests(monkeypatch, [FakeResponse(404)])
    with pytest.raises(SystemExit, match="HTTP 404"):
        body_replay.download("http://x/f", str(tmp_path / "f.zip"), log=lambda s: None)


def test_download_walking_dataset_unpacks_and_finds_h5(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("inner/walking-dataset_full.hdf5", b"not really hdf5")
        z.writestr("inner/walking-dataset-small.hdf5", b"not really hdf5")
    fake_requests(monkeypatch, [FakeResponse(200, buf.getvalue(),
                                             {"Content-Length": str(len(buf.getvalue()))})])
    found = body_replay.download_walking_dataset(str(tmp_path), log=lambda s: None)
    assert found and found.endswith("walking-dataset-small.hdf5") and os.path.exists(found)
    assert not os.path.exists(tmp_path / body_replay.WALKING_ZIP)          # zip removed
