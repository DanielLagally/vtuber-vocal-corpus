"""offload_raw_audio: upload a video's raw wav to Drive and delete the
local copy — but ONLY after a confirmed-successful upload. Never delete
without a confirmed-good upload; never invent an upload for a file that
isn't there.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from vanalysis import offload
from vanalysis.fetch import audio_path


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-wav-bytes")


class _FakeUploadRunner:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> None:
        self.calls.append(list(argv))
        if self.fail:
            raise subprocess.CalledProcessError(1, argv)


def test_offload_deletes_local_copy_after_successful_upload(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raw_wav = audio_path("okid0000001", data_dir)
    _write_wav(raw_wav)
    runner = _FakeUploadRunner()

    result = offload.offload_raw_audio("okid0000001", data_dir, runner=runner)

    assert result is True
    assert not raw_wav.exists()
    assert len(runner.calls) == 1


def test_offload_keeps_local_copy_when_upload_fails(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    raw_wav = audio_path("failid00001", data_dir)
    _write_wav(raw_wav)
    runner = _FakeUploadRunner(fail=True)

    result = offload.offload_raw_audio("failid00001", data_dir, runner=runner)

    assert result is False
    assert raw_wav.exists()


def test_offload_is_a_noop_when_no_local_raw_audio(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    runner = _FakeUploadRunner()

    result = offload.offload_raw_audio("neveraudio1", data_dir, runner=runner)

    assert result is False
    assert runner.calls == []
