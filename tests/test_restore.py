"""Restoring clips whose local audio is gone, from the archive.

Roughly a quarter of measured clips no longer have local audio; most of them
are in the Drive archive as the original 15-minute fetch. Each record stores
the window offsets that were cut from it, so replaying those offsets
reproduces the exact clip the record was measured from.

That exactness is the point. Re-fetching the stream and hunting a window again
would yield a *different* 90 seconds, and a v1-vs-v2 difference would then
confound a new method with a new sample. Replaying the stored window holds the
sample fixed so the method is the only thing that changed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vvc import restore


@pytest.fixture
def records():
    return [
        {"id": "aaa", "window": {"start_s": 690.0, "end_s": 780.0}},
        {"id": "bbb", "window": {"start_s": 15.0, "end_s": 105.0}},
    ]


class TestPlan:
    def test_skips_clips_that_already_have_local_audio(self, tmp_path, records):
        windows = tmp_path / "windows"
        windows.mkdir()
        (windows / "aaa_raw90.wav").touch()
        plan = restore.plan_restore(records, tmp_path, available={"aaa", "bbb"})
        assert [row["id"] for row in plan] == ["bbb"]

    def test_skips_clips_the_archive_does_not_have(self, tmp_path, records):
        plan = restore.plan_restore(records, tmp_path, available={"aaa"})
        assert [row["id"] for row in plan] == ["aaa"]

    def test_carries_the_stored_window_offsets(self, tmp_path, records):
        plan = restore.plan_restore(records, tmp_path, available={"aaa"})
        assert plan[0]["start_s"] == 690.0
        assert plan[0]["end_s"] == 780.0

    def test_a_record_without_a_window_cannot_be_restored_exactly(self, tmp_path):
        """Guessing a window would silently produce a different clip."""
        plan = restore.plan_restore([{"id": "ccc"}], tmp_path, available={"ccc"})
        assert plan == []


class TestArchiveIds:
    def test_strips_the_extension_to_recover_video_ids(self):
        got = restore.archive_ids(lister=lambda remote: "aaa.wav\nbbb.wav\n\n")
        assert got == {"aaa", "bbb"}

    def test_an_empty_archive_is_not_an_error(self):
        assert restore.archive_ids(lister=lambda remote: "") == set()


class TestRun:
    def _runner(self, calls, *, fail_download=False):
        def run(argv):
            calls.append(argv)
            if argv[0] == "rclone":
                if fail_download:
                    raise RuntimeError("network")
                Path(argv[-1]).mkdir(parents=True, exist_ok=True)
                (Path(argv[-1]) / "aaa.wav").write_bytes(b"RAW")
            else:  # ffmpeg
                Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(argv[-1]).write_bytes(b"CLIP")
        return run

    def test_writes_the_window_slice_and_removes_the_raw_download(self, tmp_path):
        calls: list[list[str]] = []
        summary = restore.run_restore(
            [{"id": "aaa", "start_s": 690.0, "end_s": 780.0}],
            tmp_path,
            runner=self._runner(calls),
        )
        assert (tmp_path / "windows" / "aaa_raw90.wav").is_file()
        assert not (tmp_path / "restore_tmp" / "aaa.wav").exists()
        assert summary["restored"] == 1

    def test_passes_the_stored_offsets_to_the_cut(self, tmp_path):
        calls: list[list[str]] = []
        restore.run_restore(
            [{"id": "aaa", "start_s": 690.0, "end_s": 780.0}],
            tmp_path,
            runner=self._runner(calls),
        )
        ffmpeg = next(c for c in calls if c[0] == "ffmpeg")
        assert "690.0" in ffmpeg and "780.0" in ffmpeg

    def test_a_failed_download_skips_and_continues(self, tmp_path):
        """One unreachable file must not abort a long restore batch."""
        calls: list[list[str]] = []
        summary = restore.run_restore(
            [{"id": "aaa", "start_s": 0.0, "end_s": 90.0}],
            tmp_path,
            runner=self._runner(calls, fail_download=True),
        )
        assert summary["restored"] == 0
        assert summary["failed"] == 1

    def test_never_overwrites_an_existing_window(self, tmp_path):
        """A restored clip must not silently replace audio already on disk."""
        windows = tmp_path / "windows"
        windows.mkdir()
        target = windows / "aaa_raw90.wav"
        target.write_bytes(b"ORIGINAL")
        calls: list[list[str]] = []
        summary = restore.run_restore(
            [{"id": "aaa", "start_s": 0.0, "end_s": 90.0}],
            tmp_path,
            runner=self._runner(calls),
        )
        assert target.read_bytes() == b"ORIGINAL"
        assert summary["skipped_existing"] == 1
