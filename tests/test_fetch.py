"""Product tests for audio fetching.

User-visible rules (local, explicit ids only — never the network in tests):

1. ``audio_path(video_id, data_dir)`` is ``data_dir / "audio" /
   f"{video_id}.wav"``.
2. ``fetch_audio(video_id, data_dir, *, runner=None)``:
   a. creates ``data_dir / "audio"`` if it does not exist yet,
   b. invokes ``runner`` (a callable) with a list of argv whose first
      element is ``"yt-dlp"`` and that includes the video id (a
      ``https://www.youtube.com/watch?v=<id>`` URL counts as including it),
   c. returns exactly ``data_dir / "audio" / f"{video_id}.wav"`` — never a
      path outside ``data_dir``,
   d. accepts keyword-only ``cookies: Path | str | None = None``; when it
      is None the argv must NOT contain ``--cookies``, and when it is a
      path the argv must contain ``--cookies`` and that path as a string.
3. The default runner would subprocess yt-dlp; these tests always pass a
   fake runner so pytest never hits the network and never requires
   yt-dlp to be installed. The fake only materializes a tiny wav at the
   contracted path (fetch_audio may create it itself, or leave it to the
   runner — either way the returned path must exist afterwards).
4. data_dir is always pytest's tmp_path — nothing is written into the
   repo root, and no Cover/hololive audio is involved.
5. The yt-dlp argv must pin the extractor with ``--extractor-args
   youtube:player_client=android`` — never ``mweb``. Cookies, the
   ``15:00-30:00`` section, and sequential one-at-a-time runs stay.
6. ``fetch_audio_many(video_ids, data_dir, *, runner=None, cookies=None)``
   fetches the ids in order, one yt-dlp run per id (the CLI ``fetch``
   loop over several ids is a batch of these). If one id's run fails
   (yt-dlp exits non-zero → the runner raises), that id is skipped and
   the failure is logged, and the batch CONTINUES with the remaining
   ids: a failed month is a gap in data/audio, never an aborted run.
"""

import subprocess
from pathlib import Path

from vanalysis import fetch

VIDEO_ID = "abc123stream"


# ---------------------------------------------------------------- helpers


def _make_fake_runner(calls: list[list[str]], wav_path: Path):
    """A yt-dlp stand-in: records argv, materializes a tiny wav at the
    expected output path if the implementation did not already create it."""

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        if not wav_path.exists():
            wav_path.write_bytes(b"")

    return runner


def _assert_under_data_dir(result: Path, data_dir: Path) -> None:
    result_real = result.resolve()
    data_real = data_dir.resolve()
    assert result_real.is_relative_to(data_real), (
        f"{result} is outside data_dir {data_dir}"
    )


def _make_batch_runner(
    calls: list[list[str]],
    data_dir: Path,
    ids: list[str],
    failing_id: str,
):
    """Fake yt-dlp for a batch: records argv, then either raises (exactly
    like the default check=True runner would after a non-zero yt-dlp exit)
    for ``failing_id``, or materializes the expected wav for the requested
    id — otherwise the later-id assertions would fail for the wrong reason."""

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        joined = " ".join(argv)
        if failing_id in joined:
            raise subprocess.CalledProcessError(returncode=1, cmd=argv)
        video_id = next(vid for vid in ids if vid in joined)
        fetch.audio_path(video_id, data_dir).write_bytes(b"")

    return runner


# ------------------------------------------------------------------- tests


def test_audio_path_layout(tmp_path: Path) -> None:
    """Rule 1: audio_path is data_dir/audio/<video_id>.wav."""
    assert fetch.audio_path(VIDEO_ID, tmp_path) == tmp_path / "audio" / f"{VIDEO_ID}.wav"


def test_fetch_audio_calls_runner_with_ytdlp_and_video_id(tmp_path: Path) -> None:
    """Rule 2b: the runner gets an argv list led by "yt-dlp" that includes
    the video id (bare id or YouTube URL form)."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    assert calls, "fetch_audio must invoke the runner"
    assert len(calls) == 1, f"runner should be invoked exactly once, got {len(calls)}"
    argv = calls[0]
    assert isinstance(argv, list), "runner must receive a list of argv, not a string"
    assert argv[0] == "yt-dlp", f"argv[0] must be 'yt-dlp', got {argv!r}"
    assert any(VIDEO_ID in part for part in argv[1:]), (
        f"video id {VIDEO_ID!r} missing from yt-dlp argv {argv!r}"
    )


def test_fetch_audio_skips_waiting_room(tmp_path: Path) -> None:
    """Waiting-room intro is not the sample: the yt-dlp section must not
    start at 0:00. Default is 15:00–30:00 (skip first 15 min, take 15)."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    argv = calls[0]
    joined = " ".join(argv)
    assert "*0:00-" not in joined and "*0:00:" not in joined, (
        f"must not sample the intro, got {argv!r}"
    )
    assert "15:00-30:00" in joined, f"expected 15:00-30:00 slice, got {argv!r}"


def test_fetch_audio_creates_audio_dir_and_returns_wav_path(tmp_path: Path) -> None:
    """Rule 2a + 2c: data_dir/audio is created (even if data_dir itself did
    not exist) and the returned path is data_dir/audio/<id>.wav."""
    data_dir = tmp_path / "data"
    assert not data_dir.exists(), "precondition: data_dir starts absent"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, data_dir / "audio" / f"{VIDEO_ID}.wav")

    result = fetch.fetch_audio(VIDEO_ID, data_dir, runner=runner)

    expected = data_dir / "audio" / f"{VIDEO_ID}.wav"
    assert (data_dir / "audio").is_dir(), "fetch_audio must create data_dir/audio"
    assert result == expected, f"expected {expected}, got {result}"
    assert result.exists(), "the returned wav path must exist after fetch_audio"


def test_fetch_audio_result_stays_under_data_dir(tmp_path: Path) -> None:
    """Rule 2c: the result path never escapes data_dir (no repo-root or
    cwd writes)."""
    data_dir = tmp_path / "somewhere" / "vanalysis-data"
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, data_dir / "audio" / f"{VIDEO_ID}.wav")

    result = fetch.fetch_audio(VIDEO_ID, data_dir, runner=runner)

    assert result.name == f"{VIDEO_ID}.wav", f"expected a .wav filename, got {result}"
    assert result.parent == data_dir / "audio"
    _assert_under_data_dir(result, data_dir)


def test_fetch_audio_returns_same_path_as_audio_path(tmp_path: Path) -> None:
    """Consistency: fetch_audio's result and audio_path agree for the same
    video id and data_dir."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, fetch.audio_path(VIDEO_ID, tmp_path))

    result = fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    assert result == fetch.audio_path(VIDEO_ID, tmp_path)


def test_fetch_audio_without_cookies_omits_cookies_flag(tmp_path: Path) -> None:
    """Rule 2d (default): no cookies argument means yt-dlp must not be
    handed ``--cookies`` at all."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner, cookies=None)

    argv = calls[0]
    assert "--cookies" not in argv, (
        f"--cookies must be absent when cookies is None, got {argv!r}"
    )


def test_fetch_audio_with_cookies_passes_flag_and_path(tmp_path: Path) -> None:
    """Rule 2d (cookies given): a cookie file path becomes ``--cookies
    <str(path)>`` in the yt-dlp argv."""
    cookies = tmp_path / "youtube.cookies.txt"
    cookies.write_bytes(b"")  # empty file; contents are irrelevant here
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner, cookies=cookies)

    argv = calls[0]
    joined = " ".join(argv)
    assert "--cookies" in joined, f"--cookies missing from yt-dlp argv {argv!r}"
    assert str(cookies) in argv, (
        f"cookie path {str(cookies)!r} missing from yt-dlp argv {argv!r}"
    )


def test_fetch_audio_uses_android_player_client(tmp_path: Path) -> None:
    """Rule 5: the yt-dlp extractor must be pinned to player_client=android
    (tv now fails with "The page needs to be reloaded" on current YouTube);
    mweb no longer passes YouTube's gate for this flow."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    argv = calls[0]
    assert "--extractor-args" in argv, (
        f"--extractor-args missing from yt-dlp argv {argv!r}"
    )
    assert "youtube:player_client=android" in argv, (
        f"expected youtube:player_client=android in yt-dlp argv, got {argv!r}"
    )
    assert not any("mweb" in part for part in argv), (
        f"mweb must not appear anywhere in yt-dlp argv, got {argv!r}"
    )


def test_fetch_batch_skips_failed_id_and_continues(
    tmp_path: Path, capsys, caplog
) -> None:
    """Rule 6: one failed yt-dlp run skips that id (logged) and the batch
    keeps fetching the rest — a failed month is a gap, not an aborted run."""
    ids = ["month1clip", "month2fail", "month3clip"]
    calls: list[list[str]] = []
    runner = _make_batch_runner(calls, tmp_path, ids, failing_id="month2fail")

    fetch.fetch_audio_many(ids, tmp_path, runner=runner)  # must not raise

    # Every id was attempted exactly once, in order — including the ids
    # that come after the failing one.
    assert [argv[0] for argv in calls] == ["yt-dlp"] * len(ids)
    attempted = [next(vid for vid in ids if vid in " ".join(argv)) for argv in calls]
    assert attempted == ids, f"batch must attempt every id in order, got {attempted}"

    # Later ids really got their wavs; the failed id's wav stays absent
    # (the gap downstream commands already treat as "missing").
    assert fetch.audio_path("month1clip", tmp_path).exists()
    assert fetch.audio_path("month3clip", tmp_path).exists()
    assert not fetch.audio_path("month2fail", tmp_path).exists(), (
        "a failed id must be skipped, not left half-written"
    )

    # The failure is logged somewhere the operator can see it (print to
    # stdout/stderr or the logging module are both acceptable).
    captured = capsys.readouterr()
    logged = captured.out + captured.err + "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert "month2fail" in logged, (
        "the failed id must be logged (stdout, stderr, or logging)"
    )
