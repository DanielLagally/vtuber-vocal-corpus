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
      is None the argv must NOT contain ``--cookies``. When it is a path,
      the argv contains ``--cookies`` pointing at a *disposable copy* of
      that file whose contents match the original — never the caller's
      own path. yt-dlp rewrites its cookie jar in place after every run
      (YouTube rotates the session cookies mid-batch), so handing it the
      user's real export degrades that file to an auth-less stub within
      ~100 fetches and re-trips the bot-check on long unattended runs;
      fetch_audio must leave the caller's file byte-for-byte intact.
3. The default runner would subprocess yt-dlp; these tests always pass a
   fake runner so pytest never hits the network and never requires
   yt-dlp to be installed. The fake only materializes a tiny wav at the
   contracted path (fetch_audio may create it itself, or leave it to the
   runner — either way the returned path must exist afterwards).
4. data_dir is always pytest's tmp_path — nothing is written into the
   repo root, and no Cover/hololive audio is involved.
5. The yt-dlp argv must pin the extractor with ``--extractor-args
   youtube:player_client=web_embedded`` (2026-09: ``android``/``android_vr``
   are skipped by yt-dlp entirely whenever ``--cookies`` is passed — "does
   not support cookies" — so pairing either with our cookies file
   silently drops the cookies and the request still hits YouTube's
   bot-check; ``tv`` still fails with "The page needs to be reloaded".
   ``web``, then ``mweb``, each worked at first but YouTube's GVS PO
   Token experiment (binding a PO Token requirement to specific video
   ids) expanded to cover each client in turn — ``web`` first (71/74
   lost in one Lamy batch), then ``mweb`` (2026-09-05: 89% lost in a
   Flare batch, confirmed via yt-dlp's own debug log, "Detected
   experiment to bind GVS PO Token to video ID for mweb client"; the
   failed ids were confirmed genuinely public via Holodex's own "past"
   status, not really unavailable). ``web_embedded`` resolves a real
   format on ids gated under both ``web`` and ``mweb`` with no PO
   token provider needed — if it starts failing the same way too, the
   durable fix is a self-hosted PO token provider, not another client
   swap.). Cookies and sequential one-at-a-time runs stay.
   The ``15:00-30:00`` window is now cut LOCALLY with ffmpeg after
   yt-dlp downloads the full compressed audio: yt-dlp's
   ``--download-sections`` falls back to its ffmpeg downloader for many
   older VODs, which streams at ~playback speed (~10 min for this
   window vs ~20 s for a full download). So fetch_audio makes two runner
   calls — ``yt-dlp`` (full audio, no ``--download-sections``) then
   ``ffmpeg`` (``-ss 900 -to 1800``) — and deletes the ``<id>.full.*``
   download once the ``<id>.wav`` cut exists.
6. ``fetch_audio_many(video_ids, data_dir, *, runner=None, cookies=None)``
   fetches the ids in order, one yt-dlp run per id (the CLI ``fetch``
   loop over several ids is a batch of these). If one id's run fails
   with an ORDINARY yt-dlp error (private/deleted/region-locked — the
   runner raises plain ``subprocess.CalledProcessError``), that id is
   skipped and the failure is logged, and the batch CONTINUES with the
   remaining ids: a failed month is a gap in data/audio, never an
   aborted run.
7. ``looks_like_bot_check(text)`` recognizes yt-dlp's "Sign in to
   confirm you're not a bot" output regardless of the exact apostrophe
   glyph (matches "Sign in to confirm you" AND "not a bot" as two
   separate substrings). This is a session/IP-level signal, not a
   per-video problem — when the runner raises ``BotCheckDetected``,
   ``fetch_audio_many`` does NOT skip-and-continue: it cleans up any
   partial dest for that id and re-raises immediately, WITHOUT
   attempting any remaining ids. Continuing to hammer YouTube after
   this signal only raises more suspicion; the caller must stop the
   whole batch, not just this id.
"""

import subprocess
from pathlib import Path

from vvc import fetch

VIDEO_ID = "abc123stream"


# ---------------------------------------------------------------- helpers


def _make_fake_runner(calls: list[list[str]], wav_path: Path):
    """yt-dlp + ffmpeg stand-in: records argv and mimics the real
    two-step flow — the ``yt-dlp`` call "downloads" ``<id>.full.wav``,
    the ``ffmpeg`` call "cuts" that to the final ``<id>.wav``."""
    full_path = wav_path.with_name(wav_path.stem + ".full.wav")

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if argv[0] == "yt-dlp":
            full_path.write_bytes(b"")
        elif argv[0] == "ffmpeg":
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
        video_id = next(vid for vid in ids if vid in joined)
        if argv[0] == "yt-dlp":
            if video_id == failing_id:
                raise subprocess.CalledProcessError(returncode=1, cmd=argv)
            (data_dir / "audio").mkdir(parents=True, exist_ok=True)
            (data_dir / "audio" / f"{video_id}.full.wav").write_bytes(b"")
        elif argv[0] == "ffmpeg":
            fetch.audio_path(video_id, data_dir).write_bytes(b"")

    return runner


# ------------------------------------------------------------------- tests


def test_audio_path_layout(tmp_path: Path) -> None:
    """Rule 1: audio_path is data_dir/audio/<video_id>.wav."""
    assert fetch.audio_path(VIDEO_ID, tmp_path) == tmp_path / "audio" / f"{VIDEO_ID}.wav"


def test_fetch_audio_calls_runner_with_ytdlp_and_video_id(tmp_path: Path) -> None:
    """Rule 2b: the first runner call is an argv list led by "yt-dlp" that
    includes the video id (bare id or YouTube URL form). fetch_audio then
    makes a second call — ffmpeg, to cut the window locally."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    assert calls, "fetch_audio must invoke the runner"
    assert [c[0] for c in calls] == ["yt-dlp", "ffmpeg"], (
        f"expected a yt-dlp download then an ffmpeg cut, got {[c[0] for c in calls]!r}"
    )
    ytdlp = calls[0]
    assert isinstance(ytdlp, list), "runner must receive a list of argv, not a string"
    assert any(VIDEO_ID in part for part in ytdlp[1:]), (
        f"video id {VIDEO_ID!r} missing from yt-dlp argv {ytdlp!r}"
    )
    assert "--download-sections" not in ytdlp, (
        "the full audio is downloaded fast, then cut locally — no yt-dlp "
        f"--download-sections (its ffmpeg fallback is ~playback speed), got {ytdlp!r}"
    )


def test_fetch_audio_skips_waiting_room(tmp_path: Path) -> None:
    """Waiting-room intro is not the sample: the local cut must start at
    15:00, not 0:00 (skip first 15 min, take 15 -> 15:00-30:00)."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    ffmpeg = next(c for c in calls if c[0] == "ffmpeg")
    assert "-ss" in ffmpeg, f"ffmpeg cut must seek to the window start, got {ffmpeg!r}"
    start = ffmpeg[ffmpeg.index("-ss") + 1]
    assert start not in ("0", "0.0", "00:00:00"), (
        f"must not sample the intro, cut starts at {start!r}"
    )
    assert int(float(start)) == 15 * 60, f"expected a 15:00 start, got {start!r}"
    assert "-to" in ffmpeg and int(float(ffmpeg[ffmpeg.index("-to") + 1])) == 30 * 60, (
        f"expected a 30:00 end, got {ffmpeg!r}"
    )


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
    data_dir = tmp_path / "somewhere" / "vvc-data"
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


def test_fetch_audio_with_cookies_passes_flag_and_copy(tmp_path: Path) -> None:
    """Rule 2d (cookies given): yt-dlp is handed ``--cookies <path>`` where
    <path> is a readable copy of the caller's file (same contents) that
    exists during the call, not the caller's own path."""
    cookies = tmp_path / "youtube.cookies.txt"
    body = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tv\n"
    cookies.write_text(body)
    calls: list[list[str]] = []
    seen: dict[str, object] = {}

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
        if argv[0] == "yt-dlp":
            passed = Path(argv[argv.index("--cookies") + 1])
            seen["path"] = passed
            seen["contents"] = passed.read_text()  # must exist mid-call
            (tmp_path / "audio" / f"{VIDEO_ID}.full.wav").write_bytes(b"")
        elif argv[0] == "ffmpeg":
            fetch.audio_path(VIDEO_ID, tmp_path).write_bytes(b"")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner, cookies=cookies)

    assert "--cookies" in calls[0], f"--cookies missing from yt-dlp argv {calls[0]!r}"
    assert seen["path"] != cookies, (
        f"yt-dlp was handed the caller's real cookie path {cookies!r}, not a copy"
    )
    assert seen["contents"] == body, (
        "the cookie copy handed to yt-dlp does not match the caller's file"
    )


def test_fetch_audio_does_not_mutate_callers_cookie_file(tmp_path: Path) -> None:
    """Rule 2d: yt-dlp rewrites its cookie jar in place after each run
    (YouTube rotates the session cookies and yt-dlp saves the ever-smaller
    jar back over the file it was given). fetch_audio must shield the
    caller's export — it stays byte-for-byte intact even when the runner
    truncates the path it received."""
    cookies = tmp_path / "youtube.cookies.txt"
    original = (
        "# Netscape HTTP Cookie File\n"
        ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\treal\n"
        ".youtube.com\tTRUE\t/\tTRUE\t0\tHSID\treal\n"
    )
    cookies.write_text(original)
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
        if argv[0] == "yt-dlp":
            # mimic yt-dlp's cookie-jar write-back: the file it was handed
            # comes back rewritten, here truncated to an auth-less stub.
            Path(argv[argv.index("--cookies") + 1]).write_text(
                "# Netscape HTTP Cookie File\n"
            )
            (tmp_path / "audio" / f"{VIDEO_ID}.full.wav").write_bytes(b"")
        elif argv[0] == "ffmpeg":
            fetch.audio_path(VIDEO_ID, tmp_path).write_bytes(b"")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner, cookies=cookies)

    assert cookies.read_text() == original, (
        "fetch_audio let yt-dlp rewrite the caller's cookie file; it must "
        "pass a disposable copy instead"
    )


def test_fetch_audio_uses_web_embedded_player_client(tmp_path: Path) -> None:
    """Rule 5: the yt-dlp extractor must be pinned to
    player_client=web_embedded. android/android_vr are skipped by
    yt-dlp whenever --cookies is passed ("does not support cookies") —
    silently dropping our cookies and still hitting the bot-check; tv
    fails with "The page needs to be reloaded"; web and then mweb each
    worked only until YouTube's per-video GVS PO Token experiment
    expanded to cover them too (2026-09-05: mweb started failing
    "Video unavailable" on ~89% of a Flare densify run, confirmed via
    yt-dlp -v: "Detected experiment to bind GVS PO Token to video ID
    for mweb client" — 134/151 of the failed ids were confirmed
    non-private via Holodex's own "past" status, spanning 2021-2026,
    so this was never real unavailability). web_embedded resolves a
    real, non-PO-token-gated format on both the ids that failed under
    mweb and ids that already worked under mweb (no regression)."""
    calls: list[list[str]] = []
    runner = _make_fake_runner(calls, tmp_path / "audio" / f"{VIDEO_ID}.wav")

    fetch.fetch_audio(VIDEO_ID, tmp_path, runner=runner)

    argv = calls[0]
    assert "--extractor-args" in argv, (
        f"--extractor-args missing from yt-dlp argv {argv!r}"
    )
    assert "youtube:player_client=web_embedded" in argv, (
        f"expected youtube:player_client=web_embedded in yt-dlp argv, got {argv!r}"
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
    # that come after the failing one. (One yt-dlp call per id; successful
    # ids get a follow-up ffmpeg cut.)
    ytdlp = [argv for argv in calls if argv[0] == "yt-dlp"]
    attempted = [next(vid for vid in ids if vid in " ".join(argv)) for argv in ytdlp]
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


def test_looks_like_bot_check_matches_real_message() -> None:
    """Rule 7: recognizes yt-dlp's real output regardless of apostrophe
    glyph (curly vs straight), and does not false-positive on an
    ordinary private-video error."""
    curly = (
        "ERROR: [youtube] abc123: Sign in to confirm you’re not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )
    straight = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot."
    private = "ERROR: [youtube] abc123: Private video. If the owner of this video..."
    assert fetch.looks_like_bot_check(curly) is True
    assert fetch.looks_like_bot_check(straight) is True
    assert fetch.looks_like_bot_check(private) is False
    assert fetch.looks_like_bot_check("") is False


def test_fetch_batch_stops_immediately_on_bot_check(tmp_path: Path) -> None:
    """Rule 7: a BotCheckDetected on one id must propagate out of
    fetch_audio_many immediately — no later id may be attempted, and
    the failing id's partial dest (if any) is cleaned up. An earlier
    successful id's wav is untouched."""
    ids = ["month1clip", "month2bot", "month3never"]
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> None:
        calls.append(list(argv))
        joined = " ".join(argv)
        if "month2bot" in joined:
            fetch.audio_path("month2bot", tmp_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            fetch.audio_path("month2bot", tmp_path).write_bytes(b"partial")
            raise fetch.BotCheckDetected(
                "month2bot", "Sign in to confirm you’re not a bot."
            )
        video_id = next(vid for vid in ids if vid in joined)
        fetch.audio_path(video_id, tmp_path).write_bytes(b"")

    try:
        fetch.fetch_audio_many(ids, tmp_path, runner=runner)
        raised = False
    except fetch.BotCheckDetected:
        raised = True

    assert raised, "fetch_audio_many must propagate BotCheckDetected, not swallow it"
    attempted = [next(vid for vid in ids if vid in " ".join(argv)) for argv in calls]
    assert attempted == ["month1clip", "month2bot"], (
        f"month3never must never be attempted after the bot-check, got {attempted}"
    )
    assert fetch.audio_path("month1clip", tmp_path).exists()
    assert not fetch.audio_path("month2bot", tmp_path).exists(), (
        "the partial dest for the bot-checked id must be cleaned up"
    )
    assert not fetch.audio_path("month3never", tmp_path).exists()
