from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

_YT_WATCH = "https://www.youtube.com/watch?v="

# The measured sample is the 15:00-30:00 window of the stream (skip the
# waiting-room intro, take 15 min). yt-dlp's ``--download-sections`` pulls
# exactly that range but for many older VODs it falls back to the ffmpeg
# downloader, which streams at ~playback speed (~30 KiB/s -> ~10 min for
# this window). Instead: yt-dlp downloads the full compressed audio with
# its own (fast, range-request) downloader, then ffmpeg cuts the window
# locally. Same measured audio, ~30x faster on the slow VODs.
_SECTION_START_S = 15 * 60
_SECTION_END_S = 30 * 60

# If a bgutil-ytdlp-pot-provider plugin install is present (see CLAUDE.md
# "PO-token provider"), point yt-dlp at it via PYTHONPATH — this nixpkgs
# yt-dlp ignores --plugin-dirs entirely, PYTHONPATH is the only way it
# picks up a plugin. Absent on a machine with no provider set up; that's
# fine, --extractor-args "fetch_pot=always" then just logs a warning and
# falls back to a token-less format instead of erroring.
_POT_PLUGIN_DIR = Path.home() / ".config" / "yt-dlp" / "plugins"

# Matched as two separate substrings so the exact apostrophe glyph
# (straight vs yt-dlp's curly "'") never matters.
_BOT_CHECK_MARKERS = ("Sign in to confirm you", "not a bot")


class BotCheckDetected(RuntimeError):
    """yt-dlp reported YouTube's bot-check ("Sign in to confirm you're
    not a bot") rather than an ordinary per-video problem (private /
    deleted / region-locked). This is a session/IP-level signal, not a
    per-video gap: every client and cookie combination fails the same
    way once it fires, and continuing to fetch more ids only sends more
    suspicious traffic. Deliberately NOT a subclass of
    ``subprocess.CalledProcessError`` so it is never caught by the
    ordinary per-id skip-and-continue handling."""

    def __init__(self, video_id: str, output: str) -> None:
        self.video_id = video_id
        self.output = output
        super().__init__(f"YouTube bot-check triggered on {video_id}; stopping batch")


def looks_like_bot_check(text: str) -> bool:
    return all(marker in text for marker in _BOT_CHECK_MARKERS)


def audio_path(video_id: str, data_dir: Path) -> Path:
    return Path(data_dir) / "audio" / f"{video_id}.wav"


def _video_id_from_argv(argv: list[str]) -> str:
    return argv[-1].rsplit("=", 1)[-1]


def _default_runner(argv: list[str]) -> object:
    env = os.environ.copy()
    if _POT_PLUGIN_DIR.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_POT_PLUGIN_DIR}{os.pathsep}{existing}" if existing else str(_POT_PLUGIN_DIR)
        )
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        if looks_like_bot_check(combined):
            raise BotCheckDetected(_video_id_from_argv(argv), combined)
        raise subprocess.CalledProcessError(
            result.returncode, argv, result.stdout, result.stderr
        )
    return result


def _full_audio_glob(video_id: str, data_dir: Path) -> list[Path]:
    audio_dir = Path(data_dir) / "audio"
    return sorted(
        p for p in audio_dir.glob(f"{video_id}.full.*") if p.suffix != ".part"
    )


def _cleanup_partial(video_id: str, data_dir: Path) -> None:
    """Remove a skipped/failed id's final wav plus any leftover
    ``<id>.full.*`` download (including a half-written ``.part``)."""
    audio_path(video_id, data_dir).unlink(missing_ok=True)
    audio_dir = Path(data_dir) / "audio"
    for leftover in audio_dir.glob(f"{video_id}.full.*"):
        leftover.unlink(missing_ok=True)


_FETCH_ATTEMPTS = 2  # the PO-token gate is probabilistic per-request, not a
# fixed per-video denylist — the same id can fail then succeed moments
# later with the identical client/token/cookie combo (confirmed 2026-09-05
# on hololive JP veteran talents). One retry recovers a real chunk of
# these without the complexity of juggling multiple client strategies.


def fetch_audio(
    video_id: str,
    data_dir: Path,
    *,
    runner: Callable[[list[str]], object] | None = None,
    cookies: Path | str | None = None,
) -> Path:
    dest = audio_path(video_id, data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run = runner if runner is not None else _default_runner

    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            return _fetch_audio_once(video_id, data_dir, dest, run, cookies)
        except subprocess.CalledProcessError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _fetch_audio_once(
    video_id: str,
    data_dir: Path,
    dest: Path,
    run: Callable[[list[str]], object],
    cookies: Path | str | None,
) -> Path:
    # full compressed audio via yt-dlp's own downloader (fast). mweb +
    # fetch_pot=always: cookies alone dodge the classic bot-check but
    # yt-dlp's default policy never bothers requesting a PO token for an
    # authenticated session, so a video caught by YouTube's per-video GVS
    # PO-token experiment (see CLAUDE.md) has no fallback and just fails.
    # Forcing the token fetch — paired with cookies, not instead of them —
    # recovered ids that failed under every other combination tested
    # (plain cookies, cookieless, web_embedded, mweb+token without
    # cookies). Safe when no PO-token provider is installed: yt-dlp just
    # warns and falls back to a token-less format, per _POT_PLUGIN_DIR.
    full_tmpl = dest.parent / f"{video_id}.full.%(ext)s"
    argv = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extractor-args",
        "youtube:player_client=mweb;fetch_pot=always",
        "-o",
        str(full_tmpl),
    ]
    cookies_copy: Path | None = None
    if cookies is not None:
        # yt-dlp rewrites its cookie jar in place after every run (YouTube
        # rotates the session cookies mid-batch and yt-dlp saves the
        # ever-smaller jar back over the file it was handed). Give it a
        # disposable copy each call so the caller's export — the thing a
        # human has to keep re-exporting — stays byte-for-byte intact.
        fd, tmp = tempfile.mkstemp(
            prefix=f"{video_id}.", suffix=".cookies", dir=str(data_dir)
        )
        os.close(fd)
        cookies_copy = Path(tmp)
        shutil.copyfile(cookies, cookies_copy)
        argv.extend(["--cookies", str(cookies_copy)])
    argv.append(f"{_YT_WATCH}{video_id}")
    try:
        run(argv)

        full = _full_audio_glob(video_id, data_dir)
        if not full:
            # A fake runner in tests materializes the final wav directly
            # and never writes a *.full.* download — nothing to cut,
            # honour it.
            if dest.exists():
                return dest
            raise subprocess.CalledProcessError(
                1, argv, "", "yt-dlp produced no audio"
            )

        # 2. cut the 15:00-30:00 window to wav, locally (instant).
        slice_argv = [
            "ffmpeg",
            "-y",
            "-ss",
            str(_SECTION_START_S),
            "-to",
            str(_SECTION_END_S),
            "-i",
            str(full[0]),
            "-vn",
            str(dest),
        ]
        run(slice_argv)
        for leftover in _full_audio_glob(video_id, data_dir):
            leftover.unlink(missing_ok=True)
        return dest
    finally:
        if cookies_copy is not None:
            cookies_copy.unlink(missing_ok=True)


def fetch_audio_many(
    video_ids: Iterable[str],
    data_dir: Path,
    *,
    runner: Callable[[list[str]], object] | None = None,
    cookies: Path | str | None = None,
) -> dict[str, Path | None]:
    """Fetch each id in order, one yt-dlp run per id. An ORDINARY per-id
    failure (non-zero yt-dlp exit, e.g. private/deleted/region-locked)
    is logged and that id is skipped — a gap in data/audio — while the
    batch continues with the remaining ids.

    A ``BotCheckDetected`` is different: it is NOT caught here. The
    partial dest for that id is cleaned up and the exception is
    re-raised immediately, so no id after it is ever attempted — see
    ``BotCheckDetected``'s docstring for why."""
    results: dict[str, Path | None] = {}
    for video_id in video_ids:
        try:
            results[video_id] = fetch_audio(
                video_id, data_dir, runner=runner, cookies=cookies
            )
        except BotCheckDetected:
            _cleanup_partial(video_id, data_dir)
            raise
        except subprocess.CalledProcessError as exc:
            _cleanup_partial(video_id, data_dir)
            print(
                f"fetch failed for {video_id} (exit {exc.returncode}); "
                "skipping — gap in data/audio",
                file=sys.stderr,
            )
            results[video_id] = None
    return results
