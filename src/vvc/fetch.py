from __future__ import annotations

import subprocess
import sys
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
    result = subprocess.run(argv, capture_output=True, text=True)
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

    # 1. full compressed audio via yt-dlp's own downloader (fast).
    full_tmpl = dest.parent / f"{video_id}.full.%(ext)s"
    argv = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extractor-args",
        "youtube:player_client=mweb",
        "-o",
        str(full_tmpl),
    ]
    if cookies is not None:
        argv.extend(["--cookies", str(cookies)])
    argv.append(f"{_YT_WATCH}{video_id}")
    run(argv)

    full = _full_audio_glob(video_id, data_dir)
    if not full:
        # A fake runner in tests materializes the final wav directly and
        # never writes a *.full.* download — nothing to cut, honour it.
        if dest.exists():
            return dest
        raise subprocess.CalledProcessError(1, argv, "", "yt-dlp produced no audio")

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
