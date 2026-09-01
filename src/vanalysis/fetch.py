from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

_YT_WATCH = "https://www.youtube.com/watch?v="

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
    argv = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        "wav",
        "--force-keyframes-at-cuts",
        "--extractor-args",
        "youtube:player_client=web",
        "--download-sections",
        "*15:00-30:00",
        "-o",
        str(dest.with_suffix(".%(ext)s")),
    ]
    if cookies is not None:
        argv.extend(["--cookies", str(cookies)])
    argv.append(f"{_YT_WATCH}{video_id}")
    run(argv)
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
            audio_path(video_id, data_dir).unlink(missing_ok=True)
            raise
        except subprocess.CalledProcessError as exc:
            dest = audio_path(video_id, data_dir)
            dest.unlink(missing_ok=True)
            print(
                f"fetch failed for {video_id} (exit {exc.returncode}); "
                "skipping — gap in data/audio",
                file=sys.stderr,
            )
            results[video_id] = None
    return results
