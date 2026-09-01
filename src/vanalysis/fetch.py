from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

_YT_WATCH = "https://www.youtube.com/watch?v="


def audio_path(video_id: str, data_dir: Path) -> Path:
    return Path(data_dir) / "audio" / f"{video_id}.wav"


def _default_runner(argv: list[str]) -> object:
    return subprocess.run(argv, check=True)


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
        "youtube:player_client=android",
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
    """Fetch each id in order, one yt-dlp run per id. A per-id failure
    (non-zero yt-dlp exit) is logged and that id is skipped — a gap in
    data/audio — while the batch continues with the remaining ids."""
    results: dict[str, Path | None] = {}
    for video_id in video_ids:
        try:
            results[video_id] = fetch_audio(
                video_id, data_dir, runner=runner, cookies=cookies
            )
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
