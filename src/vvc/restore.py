"""Restoring clips whose local audio is gone, from the archive.

Roughly a quarter of measured clips no longer have local audio, and most of
those are in the Drive archive as the original 15-minute fetch. Every record
stores the window offsets that were cut from that file, so replaying them
reproduces the exact clip the record was measured from.

That exactness is the whole point. Re-fetching the stream and hunting a window
again would produce a *different* 90 seconds, and any v1-vs-v2 difference would
then confound a new measurement method with a new sample. Replaying the stored
window holds the sample fixed so the method is the only thing that varied — and
it sidesteps the YouTube access gate entirely, which is the expensive part of
acquisition (see pipeline.md).

Restored clips land in the ordinary windows directory, so `build-v2` picks them
up with no special casing.

See reference/pipeline.md.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

DEFAULT_REMOTE = "Google Drive:vanalysis-raw-audio"


def _default_runner(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True, timeout=1800)


def window_path(video_id: str, data_dir: Path) -> Path:
    return Path(data_dir) / "windows" / f"{video_id}_raw90.wav"


def plan_restore(
    records: list[dict], data_dir: Path | str, *, available: set[str]
) -> list[dict]:
    """Which clips can be restored, and with which offsets.

    A record with no stored window is skipped: without offsets the only way to
    produce a clip would be to guess, and a guessed window is a different
    sample wearing the same video id.
    """
    data_dir = Path(data_dir)
    plan: list[dict] = []
    seen: set[str] = set()
    for record in records:
        video_id = str(record.get("id", ""))
        if not video_id or video_id in seen or video_id not in available:
            continue
        if window_path(video_id, data_dir).is_file():
            continue
        window = record.get("window") or {}
        start, end = window.get("start_s"), window.get("end_s")
        if start is None or end is None:
            continue
        seen.add(video_id)
        plan.append({"id": video_id, "start_s": float(start), "end_s": float(end)})
    return plan


def run_restore(
    plan: list[dict],
    data_dir: Path | str,
    *,
    remote: str = DEFAULT_REMOTE,
    runner: Callable[[list[str]], object] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Download each archived raw file, cut its stored window, drop the raw.

    A failure skips and continues, as elsewhere in the pipeline: one unreachable
    file must not abort a long batch. Nothing already on disk is overwritten —
    a restored clip must never silently replace audio that is already there.
    """
    data_dir = Path(data_dir)
    run = runner or _default_runner
    tmp_dir = data_dir / "restore_tmp"
    windows_dir = data_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    summary = {"planned": len(plan), "restored": 0, "failed": 0, "skipped_existing": 0}
    for index, row in enumerate(plan, start=1):
        video_id = row["id"]
        target = window_path(video_id, data_dir)
        if target.is_file():
            summary["skipped_existing"] += 1
            continue

        tmp_dir.mkdir(parents=True, exist_ok=True)
        raw = tmp_dir / f"{video_id}.wav"
        try:
            run(["rclone", "copy", f"{remote}/{video_id}.wav", str(tmp_dir)])
            if not raw.is_file():
                raise FileNotFoundError(f"{video_id} not present on {remote}")
            run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(row["start_s"]), "-to", str(row["end_s"]),
                    "-i", str(raw), str(target),
                ]
            )
            summary["restored"] += 1
        except Exception:
            summary["failed"] += 1
        finally:
            raw.unlink(missing_ok=True)
        if progress:
            progress(index, len(plan))
    return summary


def _default_lister(remote: str) -> str:
    done = subprocess.run(
        ["rclone", "lsf", remote], capture_output=True, text=True, timeout=1800, check=True
    )
    return done.stdout


def archive_ids(
    *, remote: str = DEFAULT_REMOTE, lister: Callable[[str], str] | None = None
) -> set[str]:
    """Video ids present in the archive.

    ``lister`` takes the remote and returns its raw listing, so a caller can
    substitute a recorded listing without a network round trip.
    """
    listing = (lister or _default_lister)(remote)
    return {
        line.strip().rsplit(".", 1)[0]
        for line in listing.splitlines()
        if line.strip()
    }
