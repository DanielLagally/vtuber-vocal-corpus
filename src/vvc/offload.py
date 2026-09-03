"""Offload a video's raw audio to Google Drive once it's genuinely done
being useful locally, so a long run doesn't fill the local disk with
15-minute raw wavs (see run_densify's docstring for what "genuinely
done" means — windowed, and QC-pass or retry/rescue-exhausted).

Uses ``rclone`` (a Drive remote must already be configured via one-time
``rclone config``, done by a human — this module never does that setup).
Never deletes the local file without a confirmed-successful upload; the
local file is left untouched on any failure.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from .fetch import audio_path

# Matches the rclone remote name configured on this machine via
# `rclone config` (Settings > Google Drive, scope drive.file — this app
# can only see/manage files it created itself, not the whole Drive).
# Named before the vanalysis -> vvc package rename; the remote itself was
# never renamed in `rclone config`, so this must keep matching reality,
# not the package name.
DEFAULT_REMOTE = "Google Drive:vanalysis-raw-audio"


def _default_runner(argv: list[str]) -> object:
    return subprocess.run(argv, check=True)


def offload_raw_audio(
    video_id: str,
    data_dir: Path | str,
    *,
    remote: str = DEFAULT_REMOTE,
    runner: Callable[[list[str]], object] | None = None,
) -> bool:
    """Upload ``video_id``'s raw wav to ``remote`` and delete the local
    copy only after the upload succeeds. Returns False, leaving the
    local file (if any) untouched, when there's nothing to offload or
    the upload fails."""
    raw_wav = audio_path(video_id, Path(data_dir))
    if not raw_wav.is_file():
        return False
    run = runner if runner is not None else _default_runner
    argv = ["rclone", "copy", str(raw_wav), remote]
    try:
        _ = run(argv)
    except (subprocess.CalledProcessError, OSError):
        return False
    raw_wav.unlink()
    return True
