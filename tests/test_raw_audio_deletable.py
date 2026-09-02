"""Raw audio (data/audio/<id>.wav) is a transient fetch cache, not a
permanent input: fetch/window CLI must recognize a video is already
done from its 90 s window slice (windows.raw90_path), not just from
raw-audio-file presence, so deleting a video's large raw wav after
it's been windowed never causes a wasted re-fetch or re-window.

Only retry/rescue genuinely still need the raw wav (they hunt a
DIFFERENT window when the first one failed QC) — see PLAN.md; that is
by design, not covered here.
"""

from __future__ import annotations

import array
import json
import wave
from pathlib import Path

from vvc import __main__ as cli


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(array.array("h", [0] * 16000).tobytes())


def test_window_cli_skips_already_windowed_id_without_raw_wav(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_wav(data_dir / "windows" / "abc123_raw90.wav")  # already windowed
    assert not (data_dir / "audio" / "abc123.wav").exists()  # raw wav deleted

    called = []
    monkeypatch.setattr(
        cli, "best_speech_window", lambda *a, **k: called.append("hunt") or (0.0, 1.0)
    )
    monkeypatch.setattr(
        cli, "slice_wav", lambda *a, **k: called.append("slice")
    )

    cli.main(["window", "--data-dir", str(data_dir), "abc123"])

    assert called == [], "an already-windowed id must not re-hunt or re-slice"


def test_fetch_cli_skips_already_windowed_id_without_raw_wav(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_wav(data_dir / "windows" / "xyz789_raw90.wav")  # already windowed
    assert not (data_dir / "audio" / "xyz789.wav").exists()  # raw wav deleted

    todo_seen = []
    monkeypatch.setattr(
        cli, "fetch_audio_many", lambda todo, *a, **k: todo_seen.append(list(todo)) or {}
    )

    cli.main(["fetch", "--data-dir", str(data_dir), "xyz789"])

    assert todo_seen == [[]], (
        f"an already-windowed id must be filtered out before fetching, got {todo_seen}"
    )
