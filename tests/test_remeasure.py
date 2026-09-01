"""Product tests for whole-corpus Praat re-measurement (PLAN lever 4
decision, following the tracker diagnostic in diagnose.py).

User-visible rules (synthetic wavs ONLY; never Cover/hololive audio,
never GPU, never downloads):

1. ``resolve_source(record, data_dir, stems_dir, windows)`` picks the
   audio that produced the record's CURRENT features: stem90 if its
   windows.json window matches the record's own ``window`` field
   exactly, else raw90b if THAT matches, else raw90 (the default every
   record starts on) when neither retry nor rescue window matches.
2. ``run_remeasure`` replaces ONLY features/qc/tracker on every record,
   preserving id/month/score/window/model unchanged, on the resolved
   audio — never re-hunts a window or re-isolates.
3. A pre-remeasure snapshot of the measurements file is written once
   (``<stem>_pre_praat_remeasure.json``) and never overwritten on a
   second run.
4. A record whose resolved audio file is missing on disk is left
   byte-for-byte unchanged and reported under ``missing`` — never a
   fabricated re-measurement.
"""

from __future__ import annotations

import array
import json
import math
import sys
import wave
from pathlib import Path

from vanalysis import remeasure
from vanalysis.isolate import DEFAULT_MODEL_FILENAME, vocals_path

SR = 16_000


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _write_tone(path: Path, freq_hz: float, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = 0.6 * 32767.0
    n = int(seconds * SR)
    fade = int(0.005 * SR)
    samples = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        samples.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / SR))))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(_samples_to_bytes(samples))


def test_resolve_source_defaults_to_raw90(tmp_path: Path) -> None:
    """Rule 1: no matching retry/rescue window in windows.json -> raw90."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    record = {"id": "abc0000001", "window": {"start_s": 900.0, "end_s": 990.0}}
    path = remeasure.resolve_source(record, data_dir, stems_dir, windows={})
    assert path == vocals_path(
        "abc0000001_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )


def test_resolve_source_prefers_stem90_when_window_matches(tmp_path: Path) -> None:
    """Rule 1: a record whose window exactly matches the recorded
    <id>_stem90 windows.json entry resolves to the stem90 slice."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    record = {"id": "abc0000002", "window": {"start_s": 12.5, "end_s": 102.5}}
    windows = {"abc0000002_stem90": {"start_s": 12.5, "end_s": 102.5}}
    path = remeasure.resolve_source(record, data_dir, stems_dir, windows)
    assert path == data_dir / "windows" / "abc0000002_stem90.wav"


def test_resolve_source_prefers_raw90b_when_window_matches(tmp_path: Path) -> None:
    """Rule 1: raw90b wins over the default when its window matches and
    there is no matching stem90 entry."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    record = {"id": "abc0000003", "window": {"start_s": 300.0, "end_s": 390.0}}
    windows = {"abc0000003_raw90b": {"start_s": 300.0, "end_s": 390.0}}
    path = remeasure.resolve_source(record, data_dir, stems_dir, windows)
    assert path == vocals_path(
        "abc0000003_raw90b.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )


def test_run_remeasure_replaces_features_preserves_identity(tmp_path: Path) -> None:
    """Rule 2: features/qc/tracker are replaced; id/month/score/window/
    model survive unchanged."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    measurements_path = tmp_path / "measurements" / "luna_monthly.json"
    measurements_path.parent.mkdir(parents=True)
    record = {
        "id": "tone0000001",
        "month": "2024-05",
        "score": 70.0,
        "window": {"start_s": 900.0, "end_s": 990.0},
        "features": {"median_f0": 999.0, "f0_iqr": 999.0, "voiced_fraction": 0.1},
        "qc": {"pass": False, "reason": "f0_iqr"},
        "model": DEFAULT_MODEL_FILENAME,
    }
    measurements_path.write_text(json.dumps([record]) + "\n", encoding="utf-8")
    raw90 = vocals_path(
        "tone0000001_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )
    _write_tone(raw90, 220.0)

    summary = remeasure.run_remeasure(
        data_dir,
        measurements_path=measurements_path,
        windows_path=tmp_path / "windows.json",
        stems_dir=stems_dir,
    )

    assert summary["remeasured"] == ["tone0000001"]
    assert summary["missing"] == []
    out = json.loads(measurements_path.read_text(encoding="utf-8"))
    assert len(out) == 1
    [updated] = out
    assert updated["id"] == "tone0000001"
    assert updated["month"] == "2024-05"
    assert updated["score"] == 70.0
    assert updated["window"] == {"start_s": 900.0, "end_s": 990.0}
    assert updated["model"] == DEFAULT_MODEL_FILENAME
    assert updated["tracker"] == remeasure.TRACKER_PRAAT
    assert math.isfinite(updated["features"]["median_f0"])
    assert abs(updated["features"]["median_f0"] - 220.0) < 5.0
    assert updated["qc"]["pass"] is True


def test_run_remeasure_snapshot_written_once(tmp_path: Path) -> None:
    """Rule 3: the pre-remeasure snapshot is created on first run and
    never overwritten on a second run."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    measurements_path = tmp_path / "measurements" / "luna_monthly.json"
    measurements_path.parent.mkdir(parents=True)
    record = {
        "id": "tone0000002",
        "month": "2024-06",
        "score": 50.0,
        "window": {"start_s": 100.0, "end_s": 190.0},
        "features": {"median_f0": 999.0, "f0_iqr": 999.0, "voiced_fraction": 0.1},
        "qc": {"pass": False, "reason": "f0_iqr"},
        "model": DEFAULT_MODEL_FILENAME,
    }
    original_bytes = json.dumps([record]) + "\n"
    measurements_path.write_text(original_bytes, encoding="utf-8")
    raw90 = vocals_path(
        "tone0000002_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )
    _write_tone(raw90, 220.0)

    remeasure.run_remeasure(
        data_dir,
        measurements_path=measurements_path,
        windows_path=tmp_path / "windows.json",
        stems_dir=stems_dir,
    )
    snapshot = measurements_path.with_name("luna_monthly_pre_praat_remeasure.json")
    assert snapshot.read_text(encoding="utf-8") == original_bytes

    # a second run must not clobber the snapshot with post-remeasure bytes
    remeasure.run_remeasure(
        data_dir,
        measurements_path=measurements_path,
        windows_path=tmp_path / "windows.json",
        stems_dir=stems_dir,
    )
    assert snapshot.read_text(encoding="utf-8") == original_bytes


def test_run_remeasure_missing_audio_leaves_record_unchanged(tmp_path: Path) -> None:
    """Rule 4: no raw90/raw90b/stem90 file on disk -> record untouched,
    id reported under 'missing', never a fabricated measurement."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    measurements_path = tmp_path / "measurements" / "luna_monthly.json"
    measurements_path.parent.mkdir(parents=True)
    record = {
        "id": "ghost000001",
        "month": "2024-07",
        "score": 40.0,
        "window": {"start_s": 5.0, "end_s": 95.0},
        "features": {"median_f0": 300.0, "f0_iqr": 20.0, "voiced_fraction": 0.7},
        "qc": {"pass": True, "reason": None},
        "model": DEFAULT_MODEL_FILENAME,
    }
    measurements_path.write_text(json.dumps([record]) + "\n", encoding="utf-8")

    summary = remeasure.run_remeasure(
        data_dir,
        measurements_path=measurements_path,
        windows_path=tmp_path / "windows.json",
        stems_dir=stems_dir,
    )
    assert summary["missing"] == ["ghost000001"]
    assert summary["remeasured"] == []
    out = json.loads(measurements_path.read_text(encoding="utf-8"))
    assert out == [record]
