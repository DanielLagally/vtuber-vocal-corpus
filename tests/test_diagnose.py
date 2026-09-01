"""Product tests for the tracker diagnostic (PLAN "Why the Luna series
disappointed" lever 4): numpy-ACF vs Praat on the SAME on-disk audio,
read-only.

User-visible rules (synthetic wavs ONLY; never Cover/hololive audio,
never GPU, never downloads):

1. ``compare_tracker(ids, data_dir, stems_dir)`` returns one record per
   (id, variant) whose audio file actually exists on disk — a variant
   with no file is skipped entirely, never a null/placeholder record.
2. Each record is ``{"id", "variant", "path", "numpy": {median_f0,
   f0_iqr, voiced_fraction, qc}, "praat": {...same shape...}}`` — same
   feature keys as vanalysis.measure.stem_features /
   vanalysis.praat_features.stem_features, plus a qc_verdict block for
   each tracker.
3. The three variants checked are raw90 (``data/stems_fast/<id>_raw90_
   (vocals)_<model>.wav``), raw90b (same dir, ``_raw90b``) and stem90
   (``data/windows/<id>_stem90.wav`` — already isolated, not re-looked-
   up in stems_fast).
4. ``run_diagnose`` never writes to ``measurements_path`` — only to
   ``out_path``, a separate file. Default ``ids=None`` is exactly the
   QC-failing ids of ``measurements_path`` (reuses retry.failing_ids).
"""

from __future__ import annotations

import array
import json
import math
import sys
import wave
from pathlib import Path

from vanalysis import diagnose
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


def test_compare_tracker_reports_only_existing_variants(tmp_path: Path) -> None:
    """Rule 1 + 3: only raw90 and stem90 exist on disk for this id ->
    exactly two records, raw90b is absent from the output."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    video_id = "toneid0001"

    raw90 = vocals_path(f"{video_id}_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME)
    _write_tone(raw90, 220.0)
    stem90 = data_dir / "windows" / f"{video_id}_stem90.wav"
    _write_tone(stem90, 330.0)

    results = diagnose.compare_tracker([video_id], data_dir, stems_dir)
    variants = {r["variant"] for r in results}
    assert variants == {"raw90", "stem90"}
    assert all(r["id"] == video_id for r in results)


def test_compare_tracker_record_shape(tmp_path: Path) -> None:
    """Rule 2: both trackers report the same feature keys plus a qc
    block; a clean 220 Hz tone passes QC on both trackers."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    video_id = "toneid0002"
    raw90 = vocals_path(f"{video_id}_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME)
    _write_tone(raw90, 220.0)

    [record] = diagnose.compare_tracker([video_id], data_dir, stems_dir)
    numpy_keys = {"median_f0", "f0_iqr", "voiced_fraction", "qc"}
    praat_keys = numpy_keys | {
        "brightness_hz", "dynamism_semitones", "jitter_local",
        "shimmer_local", "hnr_db", "loudness_dynamics_db",
    }
    for tracker, expected_keys in (("numpy", numpy_keys), ("praat", praat_keys)):
        features = record[tracker]
        assert set(features) == expected_keys
        assert math.isfinite(features["median_f0"])
        assert features["qc"]["pass"] is True


def test_run_diagnose_defaults_to_failing_ids_and_does_not_touch_measurements(
    tmp_path: Path,
) -> None:
    """Rule 4: no explicit ids -> exactly the QC-fail ids; measurements
    file bytes are untouched; results land only in out_path."""
    data_dir = tmp_path / "data"
    stems_dir = data_dir / "stems_fast"
    measurements_path = tmp_path / "measurements" / "luna_monthly.json"
    measurements_path.parent.mkdir(parents=True)
    records = [
        {
            "id": "passid00001",
            "features": {"median_f0": 300.0, "f0_iqr": 20.0, "voiced_fraction": 0.7},
            "qc": {"pass": True, "reason": None},
        },
        {
            "id": "failid00001",
            "features": {"median_f0": 400.0, "f0_iqr": 250.0, "voiced_fraction": 0.7},
            "qc": {"pass": False, "reason": "f0_iqr"},
        },
    ]
    original_bytes = json.dumps(records, indent=2) + "\n"
    measurements_path.write_text(original_bytes, encoding="utf-8")

    fail_raw90 = vocals_path(
        "failid00001_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )
    _write_tone(fail_raw90, 250.0)
    pass_raw90 = vocals_path(
        "passid00001_raw90.wav", stems_dir, model_filename=DEFAULT_MODEL_FILENAME
    )
    _write_tone(pass_raw90, 250.0)

    out_path = tmp_path / "measurements" / "luna_tracker_diagnostic.json"
    results = diagnose.run_diagnose(
        None,
        data_dir,
        measurements_path=measurements_path,
        stems_dir=stems_dir,
        out_path=out_path,
    )

    assert {r["id"] for r in results} == {"failid00001"}
    assert measurements_path.read_text(encoding="utf-8") == original_bytes
    assert json.loads(out_path.read_text(encoding="utf-8")) == results
