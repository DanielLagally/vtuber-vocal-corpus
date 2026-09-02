"""Whole-corpus Praat re-measurement (PLAN lever 4 decision: adopt Praat
as the production tracker for Luna after the diagnostic showed it clears
every one of the residual numpy-ACF failures — see diagnose.py).

Re-measures EVERY record in a measurements file with Praat autocorrelation
(praat_features.py) instead of numpy ACF, on the EXACT same audio each
record already used — id/month/score/window/model are preserved
unchanged; only features/qc are replaced, plus a "tracker" field marking
provenance. Same audio, different tracker only: this is what the "same
processing per talent" rule in PLAN.md requires, not a re-run of window
hunting or isolation.

Source resolution: a record's authoritative audio is whichever of
<id>_raw90 (original), <id>_raw90b (2nd-window retry), <id>_stem90
(stem-hunt rescue) has a window entry in windows.json matching the
record's own ``window`` field exactly (retry/rescue only ever replace a
record with the window that produced it, and only on a pass) — stem90
checked first, then raw90b, else raw90 (every record starts there).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import praat_features
from .isolate import DEFAULT_MODEL_FILENAME, vocals_path
from .qc import qc_verdict
from .retry import _window_pair

TRACKER_PRAAT = "praat_ac"


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def resolve_source(
    record: dict,
    data_dir: Path | str,
    stems_dir: Path | str,
    windows: dict,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """The audio file that produced ``record``'s CURRENT features."""
    data_dir = Path(data_dir)
    stems_dir = Path(stems_dir)
    video_id = record["id"]
    window = record.get("window") or {}
    start_s = window.get("start_s")
    end_s = window.get("end_s")

    def _matches(key: str) -> bool:
        entry = windows.get(key)
        if entry is None or start_s is None or end_s is None:
            return False
        w_start, w_end = _window_pair(entry)
        return _close(w_start, start_s) and _close(w_end, end_s)

    if _matches(f"{video_id}_stem90"):
        return data_dir / "windows" / f"{video_id}_stem90.wav"
    if _matches(f"{video_id}_raw90b"):
        return vocals_path(
            f"{video_id}_raw90b.wav", stems_dir, model_filename=model_filename
        )
    return vocals_path(
        f"{video_id}_raw90.wav", stems_dir, model_filename=model_filename
    )


def _snapshot_path(measurements_path: Path) -> Path:
    return measurements_path.with_name(
        f"{measurements_path.stem}_pre_praat_remeasure.json"
    )


def run_remeasure(
    data_dir: Path | str,
    *,
    measurements_path: Path | str,
    windows_path: Path | str,
    stems_dir: Path | str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    out_path: Path | str | None = None,
) -> dict:
    """Re-measure every record in ``measurements_path`` with Praat, on
    the same audio it already used. Snapshots the pre-remeasure file
    (once, never overwritten) before writing — same convention as
    retry/rescue. A record whose resolved audio file is missing on disk
    is left completely unchanged and reported under ``missing`` (never a
    fabricated result).

    Returns ``{"total", "remeasured": [ids], "missing": [ids], "pass":
    n, "fail": n, "snapshot": path}``.
    """
    data_dir = Path(data_dir)
    measurements_path = Path(measurements_path)
    windows_path = Path(windows_path)
    stems_dir = Path(stems_dir)
    target = Path(out_path) if out_path is not None else measurements_path

    records: list[dict] = json.loads(measurements_path.read_text(encoding="utf-8"))
    windows: dict = (
        json.loads(windows_path.read_text(encoding="utf-8"))
        if windows_path.is_file()
        else {}
    )

    remeasured: list[str] = []
    missing: list[str] = []
    out_records: list[dict] = []
    for record in records:
        path = resolve_source(record, data_dir, stems_dir, windows, model_filename)
        if not path.is_file():
            missing.append(record["id"])
            out_records.append(record)
            continue
        features = praat_features.stem_features(path)
        qc_pass, qc_reason = qc_verdict(features)
        out_records.append(
            {
                **record,
                "features": features,
                "qc": {"pass": qc_pass, "reason": qc_reason},
                "tracker": TRACKER_PRAAT,
            }
        )
        remeasured.append(record["id"])

    snapshot = _snapshot_path(measurements_path)
    if not snapshot.exists():
        shutil.copyfile(measurements_path, snapshot)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out_records, indent=2) + "\n", encoding="utf-8")

    passed = sum(1 for r in out_records if r.get("qc", {}).get("pass"))
    return {
        "total": len(records),
        "remeasured": remeasured,
        "missing": missing,
        "pass": passed,
        "fail": len(out_records) - passed,
        "snapshot": str(snapshot),
    }
