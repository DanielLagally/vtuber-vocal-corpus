from __future__ import annotations

import json
import math
import tempfile
import warnings
from pathlib import Path

from .catalog import _available_dt, score_video
from .features import f0_iqr, median_f0, voiced_fraction
from .isolate import _DEFAULT_PRESET, vocals_path
from .windows import slice_wav

_IQR_QC_MAX = 200.0


def _stem_path(video_id: str, stems_dir: Path, model_filename: str | None) -> Path:
    return vocals_path(f"{video_id}.wav", stems_dir, model_filename=model_filename)


def _model_provenance(model_filename: str | None) -> str:
    if model_filename is not None:
        return model_filename
    return f"ensemble_preset:{_DEFAULT_PRESET}"


def _window_features(stem: Path, window: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        sliced = Path(tmp) / "window.wav"
        slice_wav(stem, sliced, float(window["start_s"]), float(window["end_s"]))
        return {
            "median_f0": median_f0(sliced),
            "f0_iqr": f0_iqr(sliced),
            "voiced_fraction": voiced_fraction(sliced),
        }


def run_monthly(
    picks: list[dict],
    windows_path: Path,
    stems_dir: Path,
    out_path: Path,
    *,
    model_filename: str | None = None,
) -> list[dict]:
    windows = json.loads(Path(windows_path).read_text(encoding="utf-8"))
    stems_dir = Path(stems_dir)
    out_path = Path(out_path)
    entries = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else []
    seen = {entry["id"] for entry in entries}
    for pick in picks:
        video_id = pick["id"]
        if video_id in seen:
            continue
        stem = _stem_path(video_id, stems_dir, model_filename)
        if not stem.is_file():
            warnings.warn(f"missing stem for {video_id}: {stem}")
            continue
        window = windows.get(video_id)
        if window is None:
            warnings.warn(f"no window for {video_id} in {windows_path}")
            continue
        features = _window_features(stem, window)
        iqr = features["f0_iqr"]
        qc_pass = math.isfinite(iqr) and iqr < _IQR_QC_MAX
        when = _available_dt(pick)
        entries.append(
            {
                "id": video_id,
                "month": when.strftime("%Y-%m") if when is not None else None,
                "score": score_video(pick),
                "window": dict(window),
                "features": features,
                "qc": {"pass": qc_pass, "reason": None if qc_pass else "f0_iqr"},
                "model": _model_provenance(model_filename),
            }
        )
        seen.add(video_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return entries
