from __future__ import annotations

import json
import warnings
from pathlib import Path

from .catalog import _available_dt, score_video
from .features import f0_iqr, median_f0, voiced_fraction
from .isolate import _DEFAULT_PRESET, vocals_path
from .qc import qc_verdict


def _stem_candidates(
    video_id: str, stems_dir: Path, model_filename: str | None
) -> tuple[Path, Path]:
    """The stem paths isolate's naming produces: the fast 90 s slice
    (source ``<id>_raw90.wav`` -> ``<id>_raw90_(vocals)_<model>.wav``) and
    the full-file-era stem (``<id>_(vocals)_<model>.wav``)."""
    fast = vocals_path(f"{video_id}_raw90.wav", stems_dir, model_filename=model_filename)
    plain = vocals_path(f"{video_id}.wav", stems_dir, model_filename=model_filename)
    return fast, plain


def _stem_path(video_id: str, stems_dir: Path, model_filename: str | None) -> Path:
    fast, plain = _stem_candidates(video_id, stems_dir, model_filename)
    if fast.is_file():
        if plain.is_file():
            warnings.warn(
                f"both fast and full-file stems found for {video_id}: "
                f"measuring the fast stem {fast.name}"
            )
        return fast
    return plain


def _model_provenance(model_filename: str | None) -> str:
    if model_filename is not None:
        return model_filename
    return f"ensemble_preset:{_DEFAULT_PRESET}"


def _normalize_window(window: dict | list) -> dict:
    """windows.json entries are {"start_s": .., "end_s": ..}; the 24-clip
    era wrote legacy 2-element arrays [start_s, end_s] — accept both and
    return the object form."""
    if isinstance(window, dict):
        return {"start_s": float(window["start_s"]), "end_s": float(window["end_s"])}
    start_s, end_s = window
    return {"start_s": float(start_s), "end_s": float(end_s)}


def _stem_features(stem: Path) -> dict:
    """Measure the WHOLE stem: the 90 s stem IS the window. The full-file
    window offsets from windows.json are metadata, never a re-slice."""
    return {
        "median_f0": median_f0(stem),
        "f0_iqr": f0_iqr(stem),
        "voiced_fraction": voiced_fraction(stem),
    }


def stem_features(stem: Path) -> dict:
    """Public per-stem measurement helper: features of the WHOLE stem
    (the stem IS the window). Same numbers as run_monthly persists —
    shared with the retry batch so both measure identically."""
    return _stem_features(stem)


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
            fast, plain = _stem_candidates(video_id, stems_dir, model_filename)
            warnings.warn(
                f"missing stem for {video_id}: tried {fast.name} and {plain.name}"
            )
            continue
        window = windows.get(video_id)
        if window is None:
            warnings.warn(f"no window for {video_id} in {windows_path}")
            continue
        features = _stem_features(stem)
        qc_pass, qc_reason = qc_verdict(features)
        when = _available_dt(pick)
        entries.append(
            {
                "id": video_id,
                "month": when.strftime("%Y-%m") if when is not None else None,
                "score": score_video(pick),
                "window": _normalize_window(window),
                "features": features,
                "qc": {"pass": qc_pass, "reason": qc_reason},
                "model": _model_provenance(model_filename),
            }
        )
        seen.add(video_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return entries
