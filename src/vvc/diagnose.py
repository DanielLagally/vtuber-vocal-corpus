"""Tracker diagnostic (PLAN "Why the Luna series disappointed" lever 4).

Read-only comparison of the production numpy-ACF tracker against Praat
autocorrelation (``praat_features.py``) on audio ALREADY on disk. It
never fetches, isolates, or writes to ``data/measurements/*.json`` — the
production series is not touched. The question it answers: for the
months where the 2nd-window retry, stem-hunt rescue, and 2nd-stream
fetch all still fail QC, is the numpy tracker itself the bottleneck
(Praat pulls the same audio under the IQR gate), or is it the audio
(both trackers agree it's noisy)?

Three measured-audio variants may exist per id (only the ones present on
disk are compared):
- ``raw90``  — 1st-stream window hunted on raw audio, then isolated
  (``data/stems_fast/<id>_raw90_(vocals)_<model>.wav``).
- ``raw90b`` — 2nd-window retry, same source, non-overlapping window
  (``data/stems_fast/<id>_raw90b_(vocals)_<model>.wav``).
- ``stem90`` — stem-hunt rescue: window hunted on the already-isolated
  full stem, so the slice itself IS the measured audio
  (``data/windows/<id>_stem90.wav``).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import praat_features
from .isolate import DEFAULT_MODEL_FILENAME, vocals_path
from .measure import stem_features as numpy_stem_features
from .qc import qc_verdict
from .retry import failing_ids

_VARIANTS = ("raw90", "raw90b", "stem90")


def _variant_path(
    video_id: str, variant: str, data_dir: Path, stems_dir: Path, model_filename: str
) -> Path:
    if variant == "stem90":
        return data_dir / "windows" / f"{video_id}_stem90.wav"
    return vocals_path(
        f"{video_id}_{variant}.wav", stems_dir, model_filename=model_filename
    )


def compare_tracker(
    ids: list[str],
    data_dir: Path | str,
    stems_dir: Path | str,
    *,
    model_filename: str = DEFAULT_MODEL_FILENAME,
) -> list[dict]:
    """One record per (id, variant) found on disk: ``{"id", "variant",
    "path", "numpy": {features..., "qc"}, "praat": {features..., "qc"}}``.
    A variant whose audio file is missing is skipped entirely (not a
    zero/nan placeholder) — this only ever reports on audio that already
    exists."""
    data_dir = Path(data_dir)
    stems_dir = Path(stems_dir)
    results: list[dict] = []
    for video_id in ids:
        for variant in _VARIANTS:
            path = _variant_path(video_id, variant, data_dir, stems_dir, model_filename)
            if not path.is_file():
                continue
            numpy_features = numpy_stem_features(path)
            praat_feats = praat_features.stem_features(path)
            numpy_pass, numpy_reason = qc_verdict(numpy_features)
            praat_pass, praat_reason = qc_verdict(praat_feats)
            results.append(
                {
                    "id": video_id,
                    "variant": variant,
                    "path": str(path),
                    "numpy": {**numpy_features, "qc": {"pass": numpy_pass, "reason": numpy_reason}},
                    "praat": {**praat_feats, "qc": {"pass": praat_pass, "reason": praat_reason}},
                }
            )
    return results


def run_diagnose(
    ids: list[str] | None,
    data_dir: Path | str,
    *,
    measurements_path: Path | str,
    stems_dir: Path | str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    out_path: Path | str,
) -> list[dict]:
    """CLI entry point: ``ids`` defaults to every record whose ``qc.pass``
    is false in ``measurements_path`` (the current QC-fail set). Writes
    the comparison to ``out_path`` — a new file, never
    ``measurements_path`` itself."""
    measurements_path = Path(measurements_path)
    records = json.loads(measurements_path.read_text(encoding="utf-8"))
    if ids is None:
        ids = failing_ids(records)
    ids = list(dict.fromkeys(ids))
    results = compare_tracker(
        ids, data_dir, stems_dir, model_filename=model_filename
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
