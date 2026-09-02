"""2nd-window retry batch (PLAN next-step 1; STATE A4; DECISIONS D3/D4).

For each QC-failing measurement record, hunt a 2nd-best 90 s speech
window on the raw 15-min wav that does not overlap the record's first
window, slice it as ``<id>_raw90b``, isolate it with the SAME RoFormer
model, measure the WHOLE stem, and — only if the 2nd window passes QC —
REPLACE the record in place (D3: replace-in-place with a pre-retry
snapshot, not dual-attempt records). A failing 2nd window keeps the
first-window fail (a gap), never an invented pass.

windows.json gets APPEND-only ``<id>_raw90b`` object entries (D4); the
16 legacy array entries of the real index are never rewritten.

Crash-resume: every step is individually skippable — a recorded
``<id>_raw90b`` window is reused instead of re-hunted, a ready stem
(> 1 MB, the isolate CLI's skip-existing threshold) is not re-isolated,
and the measurements file is rewritten after each successful
replacement. Per-id errors warn and continue; the batch never aborts.

``dry_run=True`` computes everything possible but writes NOTHING.
"""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

from .isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from .measure import stem_features
from .qc import qc_verdict
from .windows import second_speech_window, slice_wav

# mirrors the isolate CLI's skip-existing threshold (__main__.py)
_STEM_MIN_BYTES = 1_000_000

RAW90B_SUFFIX = "_raw90b"

_OUTCOME_REPLACED = "replaced"
_OUTCOME_KEPT_FAIL = "kept_fail"


def failing_ids(records: list[dict]) -> list[str]:
    """The default retry set: exactly the ids whose stored ``qc.pass`` is
    false (nan / IQR / high fail reasons all included), in file order.
    Passing records are never selected."""
    return [
        record["id"]
        for record in records
        if not record.get("qc", {}).get("pass", False)
    ]


def _window_pair(value: dict | list) -> tuple[float, float]:
    """windows.json entries are {"start_s": .., "end_s": ..}; the 24-clip
    era wrote legacy 2-element arrays — accept both."""
    if isinstance(value, dict):
        return float(value["start_s"]), float(value["end_s"])
    start_s, end_s = value
    return float(start_s), float(end_s)


def _snapshot_path(measurements_path: Path) -> Path:
    return measurements_path.with_name(
        f"{measurements_path.stem}_pre_retry_snapshot.json"
    )


def run_retry(
    ids: list[str] | None,
    data_dir: Path | str,
    *,
    measurements_path: Path | str,
    windows_path: Path | str,
    stems_dir: Path | str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    isolate_runner: Callable[[list[str]], object] | None = None,
    out_path: Path | str | None = None,
    dry_run: bool = False,
    model_file_dir: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Retry QC-failing months on a 2nd 90 s window.

    ``ids``: the ids to retry, or None for ``failing_ids`` of the
    measurements file. ``data_dir``: holds ``audio/<id>.wav`` (raw
    15-min wavs) and ``windows/`` (slices). ``out_path``: where the
    updated measurements list is written; None = rewrite
    ``measurements_path`` in place (only then is the pre-retry snapshot
    taken). ``isolate_runner``: injectable audio-separator runner (tests
    pass a fake). ``model_file_dir``: optional audio-separator
    ``--model_file_dir`` value (in-tree model ckpt cache); when None the
    isolate call is unchanged. ``log``: optional callable receiving one
    line per id outcome; problems (skips/errors) additionally raise a
    UserWarning.

    Returns ``{"ids": {id: outcome}, "counts": {...}, "windows_added":
    [keys], "snapshot": path|None, "dry_run": bool}`` where outcome is
    ``replaced`` | ``kept_fail`` | ``skipped_<reason>`` | ``error``.
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
    positions = {record["id"]: i for i, record in enumerate(records)}
    by_id = {record["id"]: record for record in records}

    if ids is None:
        ids = failing_ids(records)
    ids = list(dict.fromkeys(ids))  # dedupe, keep order

    outcomes: dict[str, str] = {}
    windows_added: list[str] = []
    snapshot_taken = False

    def _emit(video_id: str, outcome: str, detail: str = "") -> None:
        outcomes[video_id] = outcome
        line = f"retry {video_id}: {outcome}" + (f" ({detail})" if detail else "")
        if outcome not in (_OUTCOME_REPLACED, _OUTCOME_KEPT_FAIL):
            warnings.warn(line, stacklevel=3)
        if log is not None:
            log(line)

    def _write_windows() -> None:
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        windows_path.write_text(json.dumps(windows, indent=2) + "\n", encoding="utf-8")

    for video_id in ids:
        try:
            record = by_id.get(video_id)
            if record is None:
                _emit(video_id, "skipped_missing_record", "no measurement record")
                continue
            if record.get("qc", {}).get("pass", False):
                _emit(video_id, "skipped_qc_pass", "record already passes QC")
                continue
            raw_wav = data_dir / "audio" / f"{video_id}.wav"
            if not raw_wav.is_file():
                _emit(video_id, "skipped_missing_wav", f"missing {raw_wav}")
                continue

            # 1+2. the 2nd window: reuse a recorded <id>_raw90b entry
            # (crash-resume, first write wins) or hunt a new one.
            key = f"{video_id}{RAW90B_SUFFIX}"
            if key in windows:
                start_s, end_s = _window_pair(windows[key])
            else:
                start_s, end_s = second_speech_window(
                    raw_wav, _window_pair(record["window"])
                )
                if not dry_run:
                    windows[key] = {"start_s": start_s, "end_s": end_s}
                    windows_added.append(key)
                    _write_windows()

            # 3+4. slice and isolate (skipped when the stem is ready).
            stem = vocals_path(
                f"{video_id}{RAW90B_SUFFIX}.wav", stems_dir,
                model_filename=model_filename,
            )
            if not (stem.is_file() and stem.stat().st_size > _STEM_MIN_BYTES):
                if dry_run:
                    _emit(video_id, "skipped_dry_run", "stem absent, isolation not run")
                    continue
                slice_path = data_dir / "windows" / f"{video_id}{RAW90B_SUFFIX}.wav"
                slice_wav(raw_wav, slice_path, start_s, end_s)
                isolate_kwargs: dict = {
                    "model_filename": model_filename,
                    "runner": isolate_runner,
                }
                if model_file_dir is not None:
                    isolate_kwargs["model_file_dir"] = model_file_dir
                isolate_vocals(slice_path, stems_dir, **isolate_kwargs)

            # 5. measure the WHOLE raw90b stem and apply the QC rule.
            features = stem_features(stem)
            qc_pass, qc_reason = qc_verdict(features)
            if not qc_pass:
                _emit(
                    video_id, _OUTCOME_KEPT_FAIL,
                    f"2nd window fails QC ({qc_reason}); first fail stays",
                )
                continue
            if dry_run:
                _emit(video_id, _OUTCOME_REPLACED, "dry-run forecast")
                continue

            # 6+7. replace in place, snapshot taken before the first one.
            if not snapshot_taken:
                snapshot = _snapshot_path(measurements_path)
                if not snapshot.exists():
                    shutil.copyfile(measurements_path, snapshot)
                snapshot_taken = True
            position = positions[video_id]
            records[position] = {
                "id": record["id"],
                "month": record["month"],
                "score": record["score"],
                "window": {"start_s": start_s, "end_s": end_s},
                "features": features,
                "qc": {"pass": qc_pass, "reason": qc_reason},
                "model": record["model"],
            }
            by_id[video_id] = records[position]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            _emit(video_id, _OUTCOME_REPLACED)
        except Exception as exc:  # noqa: BLE001 — the batch never aborts
            _emit(video_id, "error", f"{type(exc).__name__}: {exc}")

    counts = {
        "replaced": sum(1 for o in outcomes.values() if o == _OUTCOME_REPLACED),
        "kept_fail": sum(1 for o in outcomes.values() if o == _OUTCOME_KEPT_FAIL),
        "skipped": sum(1 for o in outcomes.values() if o.startswith("skipped_")),
        "error": sum(1 for o in outcomes.values() if o == "error"),
        "total": len(ids),
    }
    return {
        "ids": outcomes,
        "counts": counts,
        "windows_added": windows_added,
        "snapshot": (
            str(_snapshot_path(measurements_path))
            if snapshot_taken and counts["replaced"] > 0
            else None
        ),
        "dry_run": dry_run,
    }
