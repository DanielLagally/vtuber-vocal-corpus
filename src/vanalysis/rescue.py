"""Stem-hunt rescue batch (STATE Run 2 R1; PLAN "goal shift" lever 3).

The raw-audio window hunt locks onto music: Run 1 showed 28/34 retried
Luna months still fail IQR >= 200 because the 90 s window is chosen on
RAW audio. Rescue inverts the order: isolate the FULL 15-min wav with
the same RoFormer model, hunt ``best_speech_window`` ON THE STEM (BGM
gone), slice the stem to ``data/windows/<id>_stem90.wav`` (measurement
input — the stem is already isolated), measure the WHOLE stem90 slice,
and replace the record only if it passes QC (replace-if-pass, with a
pre-rescue snapshot). A failing stem hunt keeps the existing fail —
never an invented pass.

windows.json gets APPEND/merge-only ``<id>_stem90`` object entries on
the shared timeline; the 16 legacy array entries of the real index are
never rewritten, and the file is only rewritten when a key was added.

Crash-resume: every step is individually skippable — a recorded
``<id>_stem90`` window is reused instead of re-hunted, a ready full
stem (> 1 MB, the isolate CLI's skip-existing threshold) is not
re-isolated, an existing stem90 slice is not re-sliced, and the
measurements file is rewritten after each successful replacement.
Per-id errors warn and continue; the batch never aborts.

``dry_run=True`` computes everything possible but writes NOTHING.

Structurally a sibling of ``retry.run_retry`` (snapshot,
replace-if-pass, per-id continue, dry-run, resume, windows.json merge)
and reuses its helpers (``failing_ids``, ``_window_pair``,
``_STEM_MIN_BYTES``).
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
from .retry import _STEM_MIN_BYTES, _window_pair, failing_ids
from .windows import best_speech_window, slice_wav

STEM90_SUFFIX = "_stem90"

DEFAULT_MODEL_FILE_DIR = "data/models"

_OUTCOME_REPLACED = "replaced"
_OUTCOME_KEPT_FAIL = "kept_fail"


def _snapshot_path(measurements_path: Path) -> Path:
    return measurements_path.with_name(
        f"{measurements_path.stem}_pre_rescue.json"
    )


def run_rescue(
    ids: list[str] | None,
    data_dir: Path | str,
    *,
    measurements_path: Path | str,
    windows_path: Path | str,
    stems_dir: Path | str,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    model_file_dir: str = DEFAULT_MODEL_FILE_DIR,
    isolate_runner: Callable[[list[str]], object] | None = None,
    out_path: Path | str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Rescue QC-failing months by hunting the 90 s window on the full
    vocal stem.

    ``ids``: the ids to rescue, or None for ``failing_ids`` of the
    measurements file. ``data_dir``: holds ``audio/<id>.wav`` (raw
    15-min wavs) and ``windows/`` (stem90 slices). ``stems_dir``: where
    the full-file stem ``<id>_(vocals)_<model stem>.wav`` lives / is
    written. ``model_file_dir``: audio-separator ``--model_file_dir``
    value (in-tree model ckpt cache); ``"data/models"`` by default.
    ``out_path``: where the updated measurements list is written; None
    = rewrite ``measurements_path`` in place. ``isolate_runner``:
    injectable audio-separator runner (tests pass a fake). ``log``:
    optional callable receiving one line per id outcome; problems
    (skips/errors) additionally raise a UserWarning.

    Returns ``{"ids": {id: outcome}, "counts": {replaced, kept_fail,
    skipped, error, total}, "windows_added": [keys], "snapshot":
    path|None, "dry_run": bool}``.
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
        line = f"rescue {video_id}: {outcome}" + (f" ({detail})" if detail else "")
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

            # 1. the full-file stem: a ready one (> 1 MB) skips isolation.
            raw_wav = data_dir / "audio" / f"{video_id}.wav"
            full_stem = vocals_path(
                raw_wav, stems_dir, model_filename=model_filename
            )
            if not (
                full_stem.is_file()
                and full_stem.stat().st_size > _STEM_MIN_BYTES
            ):
                if not raw_wav.is_file():
                    _emit(video_id, "skipped_missing_wav", f"missing {raw_wav}")
                    continue
                if dry_run:
                    _emit(
                        video_id, "skipped_dry_run",
                        "full stem absent, isolation not run",
                    )
                    continue
                isolate_kwargs: dict = {
                    "model_filename": model_filename,
                    "runner": isolate_runner,
                }
                if model_file_dir is not None:
                    isolate_kwargs["model_file_dir"] = model_file_dir
                isolate_vocals(raw_wav, stems_dir, **isolate_kwargs)

            # 2. the stem90 window: reuse a recorded <id>_stem90 entry
            # (crash-resume, first write wins) or hunt ON THE STEM.
            key = f"{video_id}{STEM90_SUFFIX}"
            hunted_new = False
            if key in windows:
                start_s, end_s = _window_pair(windows[key])
            else:
                try:
                    start_s, end_s = best_speech_window(full_stem)
                except ValueError as exc:
                    _emit(video_id, "skipped_no_window", str(exc))
                    continue
                hunted_new = True
                if not dry_run:
                    windows[key] = {"start_s": start_s, "end_s": end_s}
                    windows_added.append(key)
                    _write_windows()

            # 3. slice the STEM (the measurement input — the stem is
            # already isolated). Re-slice when the window is freshly
            # hunted so the slice always matches it; an existing slice
            # with a recorded window is reused.
            slice_path = data_dir / "windows" / f"{video_id}{STEM90_SUFFIX}.wav"
            if hunted_new or not slice_path.is_file():
                if dry_run:
                    _emit(
                        video_id, "skipped_dry_run",
                        "stem90 slice absent, measurement not run",
                    )
                    continue
                slice_wav(full_stem, slice_path, start_s, end_s)

            # 4. measure the WHOLE stem90 slice and apply the QC rule.
            features = stem_features(slice_path)
            qc_pass, qc_reason = qc_verdict(features)
            if not qc_pass:
                _emit(
                    video_id, _OUTCOME_KEPT_FAIL,
                    f"stem90 fails QC ({qc_reason}); fail stays",
                )
                continue
            if dry_run:
                _emit(video_id, _OUTCOME_REPLACED, "dry-run forecast")
                continue

            # 5. replace in place, snapshot taken before the first one.
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
