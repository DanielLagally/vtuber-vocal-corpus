"""Densify Luna's monthly series: bring every month below ``target_n``
records up to ``target_n``, using the next-highest-scored eligible
stream(s) from the already-cached raw catalog
(``data/catalog/video_cache/<channel_id>.json`` — no new Holodex calls).

PLAN "how we can improve further" — the more-data lever: a single 90 s
clip is a small, noisy sample of a talent's expressive pitch range
(within-clip f0_iqr routinely 80-180 Hz on real conversational speech).
Averaging 2+ independent streams per month directly shrinks that
sampling noise; this is a variance-reduction lever, separate from (and
additive with) the Praat tracker fix.

Reuses the full production pipeline per new id: fetch (15:00-30:00),
raw90 window hunt, fast RoFormer isolation, Praat measurement — same
processing as every other Luna record, so a new record is directly
comparable to the existing ones. Skip-existing at every step
(crash-resumable); an ORDINARY per-id fetch/isolate failure (private
video, deleted, region-locked) is a warning + skip, never an aborted
batch. Only APPENDS new records — existing records are never touched or
replaced. Snapshot before the first write.

A YouTube bot-check (``fetch.BotCheckDetected``) is different and NOT
treated as an ordinary per-id failure: it's a session/IP-level signal,
not a per-video gap (the 2026-09 Lamy run hit this — 191/198 fetches
rejected with "Sign in to confirm you're not a bot" after a lot of
same-day sequential requests). Continuing to fetch more ids after that
signal only raises more suspicion, so the batch STOPS immediately —
the triggering id is recorded as ``skipped_bot_check`` and
``stopped_early`` in the summary, and no later candidate is attempted.
"""

from __future__ import annotations

import json
import shutil
import warnings
from collections.abc import Callable
from pathlib import Path

from . import praat_features
from .catalog import _available_dt, filter_videos, pick_monthly_n, score_video
from .fetch import BotCheckDetected, audio_path, fetch_audio_many
from .isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from .qc import qc_verdict
from .retry import _window_pair
from .windows import best_speech_window, slice_wav

TRACKER_PRAAT = "praat_ac"
_STEM_MIN_BYTES = 1_000_000
RAW90_SUFFIX = "_raw90"

_OUTCOME_ADDED = "added"
_OUTCOME_BOT_CHECK = "skipped_bot_check"


def candidate_ids(
    records: list[dict], cached_videos: list[dict], target_n: int
) -> dict[str, list[dict]]:
    """``{month: [video dicts to add]}`` for every month whose current
    record count is below ``target_n``: the next-highest-scored eligible
    videos (``pick_monthly_n`` order, i.e. score desc then newest first)
    that are NOT already present as a record for that month, capped so
    the month reaches (but never exceeds) ``target_n``. A month the
    cached catalog has no further eligible video for is simply absent
    (never padded, never an error)."""
    have_by_month: dict[str, set[str]] = {}
    for record in records:
        have_by_month.setdefault(record["month"], set()).add(record["id"])

    eligible = pick_monthly_n(filter_videos(cached_videos), n=target_n)
    by_month: dict[str, list[dict]] = {}
    for video in eligible:
        when = _available_dt(video)
        if when is None:
            continue
        month = f"{when.year:04d}-{when.month:02d}"
        by_month.setdefault(month, []).append(video)

    out: dict[str, list[dict]] = {}
    for month, videos in by_month.items():
        have = have_by_month.get(month, set())
        need = target_n - len(have)
        if need <= 0:
            continue
        fresh = [v for v in videos if v["id"] not in have][:need]
        if fresh:
            out[month] = fresh
    return out


def _snapshot_path(measurements_path: Path) -> Path:
    return measurements_path.with_name(f"{measurements_path.stem}_pre_densify.json")


def run_densify(
    data_dir: Path | str,
    *,
    measurements_path: Path | str,
    video_cache_path: Path | str,
    windows_path: Path | str,
    stems_dir: Path | str,
    target_n: int = 2,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    model_file_dir: str | None = None,
    fetch_runner: Callable[[list[str]], object] | None = None,
    isolate_runner: Callable[[list[str]], object] | None = None,
    cookies: Path | str | None = None,
    out_path: Path | str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Fetch, window, isolate, and Praat-measure the videos
    ``candidate_ids`` selects, appending one new record per success.
    Existing records are never modified. Returns ``{"ids": {id:
    outcome}, "counts": {added, skipped, error, total}, "months_targeted":
    [months], "snapshot": path|None, "dry_run": bool}``.
    """
    data_dir = Path(data_dir)
    measurements_path = Path(measurements_path)
    windows_path = Path(windows_path)
    stems_dir = Path(stems_dir)
    target = Path(out_path) if out_path is not None else measurements_path

    records: list[dict] = json.loads(measurements_path.read_text(encoding="utf-8"))
    cached_videos: list[dict] = json.loads(
        Path(video_cache_path).read_text(encoding="utf-8")
    )
    windows: dict = (
        json.loads(windows_path.read_text(encoding="utf-8"))
        if windows_path.is_file()
        else {}
    )

    buckets = candidate_ids(records, cached_videos, target_n)
    todo = [
        (month, video) for month in sorted(buckets) for video in buckets[month]
    ]

    outcomes: dict[str, str] = {}
    snapshot_taken = False
    stopped_early: str | None = None

    def _emit(video_id: str, outcome: str, detail: str = "") -> None:
        outcomes[video_id] = outcome
        line = f"densify {video_id}: {outcome}" + (f" ({detail})" if detail else "")
        if outcome != _OUTCOME_ADDED:
            warnings.warn(line, stacklevel=3)
        if log is not None:
            log(line)

    def _write_windows() -> None:
        windows_path.parent.mkdir(parents=True, exist_ok=True)
        windows_path.write_text(json.dumps(windows, indent=2) + "\n", encoding="utf-8")

    for month, video in todo:
        video_id = video["id"]
        try:
            raw_wav = audio_path(video_id, data_dir)
            if not (raw_wav.is_file() and raw_wav.stat().st_size > _STEM_MIN_BYTES):
                if dry_run:
                    _emit(video_id, "skipped_dry_run", "audio absent, fetch not run")
                    continue
                fetched = fetch_audio_many(
                    [video_id], data_dir, runner=fetch_runner, cookies=cookies
                )
                if fetched.get(video_id) is None:
                    _emit(video_id, "skipped_fetch_failed", "yt-dlp failed")
                    continue

            key = f"{video_id}{RAW90_SUFFIX}"
            if key in windows:
                start_s, end_s = _window_pair(windows[key])
            else:
                start_s, end_s = best_speech_window(raw_wav)
                if not dry_run:
                    windows[key] = {"start_s": start_s, "end_s": end_s}
                    _write_windows()

            slice_path = data_dir / "windows" / f"{video_id}{RAW90_SUFFIX}.wav"
            if not slice_path.is_file():
                if dry_run:
                    _emit(video_id, "skipped_dry_run", "raw90 slice absent")
                    continue
                slice_wav(raw_wav, slice_path, start_s, end_s)

            stem = vocals_path(
                f"{video_id}{RAW90_SUFFIX}.wav", stems_dir, model_filename=model_filename
            )
            if not (stem.is_file() and stem.stat().st_size > _STEM_MIN_BYTES):
                if dry_run:
                    _emit(video_id, "skipped_dry_run", "stem absent, isolation not run")
                    continue
                isolate_kwargs: dict = {
                    "model_filename": model_filename,
                    "runner": isolate_runner,
                }
                if model_file_dir is not None:
                    isolate_kwargs["model_file_dir"] = model_file_dir
                isolate_vocals(slice_path, stems_dir, **isolate_kwargs)

            if dry_run:
                _emit(video_id, _OUTCOME_ADDED, "dry-run forecast")
                continue

            features = praat_features.stem_features(stem)
            qc_pass, qc_reason = qc_verdict(features)

            if not snapshot_taken:
                snapshot = _snapshot_path(measurements_path)
                if not snapshot.exists():
                    shutil.copyfile(measurements_path, snapshot)
                snapshot_taken = True

            records.append(
                {
                    "id": video_id,
                    "month": month,
                    "score": score_video(video),
                    "window": {"start_s": start_s, "end_s": end_s},
                    "features": features,
                    "qc": {"pass": qc_pass, "reason": qc_reason},
                    "model": model_filename,
                    "tracker": TRACKER_PRAAT,
                }
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            _emit(video_id, _OUTCOME_ADDED)
        except BotCheckDetected as exc:
            # Session/IP-level signal, not a per-video gap: stop the
            # WHOLE batch here rather than skip-and-continue — every
            # further fetch would likely hit the same wall and only
            # raise more suspicion (PLAN "how we can improve further").
            _emit(video_id, _OUTCOME_BOT_CHECK, str(exc))
            stopped_early = video_id
            break
        except Exception as exc:  # noqa: BLE001 — ordinary failures never abort
            _emit(video_id, "error", f"{type(exc).__name__}: {exc}")

    counts = {
        "added": sum(1 for o in outcomes.values() if o == _OUTCOME_ADDED),
        "skipped": sum(1 for o in outcomes.values() if o.startswith("skipped_")),
        "error": sum(1 for o in outcomes.values() if o == "error"),
        "total": len(todo),
    }
    return {
        "ids": outcomes,
        "counts": counts,
        "months_targeted": sorted(buckets),
        "snapshot": (
            str(_snapshot_path(measurements_path)) if snapshot_taken else None
        ),
        "stopped_early": stopped_early,
        "dry_run": dry_run,
    }
