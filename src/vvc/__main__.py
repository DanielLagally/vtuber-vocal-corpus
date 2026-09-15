from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from vvc.catalog import filter_videos, pick_monthly
from vvc.densify import run_densify
from vvc.diagnose import run_diagnose
from vvc.remeasure import run_remeasure
from vvc.expand import fetch_channel_videos, run_plan
from vvc.fetch import BotCheckDetected, audio_path, fetch_audio_many
from vvc.holodex import holodex_key as _holodex_key
from vvc.holodex import load_dotenv as _load_dotenv  # noqa: F401 (compat alias)
from vvc.isolate import DEFAULT_MODEL_FILENAME, isolate_vocals, vocals_path
from vvc.acoustics import DEFAULT_CONFIG as _FEATURE_DEFAULTS
from vvc.background import DEFAULT_THRESHOLD_DB as _BACKGROUND_THRESHOLD_DB
from vvc.measure import run_monthly
from vvc import reliability
from vvc.plausibility import DEFAULT_HIGH_SEMITONES as _PLAUSIBILITY_HIGH
from vvc.plausibility import DEFAULT_LOW_SEMITONES as _PLAUSIBILITY_LOW
from vvc.rescue import run_rescue
from vvc.restore import DEFAULT_REMOTE as _RESTORE_REMOTE
from vvc.retry import run_retry
from vvc.roster import fetch_channels, filter_talents, write_roster
from vvc.site_data import write_site_data
from vvc.site_data_v2 import write_site_data_v2
from vvc.series import (
    new_run_dir,
    write_feature_multi_talent_yearly_plot,
    write_feature_yearly_plot,
    write_multi_talent_plot,
    write_plots,
    write_quarterly_plots,
    write_yearly_plot,
)

# (feature_key, filename, subject, subtitle, unit_label, caveat|None) - each
# plotted only when at least one measurement record actually has the key
# (older corpora / pre-backfill records won't).
_EXTRA_FEATURE_PLOTS = (
    (
        "brightness_hz", "brightness_yearly.png", "Brightness by Year",
        "Spectral centroid - a brighter/more forward vs. darker/warmer voice.\n"
        "Mic/EQ and leftover BGM affect this too; trust the relative shape.",
        "Brightness (Hz)", None,
    ),
    (
        "dynamism_semitones", "dynamism_yearly.png", "Pitch Dynamism by Year",
        "Mean semitone change between consecutive voiced frames - how much\n"
        "the pitch actually moves, not just its static spread (F0 IQR).",
        "Dynamism (semitones)", None,
    ),
    (
        "loudness_dynamics_db", "loudness_dynamics_yearly.png",
        "Loudness Dynamics by Year",
        "Spread of frame loudness (RMS in dB) within a clip - animated\n"
        "volume swings vs. a flat, even delivery.",
        "Loudness spread (dB)", None,
    ),
    (
        "jitter_local", "jitter_yearly.png", "Jitter by Year",
        "Cycle-to-cycle pitch-period timing irregularity.",
        "Jitter (local, fraction)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
    (
        "shimmer_local", "shimmer_yearly.png", "Shimmer by Year",
        "Cycle-to-cycle amplitude irregularity.",
        "Shimmer (local, fraction)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
    (
        "hnr_db", "hnr_yearly.png", "Harmonics-to-Noise Ratio by Year",
        "Higher = clearer/more tonal voice; lower = breathier/noisier.",
        "HNR (dB)",
        "CAVEAT: calibrated for a sustained vowel, not conversational speech,\n"
        "and sensitive to residual vocal-isolation artifact - trust the\n"
        "relative shape within this pipeline, not the absolute number.",
    ),
    (
        "f1_hz", "f1_yearly.png", "First Formant (F1) by Year",
        "Vocal-tract resonance most tied to jaw/tongue height - lower F1 is a\n"
        "more close vowel articulation, not directly a pitch measure.",
        "F1 (Hz)",
        "CAVEAT: formant tracking is sensitive to the maximum-formant\n"
        "parameter, pre-emphasis, and residual vocal-isolation artifact -\n"
        "trust the relative shape within this pipeline, not the absolute Hz\n"
        "against a clinical/phonetics reference.",
    ),
    (
        "f2_hz", "f2_yearly.png", "Second Formant (F2) by Year",
        "Vocal-tract resonance most tied to tongue front/back position -\n"
        "together with F1, the classic acoustic vowel-space axes.",
        "F2 (Hz)",
        "CAVEAT: formant tracking is sensitive to the maximum-formant\n"
        "parameter, pre-emphasis, and residual vocal-isolation artifact -\n"
        "trust the relative shape within this pipeline, not the absolute Hz\n"
        "against a clinical/phonetics reference.",
    ),
    (
        "f3_hz", "f3_yearly.png", "Third Formant (F3) by Year",
        "Higher vocal-tract resonance - overall formant spacing (F1-F4)\n"
        "tracks vocal tract length, the strongest acoustic correlate of\n"
        "perceived voice maturity/body size measured on this site.",
        "F3 (Hz)",
        "CAVEAT: formant tracking is sensitive to the maximum-formant\n"
        "parameter, pre-emphasis, and residual vocal-isolation artifact -\n"
        "trust the relative shape within this pipeline, not the absolute Hz\n"
        "against a clinical/phonetics reference.",
    ),
    (
        "f4_hz", "f4_yearly.png", "Fourth Formant (F4) by Year",
        "Highest formant tracked here - completes the F1-F4 vocal-tract-\n"
        "length picture alongside F1-F3.",
        "F4 (Hz)",
        "CAVEAT: formant tracking is sensitive to the maximum-formant\n"
        "parameter, pre-emphasis, and residual vocal-isolation artifact -\n"
        "trust the relative shape within this pipeline, not the absolute Hz\n"
        "against a clinical/phonetics reference.",
    ),
)
from vvc.windows import best_speech_window, raw90_path, slice_wav

_HOLODEX_VIDEOS = "https://holodex.net/api/v2/videos"
_PAGE = 50


def _list_holodex(api_key: str, channel: str | None = None) -> list[dict]:
    import requests

    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "org": "Hololive",
            "type": "stream",
            "status": "past",
            "limit": _PAGE,
            "offset": offset,
        }
        if channel is not None:
            params["channel_id"] = channel
            params["include"] = "mentions"
        response = requests.get(
            _HOLODEX_VIDEOS,
            headers={"X-APIKEY": api_key, "User-Agent": "vvc/0.1"},
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        if len(page) < _PAGE:
            break
        offset += _PAGE
    return rows


def _load_registry(path: Path) -> dict[str, str]:
    """``{measurements_path_str: talent_display_name}``. Missing file ->
    empty registry (the very first talent ever plotted starts one)."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_registry(path: Path, registry: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_registered_talents(registry: dict[str, str]) -> dict[str, list[dict]]:
    """Loads every registered talent's measurements file. A registered
    path that no longer exists (moved/renamed) is skipped with a
    warning, not a crash — the comparison still runs on whoever's left."""
    talents: dict[str, list[dict]] = {}
    for path_str, name in registry.items():
        path = Path(path_str)
        if not path.is_file():
            print(f"warning: registered talent {name!r} missing {path}, skipping", file=sys.stderr)
            continue
        talents[name] = json.loads(path.read_text(encoding="utf-8"))
    return talents


def _write_all_comparison_plots(talents: dict[str, list[dict]], run_dir: Path) -> None:
    write_multi_talent_plot(talents, run_dir)
    present_keys = {
        key
        for entries in talents.values()
        for entry in entries
        for key in (entry.get("features") or {})
    }
    for feature_key, filename, subject, subtitle, unit_label, caveat in _EXTRA_FEATURE_PLOTS:
        if feature_key in present_keys:
            write_feature_multi_talent_yearly_plot(
                talents, run_dir,
                feature_key=feature_key,
                filename=filename.replace(".png", "_multi.png"),
                subject=f"{subject} — Talent Comparison",
                subtitle=subtitle, unit_label=unit_label, caveat=caveat,
            )


def _run_reliability(
    measurements_dir: Path, out: Path, *, semitone_threshold: float
) -> dict:
    """Read-only audit: the corpus noise floor plus discordance review flags.

    Pairing is strictly within one talent (see reliability.corpus_noise_floor),
    so the per-file split is load-bearing, not just convenient bookkeeping.
    """
    # Accepts either corpus layout: v1 files are <talent>_monthly.json, v2
    # files are <talent>.json in their own directory. Globbing only the v1
    # pattern silently reported an empty corpus rather than an error.
    by_talent: dict[str, list[dict]] = {}
    paths = sorted(measurements_dir.glob("*_monthly.json")) or [
        p for p in sorted(measurements_dir.glob("*.json")) if p.name != "talents.json"
    ]
    for path in paths:
        talent = path.name.removesuffix("_monthly.json").removesuffix(".json")
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            by_talent[talent] = records
    if not by_talent:
        raise SystemExit(f"no measurement files found in {measurements_dir}")

    floor = reliability.corpus_noise_floor(by_talent)

    flags: list[dict] = []
    for talent, records in sorted(by_talent.items()):
        for flag in reliability.discordance_flags(
            records, semitone_threshold=semitone_threshold
        ):
            flags.append(
                {
                    "talent": talent,
                    "month": flag.month,
                    "ids": list(flag.ids),
                    "values": list(flag.values),
                    "semitones": flag.semitones,
                    "feature_key": flag.feature_key,
                    "reason": flag.reason,
                }
            )
    flags.sort(key=lambda row: -row["semitones"])

    payload = {
        "semitone_threshold": semitone_threshold,
        "talents": len(by_talent),
        "noise_floor": floor,
        "flags": flags,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'metric':22}{'pairs':>7}{'median |d|':>12}{'p90 |d|':>10}{'med |d| st':>12}")
    for key, row in floor.items():
        if not row["n_pairs"]:
            print(f"{key:22}{0:>7}")
            continue
        semis = row["median_abs_semitones"]
        print(
            f"{key:22}{row['n_pairs']:>7}{row['median_abs_diff']:>12.3f}"
            f"{row['p90_abs_diff']:>10.3f}"
            f"{(semis if semis is not None else float('nan')):>12.2f}"
        )
    print(
        f"\n{len(flags)} discordance flag(s) above {semitone_threshold} semitones "
        f"across {len(by_talent)} talent(s) -> {out}"
    )
    return payload


def _resolve_clip_audio(video_id: str, data_dir: Path) -> Path | None:
    """The best local audio for a measured clip, preferring the separated stem
    the corpus was actually measured from, then the raw window it came from.

    Stem lookups go through the cached directory listing rather than a glob:
    globbing scans the whole directory, and these directories hold tens of
    thousands of entries, so a corpus-sized resolution pass was costing minutes
    of pure CPU.
    """
    from vvc import background

    for subdir in ("stems_fast", "stems"):
        found = background.vocals_stem_path(video_id, data_dir / subdir)
        if found is not None:
            return found
    window = data_dir / "windows" / f"{video_id}_raw90.wav"
    return window if window.is_file() else None


def _raw_window_path(video_id: str, data_dir: Path) -> Path | None:
    candidate = data_dir / "windows" / f"{video_id}_raw90.wav"
    return candidate if candidate.is_file() else None


def _resolve_v2_audio(
    video_id: str,
    data_dir: Path,
    *,
    threshold_db: float,
    ratio_db: float | None = None,
) -> tuple[Path | None, bool, float]:
    """Pick the audio to measure, and say whether it was separated.

    Separation is lossy. Measuring the same clips both ways shows median pitch
    barely moves, but voiced fraction, harmonicity and the formants all shift
    materially — and harmonicity shifts in a systematic direction, because a
    denoiser raises it by construction. So on a clip with nothing to remove,
    separating only adds error.

    The residual stem says how much competing sound there actually was, so the
    decision is measured rather than assumed. Returns the chosen path, whether
    it is separated, and the ratio that decided it — all three get recorded,
    because a corpus whose treatment varies is usable only when the treatment
    is known.
    """
    from vvc import background

    ratio = ratio_db if ratio_db is not None else float("nan")
    if ratio != ratio:
        ratio = background.clip_background_ratio_db(video_id, data_dir / "stems_fast")
    if ratio != ratio:
        ratio = background.clip_background_ratio_db(video_id, data_dir / "stems")

    raw = _raw_window_path(video_id, data_dir)
    if raw is not None and not background.needs_separation(ratio, threshold_db=threshold_db):
        return raw, False, ratio

    stem = _resolve_clip_audio(video_id, data_dir)
    if stem is None:
        return None, False, ratio
    # _resolve_clip_audio falls back to the raw window when no stem exists.
    separated = "(vocals)" in stem.name
    return stem, separated, ratio


def _run_bakeoff(args) -> dict:
    """Tracker comparison, synthetic accuracy, and era sensitivity.

    Read-only with respect to the corpus: it reads audio and writes a report.
    """
    from vvc import bakeoff

    report: dict = {"floor": args.floor, "ceiling": args.ceiling}
    work_dir = args.out.parent / "bakeoff_work"

    if not args.skip_synthetic:
        print("synthetic accuracy (known F0) ...")
        rows = bakeoff.synthetic_accuracy(
            work_dir / "synthetic", floor=args.floor, ceiling=args.ceiling
        )
        report["synthetic"] = rows
        by_key: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row["abs_semitone_error"] is not None:
                by_key.setdefault((row["tracker"], row["condition"]), []).append(
                    row["abs_semitone_error"]
                )
        conditions = sorted({row["condition"] for row in rows})
        print(f"  {'tracker':20}" + "".join(f"{c:>12}" for c in conditions) + "   (median |semitone error|)")
        for tracker in sorted({row["tracker"] for row in rows}):
            cells = ""
            for condition in conditions:
                errs = by_key.get((tracker, condition))
                cells += f"{statistics.median(errs):>12.2f}" if errs else f"{'-':>12}"
            print(f"  {tracker:20}{cells}")

    by_talent: dict[str, list[dict]] = {}
    for path in sorted(args.measurements_dir.glob("*_monthly.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            by_talent[path.name.removesuffix("_monthly.json")] = records

    required = set(args.required_ids or [])
    sample = bakeoff.stratified_sample(
        by_talent, n=args.n, seed=args.seed, required_ids=required
    )
    clips: list[tuple[str, Path]] = []
    for row in sample:
        audio = _resolve_clip_audio(row["id"], args.data_dir)
        if audio is not None:
            clips.append((row["id"], audio))
    print(f"\nreal-audio sample: {len(clips)} clips with local audio (of {len(sample)} drawn)")

    if clips:
        print("running trackers ...")
        results = bakeoff.run_trackers(
            clips, floor=args.floor, ceiling=args.ceiling, tracker_names=args.trackers
        )
        deviation = bakeoff.consensus_deviation(results)
        report["real_audio"] = {
            "n_clips": len(clips),
            "consensus_deviation": deviation,
            "results": bakeoff.results_as_dicts(results),
        }
        print(
            f"  {'tracker':20}{'clips':>7}{'med |st|':>10}{'p90 |st|':>10}"
            f"{'gross':>7}{'gross%':>8}{'sec/clip':>10}"
        )
        for tracker, row in deviation.items():
            if not row["n_clips"]:
                print(f"  {tracker:20}{0:>7}")
                continue
            print(
                f"  {tracker:20}{row['n_clips']:>7}{row['median_abs_semitones']:>10.3f}"
                f"{row['p90_abs_semitones']:>10.3f}{row['gross_disagreements']:>7}"
                f"{100 * row['gross_rate']:>7.1f}%{row['median_seconds']:>10.2f}"
            )

    if not args.skip_era and clips:
        era_clips = [path for _, path in clips[: args.era_n]]
        print(f"\nera sensitivity on {len(era_clips)} clips ...")
        rows = bakeoff.era_sensitivity(
            era_clips,
            tracker_name=args.era_tracker,
            work_dir=work_dir / "era",
            floor=args.floor,
            ceiling=args.ceiling,
            extra_features=True,
        )
        report["era"] = rows
        by_bitrate: dict[int, list[dict]] = {}
        for row in rows:
            by_bitrate.setdefault(row["bitrate_kbps"], []).append(row)
        keys = ("median_f0_delta", "voiced_fraction_delta", "brightness_hz_delta",
                "hnr_db_delta", "jitter_local_delta", "shimmer_local_delta")
        print(f"  {'kbps':>6}" + "".join(f"{k.replace('_delta',''):>22}" for k in keys))
        for bitrate in sorted(by_bitrate):
            cells = ""
            for key in keys:
                vals = [r[key] for r in by_bitrate[bitrate] if r.get(key) is not None]
                cells += f"{statistics.median(vals):>22.4f}" if vals else f"{'-':>22}"
            print(f"  {bitrate:>6}{cells}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {args.out}")
    return report


def _load_v1_by_talent(measurements_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(measurements_dir.glob("*_monthly.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            out[path.name.removesuffix("_monthly.json")] = records
    return out


def _load_v2_by_talent(v2_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(v2_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            out[path.stem] = records
    return out


def _run_build_v2(args) -> dict:
    """Re-measure local audio into the v2 corpus. Never touches v1."""
    from vvc import background, corpus, metadata, plausibility
    from vvc.acoustics import FeatureConfig

    config = FeatureConfig(
        tracker=args.tracker, floor=args.floor, ceiling=args.ceiling
    )
    cache = metadata.load_video_cache(args.cache_dir)
    by_talent = _load_v1_by_talent(args.measurements_dir)
    if args.talents:
        by_talent = {k: v for k, v in by_talent.items() if k in set(args.talents)}

    # The background scan is pure I/O over hundreds of gigabytes of stems and
    # is otherwise the longest serial step in a build, so it runs in the pool
    # up front. It is a full read of every stem, deliberately: estimating the
    # level from a sample is far cheaper but misreads bursty residual stems by
    # several dB, and this ratio decides which audio gets measured.
    all_ids = [
        str(record.get("id", "")) for records in by_talent.values() for record in records
    ]
    print(f"scanning background levels for {len(all_ids)} clips ...", flush=True)

    def scan_progress(done: int, total: int) -> None:
        if done % 1000 == 0 or done == total:
            print(f"  scanned {done}/{total}", flush=True)

    ratios = background.ratios_for(
        all_ids,
        args.data_dir / "stems_fast",
        workers=args.workers,
        progress=scan_progress,
    )

    # Decide once per clip which audio to measure and whether it is separated,
    # so the choice, the ratio behind it, and the resulting record all agree.
    decisions: dict[str, tuple[Path | None, bool, float]] = {}

    def decide(video_id: str) -> tuple[Path | None, bool, float]:
        if video_id not in decisions:
            ratio_db = ratios.get(video_id, float("nan"))
            if args.always_separate:
                decisions[video_id] = (
                    _resolve_clip_audio(video_id, args.data_dir),
                    True,
                    ratio_db,
                )
            else:
                decisions[video_id] = _resolve_v2_audio(
                    video_id,
                    args.data_dir,
                    threshold_db=args.separation_threshold_db,
                    ratio_db=ratio_db,
                )
        return decisions[video_id]

    def resolve(video_id: str) -> Path | None:
        return decide(video_id)[0]

    def ratio(video_id: str) -> float:
        return decide(video_id)[2]

    def separated(video_id: str) -> bool:
        return decide(video_id)[1]

    # Measure everything up front so the work can be spread across processes;
    # assembling records afterwards is cheap and keeps per-talent output intact.
    audio_paths: list[Path] = []
    for records in by_talent.values():
        for record in records:
            found = resolve(str(record.get("id", "")))
            if found is not None:
                audio_paths.append(found)
    print(
        f"{len(audio_paths)} clips have local audio; measuring with "
        f"{config.tracker} on {args.workers} worker(s)",
        flush=True,
    )

    def progress(done: int, total: int) -> None:
        if done % 200 == 0 or done == total:
            print(f"  measured {done}/{total}", flush=True)

    extraction = corpus.extract_many(
        audio_paths, config=config, workers=args.workers, progress=progress
    )
    measured = extraction.features
    print(
        f"measured {extraction.succeeded}/{extraction.attempted} "
        f"({extraction.failed} failed)",
        flush=True,
    )
    if extraction.failed:
        for path, error in list(extraction.errors.items())[:5]:
            print(f"  ! {Path(path).name}: {error}", flush=True)
    # A build that measured nothing would otherwise write a corpus of pure
    # legacy records and exit zero, which is how a silent total failure once
    # looked exactly like success.
    if extraction.failure_rate > args.max_failure_rate:
        raise SystemExit(
            f"extraction failed for {100 * extraction.failure_rate:.1f}% of clips "
            f"(limit {100 * args.max_failure_rate:.0f}%). Refusing to write a corpus "
            "that would silently mark them all legacy."
        )

    def extract(path: Path, cfg) -> dict:
        found = measured.get(str(path))
        if found is None:
            raise RuntimeError(f"no measurement for {path}")
        return found

    def resolve_measured(video_id: str) -> Path | None:
        """Only offer audio we actually managed to measure, so a file that
        failed extraction becomes a legacy record rather than an exception."""
        found = resolve(video_id)
        return found if found is not None and str(found) in measured else None

    totals = {"total": 0, "remeasured": 0, "legacy": 0, "qc_pass": 0}
    reasons: dict[str, int] = {}
    for index, (talent, records) in enumerate(sorted(by_talent.items()), start=1):
        built = corpus.build_records(
            records,
            cache=cache,
            resolve_audio=resolve_measured,
            extract=extract,
            background_ratio=ratio,
            separation_applied=separated,
            config=config,
            release=args.release,
        )
        if not args.no_plausibility:
            # Corpus-level, so it runs after the talent's series exists: a clip
            # carrying someone else's voice is plausible on its own and only
            # looks wrong beside the rest of that talent's career.
            built = plausibility.apply(
                built,
                low_semitones=args.plausibility_low,
                high_semitones=args.plausibility_high,
            )
        corpus.write_talent(args.out_dir, talent, built)
        summary = corpus.summarise(built)
        for key in totals:
            totals[key] += summary[key]
        for reason, count in summary["qc_fail_reasons"].items():
            reasons[reason] = reasons.get(reason, 0) + count
        print(
            f"[{index}/{len(by_talent)}] {talent:14} "
            f"remeasured {summary['remeasured']:4}  legacy {summary['legacy']:4}  "
            f"pass {summary['qc_pass']:4}",
            flush=True,
        )

    print(f"\n{totals}")
    print(f"qc fail reasons: {reasons}")
    print(f"-> {args.out_dir}")
    return {"totals": totals, "qc_fail_reasons": reasons}


def _run_plausibility(args) -> dict:
    """Apply the per-talent plausibility pass to an existing v2 corpus.

    Separate from `build-v2` because it needs no audio: it compares each clip
    to the talent's own baseline, so a threshold change can be re-applied to a
    built corpus in seconds instead of re-measuring it.
    """
    from vvc import corpus, plausibility

    by_talent = _load_v2_by_talent(args.v2_dir)
    total = flagged = 0
    rows = []
    for talent, records in sorted(by_talent.items()):
        marks = plausibility.implausible(
            records,
            low_semitones=args.low,
            high_semitones=args.high,
        )
        total += len(records)
        flagged += len(marks)
        for flag in marks:
            rows.append((talent, flag))
        if not args.dry_run:
            corpus.write_talent(
                args.v2_dir,
                talent,
                plausibility.apply(records, low_semitones=args.low, high_semitones=args.high),
            )

    rows.sort(key=lambda r: r[1].semitones)
    print(f"{'talent':10} {'clip':14} {'Hz':>8} {'baseline':>9} {'semis':>7}")
    for talent, flag in rows[:25]:
        print(
            f"{talent:10} {flag.id:14} {flag.value_hz:8.1f} "
            f"{flag.baseline_hz:9.1f} {flag.semitones:+7.1f}"
        )
    if len(rows) > 25:
        print(f"  ... and {len(rows) - 25} more")
    verb = "would flag" if args.dry_run else "flagged"
    print(f"\n{verb} {flagged} of {total} records ({100 * flagged / max(total, 1):.2f}%)")
    return {"flagged": flagged, "total": total}


def _run_export(args) -> None:
    from vvc import exports

    source = args.v2_dir if args.v2_dir.is_dir() and not args.v1 else None
    by_talent = (
        _load_v2_by_talent(args.v2_dir) if source else _load_v1_by_talent(args.measurements_dir)
    )
    print(f"exporting {len(by_talent)} talents from {'v2' if source else 'v1'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("clips.csv", exports.write_clips_csv),
        ("talent_month.csv", exports.write_talent_month_csv),
        ("talent_summary.csv", exports.write_talent_summary_csv),
        ("correlations.csv", exports.write_correlations_csv),
    ):
        path = fn(by_talent, args.out_dir / name)
        with path.open(encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        print(f"  {name:22} {rows:6} rows -> {path}")


def _run_restore(args) -> dict:
    """Pull archived raw fetches back and replay each record's stored window."""
    from vvc import restore

    records: list[dict] = []
    for talent_records in _load_v1_by_talent(args.measurements_dir).values():
        records.extend(talent_records)

    print(f"reading archive listing from {args.remote} ...", flush=True)
    available = restore.archive_ids(remote=args.remote)
    print(f"{len(available)} files in the archive", flush=True)

    plan = restore.plan_restore(records, args.data_dir, available=available)
    if args.limit:
        plan = plan[: args.limit]
    print(f"{len(plan)} clip(s) restorable (no local audio, archived, window known)")
    if args.dry_run:
        for row in plan[:20]:
            print(f"  {row['id']}  {row['start_s']:.0f}-{row['end_s']:.0f}s")
        return {"planned": len(plan), "dry_run": True}

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    summary = restore.run_restore(
        plan, args.data_dir, remote=args.remote, progress=progress
    )
    print(f"\n{summary}")
    print("run `vvc build-v2` to fold the restored clips into the v2 corpus")
    return summary


def _run_analyse(args) -> dict:
    """Career trends, and the identifiable part of the era confound.

    Career months = calendar time - debut, so a career slope and a calendar
    slope are NOT separately identifiable (age-period-cohort). What is
    identifiable is non-linear calendar structure shared across talents who sit
    at different career stages — that is what this reports.
    """
    from vvc import analysis, quality

    by_talent = (
        _load_v2_by_talent(args.v2_dir)
        if args.v2_dir.is_dir() and not args.v1
        else _load_v1_by_talent(args.measurements_dir)
    )

    series: dict[str, list] = {}
    trends: dict[str, dict] = {}
    for talent, records in by_talent.items():
        months: dict[str, list[float]] = {}
        for record in records:
            features = record.get("features") or {}
            passed, _ = quality.verdict_any(features)
            value = features.get(args.metric)
            month = record.get("month")
            if passed and month and isinstance(value, (int, float)) and value == value:
                months.setdefault(month, []).append(float(value))
        if len(months) < args.min_months:
            continue
        ordered = sorted(months)
        first = ordered[0]
        points = [
            analysis.Point(
                period=month,
                career_months=float(_months_between_str(first, month)),
                value=statistics.mean(months[month]),
            )
            for month in ordered
        ]
        series[talent] = points
        trends[talent] = analysis.career_trend(points)

    residuals = analysis.common_period_residuals(series, period=args.period)

    slopes = [t["slope_per_year"] for t in trends.values() if t["slope_per_year"] == t["slope_per_year"]]
    negative = sum(1 for s in slopes if s < 0)
    print(f"talents analysed: {len(series)}  (metric: {args.metric})")
    if slopes:
        print(
            f"career slope per year: median {statistics.median(slopes):+.2f}, "
            f"{negative}/{len(slopes)} negative"
        )

    print(f"\ncommon calendar-{args.period} residual after removing each talent's career trend")
    print("(a shared shock survives this; a smooth era drift is absorbed and invisible)")
    print(f"  {'period':>8}{'mean':>10}{'median':>10}{'sd':>10}{'talents':>9}")
    for key, row in residuals.items():
        sd = row["sd_across_talents"]
        print(
            f"  {key:>8}{row['mean_residual']:>10.2f}{row['median_residual']:>10.2f}"
            f"{(sd if sd is not None else float('nan')):>10.2f}{row['n_talents']:>9}"
        )

    report = {
        "metric": args.metric,
        "n_talents": len(series),
        "career_trends": trends,
        "period_residuals": residuals,
        "identification_note": (
            "Career and calendar slopes are not separately identifiable "
            "(age-period-cohort). Only non-linear shared calendar structure is "
            "reported here; a smooth era drift is absorbed by the detrend."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {args.out}")
    return report


def _months_between_str(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 12 + (int(b[5:7]) - int(a[5:7]))


def _dash_ids_argv(argv: list[str] | None, ret: argparse.ArgumentParser) -> list[str]:
    """Normalize "--ids -<id>" to "--ids=<id>" so dash-leading ids survive parsing."""
    if argv is None:
        argv = sys.argv[1:]
    options = set(ret._option_string_actions)
    out: list[str] = []
    i = 0
    while i < len(argv):
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if (
            argv[i] == "--ids"
            and nxt is not None
            and nxt.startswith("-")
            and not nxt.startswith("--")
            and nxt not in options
        ):
            out.append("--ids=" + nxt)
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vvc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cat = sub.add_parser("catalog", help="write filtered Holodex stream ids")
    cat.add_argument("-o", "--out", type=Path, default=Path("catalog.json"))
    cat.add_argument("--channel", default=None)
    cat.add_argument("--monthly", action="store_true")

    ros = sub.add_parser(
        "roster",
        help="fetch the Holodex channel roster and write the active talent set",
    )
    ros.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/catalog/roster.json"),
    )
    ros.add_argument("--org", default="Hololive")

    plan = sub.add_parser(
        "plan",
        help="per-talent v1 sampling plan from cached Holodex listings (metadata only)",
    )
    plan.add_argument(
        "--roster", type=Path, default=Path("data/catalog/roster.json")
    )
    plan.add_argument(
        "--cache-dir", type=Path, default=Path("data/catalog/video_cache")
    )
    plan.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/catalog/expansion_plan.json"),
    )

    newt = sub.add_parser(
        "new-talent",
        help=(
            "bootstrap a fresh talent: cache the raw Holodex listing for "
            "channel_id and seed an empty measurements file, so densify "
            "can run immediately after"
        ),
    )
    newt.add_argument("name", help='file-path slug, e.g. "chihaya"')
    newt.add_argument("channel_id")
    newt.add_argument(
        "--cache-dir", type=Path, default=Path("data/catalog/video_cache")
    )
    newt.add_argument(
        "--measurements-dir", type=Path, default=Path("data/measurements")
    )

    get = sub.add_parser("fetch", help="download video ids to data/audio/<id>.wav")
    get.add_argument("video_ids", nargs="*")
    get.add_argument("--data-dir", type=Path, default=Path("data"))
    get.add_argument("--ids-file", type=Path, default=None)
    get.add_argument("--cookies", type=Path, default=None)

    iso = sub.add_parser("isolate", help="vocal-isolate wavs already in data/audio")
    iso.add_argument("video_ids", nargs="*")
    iso.add_argument("--data-dir", type=Path, default=Path("data"))
    iso.add_argument("--ids-file", type=Path, default=None)
    iso.add_argument("--out-dir", type=Path, default=None)
    iso.add_argument("--model-filename", default=None)
    iso.add_argument("--windowed", action="store_true")

    win = sub.add_parser("window", help="pick the 90 s speech window and slice it")
    win.add_argument("video_ids", nargs="*")
    win.add_argument("--data-dir", type=Path, default=Path("data"))
    win.add_argument("--ids-file", type=Path, default=None)
    win.add_argument("--window-s", type=float, default=90.0)
    win.add_argument("--hop-s", type=float, default=15.0)

    mea = sub.add_parser("measure", help="measure picks on their stems")
    mea.add_argument("--picks", type=Path, required=True)
    mea.add_argument("--windows", type=Path, default=Path("data/windows/windows.json"))
    mea.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    mea.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    mea.add_argument("--legacy", action="store_true")
    mea.add_argument("-o", "--out", type=Path, default=Path("data/measurements.json"))

    plo = sub.add_parser(
        "plot",
        help=(
            "write monthly/quarterly/yearly F0/IQR plots into a fresh "
            "data/plots/runs/<run>/ directory (never overwrites a prior run)"
        ),
    )
    plo.add_argument("--measurements", type=Path, required=True)
    plo.add_argument("--out-dir", type=Path, default=Path("data/plots"))
    plo.add_argument(
        "--label",
        default=None,
        help="optional suffix for the run directory name (e.g. luna-monthly)",
    )
    plo.add_argument(
        "--talent",
        default=None,
        help='display name for plot titles (e.g. "Himemori Luna")',
    )
    plo.add_argument(
        "--registry",
        type=Path,
        default=Path("data/measurements/talents.json"),
        help=(
            "measurements-path -> display-name registry; --talent auto-"
            "registers this run into it, then the cross-talent comparison "
            "plots are regenerated from every registered talent"
        ),
    )
    plo.add_argument(
        "--no-compare",
        action="store_true",
        help="skip auto-updating the registry and regenerating the "
        "cross-talent comparison plots",
    )

    cmp = sub.add_parser(
        "plot-compare",
        help=(
            "cross-talent quarterly/yearly F0 comparison (QC-pass only) "
            "into a fresh data/plots/runs/<run>/ directory"
        ),
    )
    cmp.add_argument(
        "--talent",
        nargs=2,
        action="append",
        metavar=("NAME", "MEASUREMENTS"),
        required=True,
        dest="talents",
        help='repeatable: --talent "Display Name" path/to/measurements.json',
    )
    cmp.add_argument("--out-dir", type=Path, default=Path("data/plots"))
    cmp.add_argument(
        "--label",
        default=None,
        help="optional suffix for the run directory name",
    )

    sit = sub.add_parser(
        "site-data",
        help=(
            "write the v1 interactive site's data.js (docs/v1/, served by "
            "GitHub Pages) from the "
            "talent registry — per-talent series plus the cute/mature "
            "percentile scatter, see vvc.site_data"
        ),
    )
    sit.add_argument(
        "--registry",
        type=Path,
        default=Path("data/measurements/talents.json"),
    )
    sit.add_argument(
        "--roster",
        type=Path,
        default=Path("data/catalog/roster.json"),
        help=(
            "roster.json (talents list with english_name/group) used to "
            "attach generation/branch metadata per talent; pass a "
            "nonexistent path or empty file to fall back to "
            "group=branch='Unknown'"
        ),
    )
    sit.add_argument("-o", "--out", type=Path, default=Path("docs/v1/data.js"))

    sit2 = sub.add_parser(
        "site-data-v2",
        help=(
            "write the v2 site's data.js (docs/, served by GitHub Pages) "
            "from the same talent registry plus data/measurements/v2/ — "
            "per-talent, per-metric series with bootstrap CI and noise "
            "floor, see vvc.site_data_v2"
        ),
    )
    sit2.add_argument(
        "--registry",
        type=Path,
        default=Path("data/measurements/talents.json"),
    )
    sit2.add_argument(
        "--v2-dir", type=Path, default=Path("data/measurements/v2")
    )
    sit2.add_argument(
        "--roster",
        type=Path,
        default=Path("data/catalog/roster.json"),
        help=(
            "roster.json (talents list with english_name/group) used to "
            "attach generation/branch metadata per talent; pass a "
            "nonexistent path or empty file to fall back to "
            "group=branch='Unknown'"
        ),
    )
    sit2.add_argument("-o", "--out", type=Path, default=Path("docs/data.js"))

    ret = sub.add_parser(
        "retry",
        help=(
            "re-measure QC-failing months on a 2nd 90 s window "
            "(<id>_raw90b) and replace passing ones"
        ),
    )
    ret.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    ret.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    ret.add_argument("--data-dir", type=Path, default=Path("data"))
    ret.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    ret.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    ret.add_argument(
        "--model-file-dir",
        default=None,
        help=(
            "optional audio-separator --model_file_dir (model ckpt cache "
            "dir, e.g. data/models); default: audio-separator's own"
        ),
    )
    ret.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to retry (default: every record whose qc.pass is false)",
    )
    ret.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    ret.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    resc = sub.add_parser(
        "rescue",
        help=(
            "stem-hunt rescue: isolate the FULL wav, hunt the 90 s window "
            "on the stem (<id>_stem90) and replace passing months"
        ),
    )
    resc.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    resc.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    resc.add_argument("--data-dir", type=Path, default=Path("data"))
    resc.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    resc.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    resc.add_argument(
        "--model-file-dir",
        default="data/models",
        help=(
            "audio-separator --model_file_dir (model ckpt cache dir); "
            "default: data/models"
        ),
    )
    resc.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to rescue (default: every record whose qc.pass is false)",
    )
    resc.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    resc.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    prm = sub.add_parser(
        "remeasure-praat",
        help=(
            "re-measure EVERY record with Praat instead of numpy ACF, on "
            "the same audio each record already used; snapshots first"
        ),
    )
    prm.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    prm.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    prm.add_argument("--data-dir", type=Path, default=Path("data"))
    prm.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    prm.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    prm.add_argument("-o", "--out", type=Path, default=None)

    dsf = sub.add_parser(
        "densify",
        help=(
            "bring every month below --target-n records up to target-n "
            "using the cached raw catalog (fetch+window+isolate+Praat "
            "measure); only appends, never touches existing records"
        ),
    )
    dsf.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    dsf.add_argument(
        "--video-cache",
        type=Path,
        default=Path("data/catalog/video_cache/UCa9Y57gfeY0Zro_noHRVrnw.json"),
        help="cached raw Holodex listing for the channel (no new API calls)",
    )
    dsf.add_argument(
        "--windows", type=Path, default=Path("data/windows/windows.json")
    )
    dsf.add_argument("--data-dir", type=Path, default=Path("data"))
    dsf.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    dsf.add_argument("--target-n", type=int, default=2)
    dsf.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help=(
            "clips whose window-hunt/isolate/measure stages may run "
            "concurrently (fetch always stays sequential); default 1 "
            "reproduces today's fully sequential behavior"
        ),
    )
    dsf.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    dsf.add_argument(
        "--model-file-dir",
        default=None,
        help="optional audio-separator --model_file_dir (model ckpt cache dir)",
    )
    dsf.add_argument(
        "--offload-remote",
        default=None,
        help=(
            "rclone remote (e.g. 'Google Drive:vanalysis-raw-audio') to "
            "upload a QC-pass id's raw wav to, deleting it locally once "
            "confirmed uploaded; default None disables offload entirely "
            "(a QC-fail id's raw wav is never offloaded, regardless)"
        ),
    )
    dsf.add_argument("--cookies", type=Path, default=None)
    dsf.add_argument(
        "--dry-run",
        action="store_true",
        help="compute everything possible but write nothing",
    )

    diag = sub.add_parser(
        "diagnose-tracker",
        help=(
            "read-only: compare numpy-ACF vs Praat on audio already on "
            "disk (raw90/raw90b/stem90); never touches --measurements"
        ),
    )
    diag.add_argument(
        "--measurements",
        type=Path,
        default=Path("data/measurements/luna_monthly.json"),
    )
    diag.add_argument("--data-dir", type=Path, default=Path("data"))
    diag.add_argument("--stems-dir", type=Path, default=Path("data/stems_fast"))
    diag.add_argument("--model-filename", default=DEFAULT_MODEL_FILENAME)
    diag.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="ids to compare (default: every record whose qc.pass is false)",
    )
    diag.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "file with one video id per line, appended after --ids "
            "(accepts dash-leading ids like -DwvlhziHBI)"
        ),
    )
    diag.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/measurements/luna_tracker_diagnostic.json"),
    )

    rel = sub.add_parser(
        "reliability",
        help=(
            "read-only: the measurement's own noise floor (same-talent, "
            "same-month clip disagreement) plus discordance review flags"
        ),
    )
    rel.add_argument(
        "--measurements-dir",
        type=Path,
        default=Path("data/measurements"),
        help="directory of *_monthly.json files (one per talent)",
    )
    rel.add_argument(
        "--semitone-threshold",
        type=float,
        default=reliability.DEFAULT_SEMITONE_THRESHOLD,
        help=(
            "flag same-month pairs disagreeing by more than this many "
            "semitones (default: half an octave)"
        ),
    )
    rel.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data/logs/reliability.json"),
    )

    bake = sub.add_parser(
        "bakeoff",
        help=(
            "read-only: compare pitch trackers on synthetic and real audio, "
            "and measure how far each feature moves under re-encoding"
        ),
    )
    bake.add_argument("--measurements-dir", type=Path, default=Path("data/measurements"))
    bake.add_argument("--data-dir", type=Path, default=Path("data"))
    bake.add_argument("-n", type=int, default=120, help="real-audio sample size")
    bake.add_argument("--seed", type=int, default=0)
    bake.add_argument("--floor", type=float, default=60.0)
    bake.add_argument("--ceiling", type=float, default=800.0)
    bake.add_argument(
        "--trackers", nargs="*", default=None, help="subset of trackers (default: all)"
    )
    bake.add_argument(
        "--required-ids",
        nargs="*",
        default=None,
        help="ids always included (e.g. the known-bad clips)",
    )
    bake.add_argument("--skip-synthetic", action="store_true")
    bake.add_argument("--skip-era", action="store_true")
    bake.add_argument("--era-n", type=int, default=12, help="clips for the era test")
    bake.add_argument("--era-tracker", default="praat_ac")
    bake.add_argument("-o", "--out", type=Path, default=Path("data/logs/bakeoff.json"))

    bld = sub.add_parser(
        "build-v2",
        help=(
            "re-measure local audio into the v2 corpus (data/measurements/v2/); "
            "never modifies the v1 measurement files"
        ),
    )
    bld.add_argument("--measurements-dir", type=Path, default=Path("data/measurements"))
    bld.add_argument("--out-dir", type=Path, default=Path("data/measurements/v2"))
    bld.add_argument("--data-dir", type=Path, default=Path("data"))
    bld.add_argument("--cache-dir", type=Path, default=Path("data/catalog/video_cache"))
    # Defaults follow FeatureConfig rather than repeating its values, so the
    # CLI and the library cannot drift apart.
    bld.add_argument("--tracker", default=_FEATURE_DEFAULTS.tracker)
    bld.add_argument("--floor", type=float, default=_FEATURE_DEFAULTS.floor)
    bld.add_argument("--ceiling", type=float, default=_FEATURE_DEFAULTS.ceiling)
    bld.add_argument(
        "--separation-threshold-db",
        type=float,
        default=_BACKGROUND_THRESHOLD_DB,
        help=(
            "separate a clip only when its residual-to-voice ratio reaches this; "
            "below it the raw window is measured directly"
        ),
    )
    bld.add_argument(
        "--always-separate",
        action="store_true",
        help="measure the separated stem for every clip, as v1 did",
    )
    bld.add_argument(
        "--no-plausibility",
        action="store_true",
        help="skip the per-talent plausibility pass (keeps clips that look like another speaker)",
    )
    bld.add_argument(
        "--plausibility-low",
        type=float,
        default=_PLAUSIBILITY_LOW,
        help="semitones below the talent's own baseline before a clip is rejected",
    )
    bld.add_argument(
        "--plausibility-high",
        type=float,
        default=_PLAUSIBILITY_HIGH,
        help="semitones above (deliberately loose: a performance and a wide range are alike)",
    )
    bld.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.05,
        help=(
            "abort rather than write a corpus if more than this fraction of "
            "clips fail to measure (a total failure otherwise looks like a "
            "build where every clip merely lacked audio)"
        ),
    )
    bld.add_argument("--release", default="v2-dev")
    bld.add_argument("--talents", nargs="*", default=None)
    bld.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel extraction processes (CPU-bound work; 8 is a good default here)",
    )

    exp = sub.add_parser(
        "export",
        help="write the four flat CSV tables (clips, talent-month, summary, correlations)",
    )
    exp.add_argument("--v2-dir", type=Path, default=Path("data/measurements/v2"))
    exp.add_argument("--measurements-dir", type=Path, default=Path("data/measurements"))
    exp.add_argument("--v1", action="store_true", help="export the v1 corpus instead")
    exp.add_argument("-o", "--out-dir", type=Path, default=Path("data/exports"))

    ana = sub.add_parser(
        "analyse",
        help=(
            "read-only: per-talent career trends, plus the identifiable part of "
            "the career-vs-recording-era confound"
        ),
    )
    ana.add_argument("--v2-dir", type=Path, default=Path("data/measurements/v2"))
    ana.add_argument("--measurements-dir", type=Path, default=Path("data/measurements"))
    ana.add_argument("--v1", action="store_true", help="analyse the v1 corpus instead")
    ana.add_argument("--metric", default="median_f0")
    ana.add_argument("--period", default="year", choices=["year", "month"])
    ana.add_argument("--min-months", type=int, default=12)
    ana.add_argument("-o", "--out", type=Path, default=Path("data/logs/analysis.json"))

    rst = sub.add_parser(
        "restore",
        help=(
            "pull archived raw fetches back from the remote and replay each "
            "record's stored window, so clips whose local audio was offloaded "
            "become measurable again without re-fetching from YouTube"
        ),
    )
    rst.add_argument("--measurements-dir", type=Path, default=Path("data/measurements"))
    rst.add_argument("--data-dir", type=Path, default=Path("data"))
    rst.add_argument("--remote", default=_RESTORE_REMOTE)
    rst.add_argument("--limit", type=int, default=None)
    rst.add_argument("--dry-run", action="store_true")

    pla = sub.add_parser(
        "plausibility",
        help=(
            "apply the per-talent plausibility pass to an existing v2 corpus "
            "(needs no audio; re-runnable after a threshold change)"
        ),
    )
    pla.add_argument("--v2-dir", type=Path, default=Path("data/measurements/v2"))
    pla.add_argument("--low", type=float, default=_PLAUSIBILITY_LOW)
    pla.add_argument("--high", type=float, default=_PLAUSIBILITY_HIGH)
    pla.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(_dash_ids_argv(argv, ret))
    if args.cmd == "plausibility":
        _run_plausibility(args)
        return
    if args.cmd == "restore":
        _run_restore(args)
        return
    if args.cmd == "analyse":
        _run_analyse(args)
        return
    if args.cmd == "build-v2":
        _run_build_v2(args)
        return
    if args.cmd == "export":
        _run_export(args)
        return
    if args.cmd == "bakeoff":
        _run_bakeoff(args)
        return
    if args.cmd == "reliability":
        _run_reliability(
            args.measurements_dir, args.out, semitone_threshold=args.semitone_threshold
        )
        return
    if args.cmd == "roster":
        talents = filter_talents(
            fetch_channels(org=args.org, api_key=_holodex_key())
        )
        write_roster(talents, args.out)
        groups: dict[str, int] = {}
        for row in talents:
            groups[row["group"]] = groups.get(row["group"], 0) + 1
        print(f"{len(talents)} talents -> {args.out}")
        for group in sorted(groups):
            print(f"  {group}: {groups[group]}")
        return
    if args.cmd == "plan":
        payload = run_plan(args.roster, args.cache_dir, args.out, api_key=_holodex_key())
        print(
            f"{payload['talents_planned']}/{payload['roster_count']} talents, "
            f"{payload['total_picks']} picks, ~{payload['est_disk_gb']} GB -> {args.out}"
        )
        for record in payload["talents"]:
            if record["pick_count"] == 0:
                print(f"  zero eligible picks: {record['name']} ({record['group']})")
        return
    if args.cmd == "new-talent":
        videos = fetch_channel_videos(
            args.channel_id, api_key=_holodex_key(), cache_dir=args.cache_dir
        )
        cache_path = Path(args.cache_dir) / f"{args.channel_id}.json"
        measurements_path = Path(args.measurements_dir) / f"{args.name}_monthly.json"
        if measurements_path.is_file():
            print(f"measurements already exist, left untouched -> {measurements_path}")
        else:
            measurements_path.parent.mkdir(parents=True, exist_ok=True)
            measurements_path.write_text("[]\n", encoding="utf-8")
            print(f"measurements seeded -> {measurements_path}")
        print(f"{len(videos)} videos cached -> {cache_path}")
        print(
            "next: vvc densify --measurements "
            f"{measurements_path} --video-cache {cache_path} "
            "--target-n 3 --cpu-workers 4"
        )
        return
    if args.cmd == "catalog":
        kept = filter_videos(_list_holodex(_holodex_key(), channel=args.channel))
        if args.monthly:
            kept = pick_monthly(kept)
        args.out.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
        print(f"{len(kept)} rows -> {args.out}")
        return
    if args.cmd == "measure":
        picks = json.loads(args.picks.read_text(encoding="utf-8"))
        model_filename = None if args.legacy else args.model_filename
        entries = run_monthly(
            picks, args.windows, args.stems_dir, args.out, model_filename=model_filename
        )
        print(f"{len(entries)} entries -> {args.out}")
        return
    if args.cmd == "plot":
        entries = json.loads(args.measurements.read_text(encoding="utf-8"))
        run_dir = new_run_dir(args.out_dir, args.label)
        write_plots(entries, run_dir, talent=args.talent)
        write_quarterly_plots(entries, run_dir, talent=args.talent)
        write_yearly_plot(entries, run_dir, talent=args.talent)
        present_keys = {
            key
            for entry in entries
            for key in (entry.get("features") or {})
        }
        for feature_key, filename, subject, subtitle, unit_label, caveat in _EXTRA_FEATURE_PLOTS:
            if feature_key in present_keys:
                write_feature_yearly_plot(
                    entries, run_dir,
                    feature_key=feature_key, filename=filename,
                    subject=subject, subtitle=subtitle, unit_label=unit_label,
                    talent=args.talent, caveat=caveat,
                )
        if not args.no_compare:
            registry = _load_registry(args.registry)
            if args.talent:
                registry[str(args.measurements)] = args.talent
                _save_registry(args.registry, registry)
            talents = _load_registered_talents(registry)
            if len(talents) >= 2:
                _write_all_comparison_plots(talents, run_dir)
                print(f"comparison plots ({', '.join(sorted(talents))}) -> {run_dir}")
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "plot-compare":
        talents = {
            name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in args.talents
        }
        run_dir = new_run_dir(args.out_dir, args.label)
        _write_all_comparison_plots(talents, run_dir)
        print(f"plots -> {run_dir}")
        return
    if args.cmd == "site-data":
        registry = _load_registry(args.registry)
        roster = None
        if args.roster.is_file():
            roster = json.loads(args.roster.read_text(encoding="utf-8")).get("talents")
        payload = write_site_data(registry, args.out, roster=roster)
        print(f"site data ({', '.join(sorted(payload['talents']))}) -> {args.out}")
        return
    if args.cmd == "site-data-v2":
        registry = _load_registry(args.registry)
        roster = None
        if args.roster.is_file():
            roster = json.loads(args.roster.read_text(encoding="utf-8")).get("talents")
        payload = write_site_data_v2(
            registry, args.out, v2_dir=args.v2_dir, roster=roster
        )
        fallback = sum(1 for t in payload["talents"].values() if t["legacy_fallback"])
        print(
            f"v2 site data ({len(payload['talents'])} talents, "
            f"{fallback} v1-fallback) -> {args.out}"
        )
        return
    if args.cmd == "retry":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        summary = run_retry(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "rescue":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        summary = run_rescue(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "densify":
        cookies = args.cookies
        if cookies is None:
            fallback = args.data_dir / "youtube.cookies.txt"
            if fallback.is_file():
                cookies = fallback
        summary = run_densify(
            args.data_dir,
            measurements_path=args.measurements,
            video_cache_path=args.video_cache,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            target_n=args.target_n,
            model_filename=args.model_filename,
            model_file_dir=args.model_file_dir,
            cookies=cookies,
            dry_run=args.dry_run,
            log=print,
            cpu_workers=args.cpu_workers,
            offload_remote=args.offload_remote,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "remeasure-praat":
        summary = run_remeasure(
            args.data_dir,
            measurements_path=args.measurements,
            windows_path=args.windows,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            out_path=args.out,
        )
        print(json.dumps(summary, indent=2))
        return
    if args.cmd == "diagnose-tracker":
        ids = list(args.ids) if args.ids else []
        if args.ids_file is not None:
            ids.extend(
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if args.ids_file is not None and not ids:
            parser.error("give at least one video id via --ids or --ids-file")
        results = run_diagnose(
            ids or None,
            args.data_dir,
            measurements_path=args.measurements,
            stems_dir=args.stems_dir,
            model_filename=args.model_filename,
            out_path=args.out,
        )
        print(f"{len(results)} comparison records -> {args.out}")
        return
    ids = list(args.video_ids)
    if args.ids_file is not None:
        ids.extend(
            line.strip()
            for line in args.ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not ids:
        parser.error("give at least one video id or --ids-file")
    if args.cmd == "window":
        windows_dir = args.data_dir / "windows"
        windows_dir.mkdir(parents=True, exist_ok=True)
        index_path = windows_dir / "windows.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8"))
            if index_path.is_file()
            else {}
        )
        for video_id in ids:
            dest = raw90_path(video_id, args.data_dir)
            if dest.is_file():
                print(f"skip {dest} (already windowed)")
                continue
            src = audio_path(video_id, args.data_dir)
            if not src.is_file():
                print(f"missing {src}", file=sys.stderr)
                continue
            start_s, end_s = best_speech_window(
                src, window_s=args.window_s, hop_s=args.hop_s
            )
            slice_wav(src, dest, start_s, end_s)
            index[video_id] = {"start_s": start_s, "end_s": end_s}
            print(f"{video_id} {start_s:.1f}-{end_s:.1f} -> {dest}")
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return
    if args.cmd == "isolate":
        if args.windowed:
            out_dir = (
                args.out_dir if args.out_dir is not None else args.data_dir / "stems_fast"
            )
        else:
            out_dir = (
                args.out_dir if args.out_dir is not None else args.data_dir / "stems"
            )
        for video_id in ids:
            if args.windowed:
                src_path = args.data_dir / "windows" / f"{video_id}_raw90.wav"
            else:
                src_path = audio_path(video_id, args.data_dir)
            dest = vocals_path(src_path, out_dir, model_filename=args.model_filename)
            if dest.is_file() and dest.stat().st_size > 1_000_000:
                print(f"skip {dest}")
                continue
            if not src_path.is_file():
                print(f"missing {src_path}", file=sys.stderr)
                continue
            print(isolate_vocals(src_path, out_dir, model_filename=args.model_filename))
        return
    cookies = args.cookies
    if cookies is None:
        fallback = args.data_dir / "youtube.cookies.txt"
        if fallback.is_file():
            cookies = fallback
    todo = []
    for video_id in ids:
        dest = audio_path(video_id, args.data_dir)
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"skip {dest}")
            continue
        if raw90_path(video_id, args.data_dir).is_file():
            print(f"skip {dest} (already windowed, raw wav not needed)")
            continue
        todo.append(video_id)
    try:
        for video_id, dest in fetch_audio_many(
            todo, args.data_dir, cookies=cookies
        ).items():
            if dest is not None:
                print(dest)
    except BotCheckDetected as exc:
        print(
            f"stopped: YouTube bot-check triggered on {exc.video_id} — "
            "this is a session/IP-level signal, not a per-video problem; "
            "further fetches in this batch were not attempted. Refresh "
            "cookies (see CLAUDE.md) before retrying.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
