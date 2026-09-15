"""Aggregate export for the v2 site (``docs/``, replacing the placeholder
that pointed at ``docs/v1/``).

Pure data, same spirit as v1's ``site_data.py``: JSON-serializable Python
values only, so the frontend never re-derives a statistic and this module is
testable without a browser. It deliberately does not reuse v1's aggregation
(``series.py``'s mean-at-month/quarter, ``median_low``-at-year) — see
``series_v2.py`` for why a v2 site needs its own month-first, ordinary-median,
bootstrap-CI aggregation.

Metric families mirror reference/statistics.md's "Presenting fewer things
than there are columns" section exactly, and each family's ``robust`` flag
mirrors reference/measurement.md's per-feature validity table. Getting these
two documents and this module out of sync would let the site claim more than
the methodology supports, so a family/metric added here should be added there
first.

Talent branch/generation resolution (roster matching, name aliases, the
graduated-talent lookup, multi-generation overrides) is NOT duplicated here —
it is imported from ``site_data`` and reused as-is. That table is hand-curated
real-world fact, and two independently-maintained copies would drift exactly
the way reference/qc.md warns a stored-vs-recomputed verdict drifts.

Talent colors are assigned client-side (``docs/app.js``), not here — see that
file for why.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from . import series_v2
from .quality import verdict_any
from .reliability import corpus_noise_floor, noise_floor
from .site_data import _generation_order, _talent_groups_and_branch

Loader = Callable[[str], list[dict]]

#: reference/statistics.md's five families, in the order that document lists
#: them. ``robust`` is None for the context/quality family: those metrics are
#: recording/selection diagnostics, not claims about a voice, so the
#: robust/experimental badge (which is about measuring the voice faithfully)
#: does not apply to them the same way.
FAMILIES: tuple[dict, ...] = (
    {
        "key": "pitch_level",
        "label": "Pitch level",
        "robust": True,
        "note": "The one feature that survives the separation chain cleanly.",
        "metrics": ("median_f0",),
    },
    {
        "key": "pitch_movement",
        "label": "Pitch movement",
        "robust": True,
        "note": (
            "Scale-free spread and frame-to-frame movement; inherits any "
            "octave error in the underlying pitch track."
        ),
        "metrics": ("f0_iqr_semitones", "dynamism_semitones"),
    },
    {
        "key": "periodicity_noise",
        "label": "Periodicity & noise",
        "robust": False,
        "note": (
            "Calibrated for a sustained vowel, not conversational speech, and "
            "altered by the vocal separator — harmonicity in particular is "
            "inflated by it systematically, since denoising raises harmonicity "
            "whether or not the voice changed."
        ),
        "metrics": ("jitter_local", "shimmer_local", "hnr_db"),
    },
    {
        "key": "spectrum_resonance",
        "label": "Spectrum & resonance",
        "robust": False,
        "note": (
            "Reshaped directly by the separator and by mic/EQ differences "
            "between talents and eras. The four formants move together "
            "strongly enough to read as one resonance summary rather than "
            "four independent axes; trust the shape within one talent over a "
            "cross-talent ranking."
        ),
        "metrics": ("brightness_hz", "f1_hz", "f2_hz", "f3_hz", "f4_hz"),
    },
    {
        "key": "context_quality",
        "label": "Context & measurement quality",
        "robust": None,
        "note": (
            "Describes the recording and the sampling rule, not the voice: "
            "how much of the window was voiced, how dynamic the loudness "
            "was, and how much competing sound the source contained."
        ),
        "metrics": ("voiced_fraction", "loudness_dynamics_db", "background_ratio_db"),
    },
)

ALL_METRICS: tuple[str, ...] = tuple(
    metric for family in FAMILIES for metric in family["metrics"]
)


def _default_loader(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _v2_path_for(v1_path: str, v2_dir: Path) -> str:
    stem = Path(v1_path).stem
    if stem.endswith("_monthly"):
        stem = stem[: -len("_monthly")]
    return str(v2_dir / f"{stem}.json")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    """0-100 rank percentile (0 = lowest, 100 = highest). Every name tied at
    the same value lands at the midpoint 50.0 instead of an arbitrary
    tie-break order."""
    names = sorted(values)
    if max(values.values()) == min(values.values()):
        return {name: 50.0 for name in names}
    ranked = sorted(names, key=lambda name: values[name])
    n = len(names)
    return {name: 100.0 * rank / (n - 1) for rank, name in enumerate(ranked)}


def _legacy_fraction(records: list[dict]) -> float:
    if not records:
        return 0.0
    legacy = sum(1 for r in records if r.get("legacy"))
    return legacy / len(records)


def _reliability_band_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        band = record.get("reliability_band") or "unknown"
        counts[band] = counts.get(band, 0) + 1
    return counts


def build_site_data_v2(
    registry: dict[str, str],
    *,
    v2_dir: Path | str = Path("data/measurements/v2"),
    roster: list[dict] | None = None,
    v2_loader: Loader | None = None,
    v1_loader: Loader | None = None,
) -> dict:
    """Build the full v2 export.

    ``registry`` is talents.json's ``{v1_measurements_path: display_name}``
    shape, reused so the v2 site names talents identically to v1's. Each
    entry's v2 file is derived by stripping v1's ``_monthly`` suffix from the
    stem (the convention ``corpus.write_talent`` uses). A talent whose v2 file
    does not exist at all (``build-v2`` never ran for them) falls back to
    their v1 records directly, rather than vanishing from the site.
    """
    v2_dir = Path(v2_dir)
    load_v2 = v2_loader if v2_loader is not None else _default_loader
    load_v1 = v1_loader if v1_loader is not None else _default_loader

    roster_by_english_name = {
        entry["english_name"]: entry["group"]
        for entry in (roster or [])
        if entry.get("english_name") and entry.get("group")
    }

    talents_records: dict[str, list[dict]] = {}
    legacy_fallback: dict[str, bool] = {}
    for v1_path, name in registry.items():
        v2_path = _v2_path_for(v1_path, v2_dir)
        try:
            records = load_v2(v2_path)
            legacy_fallback[name] = False
        except FileNotFoundError:
            records = load_v1(v1_path)
            legacy_fallback[name] = True
        talents_records[name] = records

    # Per-metric typical value for every talent, needed before percentiles
    # can be ranked (a percentile compares across the whole included set).
    typicals: dict[str, dict[str, float]] = {name: {} for name in talents_records}
    summaries: dict[str, dict[str, dict]] = {name: {} for name in talents_records}
    for name, records in talents_records.items():
        for metric in ALL_METRICS:
            summary = series_v2.typical_and_trend(records, metric)
            summaries[name][metric] = summary
            if _finite(summary["typical"]):
                typicals[name][metric] = summary["typical"]

    percentiles: dict[str, dict[str, float]] = {name: {} for name in talents_records}
    for metric in ALL_METRICS:
        by_talent = {
            name: values[metric] for name, values in typicals.items() if metric in values
        }
        if len(by_talent) < 2:
            continue
        ranked = _rank_percentiles(by_talent)
        for name, pct in ranked.items():
            percentiles[name][metric] = pct

    talents: dict[str, dict] = {}
    groups_present: set[str] = set()
    for name, records in talents_records.items():
        if roster is None:
            groups, branch = ["Unknown"], "Unknown"
        else:
            groups, branch = _talent_groups_and_branch(name, roster_by_english_name)
        groups_present.update(groups)

        metrics: dict[str, dict] = {}
        for metric in ALL_METRICS:
            summary = summaries[name][metric]
            entry = {
                "typical": summary["typical"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "trend": summary["trend"],
                "noise_floor": noise_floor(records, feature_keys=(metric,))[metric],
                "monthly": series_v2.monthly_series(records, metric),
                "quarterly": series_v2.quarterly_series(records, metric),
                "yearly": series_v2.yearly_series(records, metric),
            }
            if metric in percentiles[name]:
                entry["percentile"] = percentiles[name][metric]
            metrics[metric] = entry

        months = sorted({r.get("month") for r in records if r.get("month")})
        n_pass = sum(1 for r in records if verdict_any(r.get("features") or {})[0])
        talents[name] = {
            "group": groups,
            "branch": branch,
            "legacy_fallback": legacy_fallback[name],
            "legacy_fraction": _legacy_fraction(records),
            "n_clips": len(records),
            "n_pass": n_pass,
            "months_covered": len(months),
            "first_month": months[0] if months else None,
            "last_month": months[-1] if months else None,
            "reliability_band": _reliability_band_counts(records),
            "metrics": metrics,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": [dict(family) for family in FAMILIES],
        "corpus_noise_floor": corpus_noise_floor(talents_records, feature_keys=ALL_METRICS),
        "generation_order": _generation_order(groups_present),
        "talents": talents,
    }


def write_site_data_v2(
    registry: dict[str, str],
    out_path: Path | str,
    *,
    v2_dir: Path | str = Path("data/measurements/v2"),
    roster: list[dict] | None = None,
    v2_loader: Loader | None = None,
    v1_loader: Loader | None = None,
) -> dict:
    """Writes ``window.SITE_DATA_V2 = {...};`` — a plain script the page
    loads directly, matching v1's convention (see site_data.write_site_data
    for why: a static, no-server ``file://`` page cannot ``fetch()`` a
    separate JSON file past CORS)."""
    payload = build_site_data_v2(
        registry, v2_dir=v2_dir, roster=roster, v2_loader=v2_loader, v1_loader=v1_loader
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    out.write_text(f"window.SITE_DATA_V2 = {body};\n", encoding="utf-8")
    return payload
