"""Flat exports: the corpus in a form someone can check.

Four non-destructive tables. The point is that a reader can reproduce any
published figure from the measurements instead of taking prose on trust, and
that the exclusions are visible — failing clips appear in the clip table with
their reason, rather than quietly not being there.

The correlation table carries the rule worth stating outright: between-talent
and within-talent correlations are reported separately and never pooled. "Do
higher-pitched people sound brighter?" and "when this person's pitch rises,
does their brightness rise?" are different questions with different answers,
and a pooled coefficient is a mixture of the two that answers neither.

See reference/statistics.md.
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from . import analysis, reliability
from .quality import verdict_any

#: Metrics carried through the flat exports.
#:
#: Both spread fields are listed so one export works on either corpus: v1
#: recorded spread in Hz, v2 in semitones, and whichever a record lacks simply
#: comes out blank. Emitting only the v2 name would silently produce an empty
#: column for the entire published corpus.
DEFAULT_METRICS = (
    "median_f0",
    "f0_iqr_semitones",
    "f0_iqr",
    "voiced_fraction",
    "brightness_hz",
    "dynamism_semitones",
    "jitter_local",
    "shimmer_local",
    "hnr_db",
    "loudness_dynamics_db",
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "f4_hz",
)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Correlation, or NaN when it is undefined.

    NaN — never 0.0 — when there are too few points or no variance. Zero would
    read as "measured, and unrelated"; the truth is "not measurable".
    """
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    if len(pairs) < 3:
        return math.nan
    xs_, ys_ = zip(*pairs)
    mx, my = statistics.mean(xs_), statistics.mean(ys_)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs_))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys_))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def _passing(records: list[dict]) -> list[dict]:
    """Clips passing the CURRENT gate, recomputed rather than read back.

    The stored ``qc`` block is a convenience copy that goes stale the moment a
    threshold moves. v1 shipped exactly this bug — its plots recomputed the
    verdict while its site export read the stored one, so the two could
    disagree about which clips existed. Recomputing here keeps the exports
    consistent with the noise floor, which also recomputes.
    """
    return [r for r in records if verdict_any(r.get("features") or {})[0]]


def _monthly_means(records: list[dict], metric: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in _passing(records):
        value = (record.get("features") or {}).get(metric)
        month = record.get("month")
        if month and _finite(value):
            buckets.setdefault(month, []).append(float(value))
    return {month: statistics.mean(values) for month, values in buckets.items()}


def _write(path: Path | str, fieldnames: list[str], rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in fieldnames})
    return path


def _cell(value: object) -> object:
    """Empty string for an absent value, never 0. A blank cell reads as
    missing; a zero reads as a measurement."""
    if value is None:
        return ""
    if isinstance(value, float) and not _finite(value):
        return ""
    return value


# --------------------------------------------------------------------------


def write_clips_csv(
    by_talent: dict[str, list[dict]],
    path: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Path:
    """One row per measurement, passing or not.

    Failing clips are included with their reason so a reader can see what was
    excluded and why — an export that silently omits them would make the QC
    gate unauditable.
    """
    fields = [
        "talent", "id", "month", "available_at", "topic_id", "reliability_band",
        "score", "window_start_s", "window_end_s", "tracker", "model",
        "background_ratio_db", "separation_applied", "legacy",
        "qc_pass", "qc_reason", *metrics,
    ]
    rows = []
    for talent in sorted(by_talent):
        for record in by_talent[talent]:
            features = record.get("features") or {}
            window = record.get("window") or {}
            qc = record.get("qc") or {}
            row = {
                "talent": talent,
                "id": record.get("id"),
                "month": record.get("month"),
                "available_at": record.get("available_at"),
                "topic_id": record.get("topic_id"),
                "reliability_band": record.get("reliability_band"),
                "score": record.get("score"),
                "window_start_s": window.get("start_s"),
                "window_end_s": window.get("end_s"),
                "tracker": record.get("tracker"),
                "model": record.get("model"),
                "background_ratio_db": record.get("background_ratio_db"),
                "separation_applied": record.get("separation_applied"),
                "legacy": record.get("legacy"),
                "qc_pass": qc.get("pass"),
                "qc_reason": qc.get("reason"),
            }
            row.update({metric: features.get(metric) for metric in metrics})
            rows.append(row)
    return _write(path, fields, rows)


def write_talent_month_csv(
    by_talent: dict[str, list[dict]],
    path: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Path:
    """One row per talent-month: the reporting unit.

    ``within_month_abs_diff`` is that month's own contribution to the noise
    floor. A month with one clip leaves it blank rather than 0 — one
    measurement cannot agree with itself, and 0 would read as perfect
    agreement.
    """
    fields = ["talent", "month", "n_clips", "n_pass", "within_month_abs_diff", *metrics]
    rows = []
    for talent in sorted(by_talent):
        records = by_talent[talent]
        pairs_by_month: dict[str, list[float]] = {}
        for pair in reliability.within_month_pairs(records, feature_key="median_f0"):
            pairs_by_month.setdefault(pair.month, []).append(pair.abs_diff)

        months = sorted({r.get("month") for r in records if r.get("month")})
        means = {metric: _monthly_means(records, metric) for metric in metrics}
        for month in months:
            in_month = [r for r in records if r.get("month") == month]
            diffs = pairs_by_month.get(month) or []
            row = {
                "talent": talent,
                "month": month,
                "n_clips": len(in_month),
                "n_pass": len(_passing(in_month)),
                "within_month_abs_diff": statistics.median(diffs) if diffs else None,
            }
            row.update({metric: means[metric].get(month) for metric in metrics})
            rows.append(row)
    return _write(path, fields, rows)


def write_talent_summary_csv(
    by_talent: dict[str, list[dict]],
    path: Path | str,
    *,
    metric: str = "median_f0",
) -> Path:
    """One row per talent: the numbers a talent card should quote.

    Career time is measured from the talent's own first observed month, which
    is a proxy for debut — a talent whose earliest streams were privated will
    have their career clock start late. The trend is Theil-Sen, and its fit
    quality travels with it.
    """
    fields = [
        "talent", "n_clips", "n_pass", "months_covered", "first_month", "last_month",
        f"{metric}_typical", f"{metric}_ci_low", f"{metric}_ci_high",
        f"{metric}_slope_per_year", f"{metric}_trend_r2", f"{metric}_span_months",
        f"noise_floor_{metric}", f"noise_floor_{metric}_semitones", "n_review_flags",
    ]
    rows = []
    for talent in sorted(by_talent):
        records = by_talent[talent]
        means = _monthly_means(records, metric)
        months = sorted(means)
        floor = reliability.noise_floor(records, feature_keys=(metric,))[metric]
        flags = reliability.discordance_flags(records)

        points = []
        if months:
            first = months[0]
            for month in months:
                points.append(
                    analysis.Point(
                        period=month,
                        career_months=float(_months_between(first, month)),
                        value=means[month],
                    )
                )
        trend = analysis.career_trend(points) if points else {}
        values = [means[m] for m in months]
        ci_low, ci_high = analysis.bootstrap_ci(values) if len(values) > 1 else (math.nan, math.nan)

        rows.append(
            {
                "talent": talent,
                "n_clips": len(records),
                "n_pass": len(_passing(records)),
                "months_covered": len(months),
                "first_month": months[0] if months else None,
                "last_month": months[-1] if months else None,
                f"{metric}_typical": statistics.median(values) if values else None,
                f"{metric}_ci_low": ci_low,
                f"{metric}_ci_high": ci_high,
                f"{metric}_slope_per_year": trend.get("slope_per_year"),
                f"{metric}_trend_r2": trend.get("r2"),
                f"{metric}_span_months": trend.get("span_months"),
                f"noise_floor_{metric}": floor["median_abs_diff"],
                f"noise_floor_{metric}_semitones": floor["median_abs_semitones"],
                "n_review_flags": len(flags),
            }
        )
    return _write(path, fields, rows)


def _months_between(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)


def write_correlations_csv(
    by_talent: dict[str, list[dict]],
    path: Path | str,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> Path:
    """One row per metric pair, with between- and within-talent correlations
    side by side and no pooled coefficient anywhere.

    Between-talent uses one point per talent (their typical value), so a
    densely sampled talent cannot dominate. Within-talent correlates the two
    metrics over a talent's months and reports the median across talents, with
    the count, so a figure driven by two talents is visible as such.
    """
    fields = [
        "metric_a", "metric_b",
        "between_talent_r", "between_talent_n",
        "within_talent_r", "within_talent_n_talents",
    ]
    talent_means = {
        metric: {
            talent: statistics.median(list(_monthly_means(records, metric).values()))
            for talent, records in by_talent.items()
            if _monthly_means(records, metric)
        }
        for metric in metrics
    }

    rows = []
    for metric_a, metric_b in itertools.combinations(metrics, 2):
        shared = sorted(set(talent_means[metric_a]) & set(talent_means[metric_b]))
        between = pearson(
            [talent_means[metric_a][t] for t in shared],
            [talent_means[metric_b][t] for t in shared],
        )

        within_values = []
        for records in by_talent.values():
            a_months = _monthly_means(records, metric_a)
            b_months = _monthly_means(records, metric_b)
            common = sorted(set(a_months) & set(b_months))
            if len(common) < 4:
                continue
            r = pearson([a_months[m] for m in common], [b_months[m] for m in common])
            if _finite(r):
                within_values.append(r)

        rows.append(
            {
                "metric_a": metric_a,
                "metric_b": metric_b,
                "between_talent_r": between,
                "between_talent_n": len(shared),
                "within_talent_r": statistics.median(within_values) if within_values else None,
                "within_talent_n_talents": len(within_values),
            }
        )
    return _write(path, fields, rows)
