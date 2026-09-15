"""Month-first aggregation for the v2 site, per reference/statistics.md.

A month's value is computed from its clips; a quarter or year is computed
from its *months' values*, never by pooling clips directly. Pooling would let
a densely-sampled month outweigh a sparse one, turning uneven sampling effort
into apparent signal. The same estimator — the ordinary median — is used at
every level, and uncertainty on an aggregate is a bootstrap CI over the
values that produced it, never min-max (which is reported alongside it as a
labelled range, not a confidence interval).

This intentionally does not reuse v1's ``series.py`` (mean at monthly/
quarterly, ``median_low`` at yearly — the estimator inconsistency
statistics.md corrects) or ``exports.py``'s single-summary helpers (mean per
month, feeding one talent-level CI). Building a full per-point time series
with its own CI at every granularity is new: it did not exist before the
site needed it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from .analysis import Point, bootstrap_ci, career_trend
from .quality import verdict_any

Gate = type(verdict_any)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _quarter_key(month: str) -> str | None:
    year, dash, mm = month.partition("-")
    if not dash or not year.isdigit() or not mm.isdigit():
        return None
    m = int(mm)
    if not 1 <= m <= 12:
        return None
    return f"{year}-Q{(m - 1) // 3 + 1}"


def _year_key(month: str) -> str | None:
    year, dash, mm = month.partition("-")
    if not dash or not year.isdigit() or not mm.isdigit():
        return None
    if not 1 <= int(mm) <= 12:
        return None
    return year


def _months_between(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)


#: Aggregates above the month level rarely hold more than a few dozen
#: contributing values, so 2000 resamples buys negligible precision over a
#: few hundred at this scale — and the site computes many thousands of these,
#: across every talent, metric, and granularity, so the constant matters.
_SITE_N_BOOT = 400


def _summarise(values: Sequence[float], *, bootstrap: bool = True) -> dict:
    """A month is the finest granularity: its own uncertainty is the
    corpus's noise floor (a same-month clip-pair statistic, computed
    separately — see reliability.py), not a bootstrap over its own 2-3
    clips, so ``bootstrap=False`` skips that computation entirely rather
    than reporting a near-meaningless interval. Quarter/year/talent-level
    aggregates DO bootstrap: those are built from several months' worth of
    values, which is exactly the case reference/statistics.md's CI replaces
    min-max for.
    """
    ci_low, ci_high = (
        bootstrap_ci(values, n_boot=_SITE_N_BOOT) if bootstrap else (math.nan, math.nan)
    )
    return {
        "median": statistics.median(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def monthly_series(
    records: list[dict], feature_key: str, *, gate=verdict_any
) -> list[dict]:
    """One point per calendar month with ``>=1`` clip passing ``gate``.

    A month whose every clip fails the gate is a gap, never a point at 0 —
    the gate is recomputed from stored features, not read back from a
    possibly-stale stored verdict.
    """
    buckets: dict[str, list[float]] = {}
    for record in records:
        features = record.get("features") or {}
        passed, _ = gate(features)
        if not passed:
            continue
        value = features.get(feature_key)
        month = record.get("month")
        if month and _finite(value):
            buckets.setdefault(month, []).append(float(value))
    return [
        {"period": month, **_summarise(values, bootstrap=False)}
        for month, values in sorted(buckets.items())
    ]


def _rollup(points: list[dict], key_fn) -> list[dict]:
    buckets: dict[str, list[float]] = {}
    for point in points:
        key = key_fn(point["period"])
        if key is None:
            continue
        buckets.setdefault(key, []).append(point["median"])
    return [
        {"period": key, **_summarise(values)} for key, values in sorted(buckets.items())
    ]


def quarterly_series(
    records: list[dict], feature_key: str, *, gate=verdict_any
) -> list[dict]:
    """Built from monthly medians (month-first), not raw clip values."""
    return _rollup(monthly_series(records, feature_key, gate=gate), _quarter_key)


def yearly_series(
    records: list[dict], feature_key: str, *, gate=verdict_any
) -> list[dict]:
    """Built from monthly medians (month-first), not raw clip values."""
    return _rollup(monthly_series(records, feature_key, gate=gate), _year_key)


def typical_and_trend(
    records: list[dict], feature_key: str, *, gate=verdict_any
) -> dict:
    """A talent-level summary for one metric: the corpus-comparable "typical"
    value (median of month medians — the same estimator the series itself
    uses), its bootstrap CI, and the career trend fit over the monthly
    series. Career time is measured from the talent's own first observed
    month, a proxy for debut (an early stream gone private starts the clock
    late)."""
    months = monthly_series(records, feature_key, gate=gate)
    if not months:
        return {"typical": math.nan, "ci_low": math.nan, "ci_high": math.nan, "trend": {}}
    values = [m["median"] for m in months]
    ci_low, ci_high = bootstrap_ci(values, n_boot=_SITE_N_BOOT)
    first = months[0]["period"]
    points = [
        Point(
            period=m["period"],
            career_months=float(_months_between(first, m["period"])),
            value=m["median"],
        )
        for m in months
    ]
    return {
        "typical": statistics.median(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "trend": career_trend(points),
    }
