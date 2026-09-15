"""Trends, uncertainty, and the part of the era confound that can be measured.

The corpus's headline pattern is a change in pitch over career time. Whether
that is about voices or about recording conditions is the question everything
else depends on, and it has a sharper answer than "unknown".

**The obvious test does not work.** Fitting career time and calendar time
together, and seeing which carries the signal, is not identified: career months
= calendar time - debut date, so once a talent intercept is in the model the two
slopes are exactly collinear. This is the age-period-cohort problem, and the
design matrix is provably rank-deficient — any coefficients such a fit returned
would be an artifact of the solver's pseudo-inverse, not a finding. Reporting
them would be worse than reporting nothing.

**What is identifiable is the non-linear part.** Remove each talent's own
linear career trend first; a calendar effect shared by talents who are at
different career stages then appears in the residuals. A shared *linear* drift
is absorbed by that first step and stays invisible, so this test can find an
era shock but cannot rule out a smooth era drift. That limit is real and is
stated wherever these numbers are shown.

**The third leg is external.** The encode-sensitivity measurement in
`bakeoff.era_sensitivity` estimates one calendar mechanism directly, from
outside the panel, which is the only way to put a number on the component the
residual test cannot see.

See reference/statistics.md.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: A talent needs enough points for a detrend to mean anything. Below this, the
#: "trend" removed is mostly noise and the residuals are meaningless.
MIN_POINTS_FOR_DETREND = 6


@dataclass(frozen=True)
class Point:
    """One observation: when it happened, how far into the career, what value."""

    period: str  # "YYYY-MM"
    career_months: float
    value: float


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Median of pairwise slopes, and the matching intercept.

    Preferred over least squares throughout because a contaminated clip is a
    large, one-sided outlier, and least squares is maximally sensitive to
    exactly that. NaN for fewer than two usable points — a slope through one
    observation is not a slope.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if _finite(x) and _finite(y)]
    if len(pairs) < 2:
        return math.nan, math.nan
    slopes = [
        (pairs[j][1] - pairs[i][1]) / (pairs[j][0] - pairs[i][0])
        for i in range(len(pairs))
        for j in range(i + 1, len(pairs))
        if pairs[j][0] != pairs[i][0]
    ]
    if not slopes:
        return math.nan, math.nan
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in pairs])
    return float(slope), float(intercept)


def _period_key(period: str, granularity: str) -> str:
    return period[:4] if granularity == "year" else period


def common_period_residuals(
    series_by_talent: dict[str, list[Point]],
    *,
    period: str = "year",
    min_points: int = MIN_POINTS_FOR_DETREND,
) -> dict[str, dict]:
    """Calendar-time structure shared across talents, after removing each
    talent's own career trend.

    Talents are weighted equally: each talent's residuals are averaged within a
    period first, and only then averaged across talents. Pooling residuals
    directly would let a densely sampled talent *become* the common effect.

    A period where several talents at different career stages all deviate the
    same way is the signature of a recording-era change. A shared linear drift
    is absorbed by the detrend and will not appear here.
    """
    per_talent: dict[str, dict[str, list[float]]] = {}
    for talent, points in series_by_talent.items():
        usable = [p for p in points if _finite(p.value) and _finite(p.career_months)]
        if len(usable) < min_points:
            continue
        slope, intercept = theil_sen(
            [p.career_months for p in usable], [p.value for p in usable]
        )
        if not _finite(slope):
            continue
        buckets: dict[str, list[float]] = {}
        for point in usable:
            residual = point.value - (intercept + slope * point.career_months)
            buckets.setdefault(_period_key(point.period, period), []).append(residual)
        per_talent[talent] = buckets

    periods: set[str] = set()
    for buckets in per_talent.values():
        periods.update(buckets)

    out: dict[str, dict] = {}
    for key in sorted(periods):
        talent_means = [
            statistics.mean(buckets[key])
            for buckets in per_talent.values()
            if buckets.get(key)
        ]
        if not talent_means:
            continue
        out[key] = {
            "mean_residual": float(statistics.mean(talent_means)),
            "median_residual": float(statistics.median(talent_means)),
            "n_talents": len(talent_means),
            "sd_across_talents": (
                float(statistics.stdev(talent_means)) if len(talent_means) > 1 else None
            ),
        }
    return out


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.median,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic``.

    This replaces min-max as the plotted band. Min-max *widens* with sample
    size, so a well-sampled period looks less certain than a poorly sampled
    one — precisely backwards. A single observation yields NaN rather than a
    zero-width interval, because zero width would claim certainty from one
    measurement.
    """
    usable = [float(v) for v in values if _finite(v)]
    if len(usable) < 2:
        return math.nan, math.nan
    rng = random.Random(seed)
    size = len(usable)
    estimates = [
        statistic([usable[rng.randrange(size)] for _ in range(size)])
        for _ in range(n_boot)
    ]
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low = estimates[max(0, int(tail * n_boot) - 1)]
    high = estimates[min(n_boot - 1, int((1.0 - tail) * n_boot))]
    return float(low), float(high)


def career_trend(points: Sequence[Point]) -> dict:
    """A talent's robust trend over career time, with its fit quality.

    A slope without a goodness-of-fit figure invites a reader to see a trend
    the scatter does not support, so both are always returned together.
    """
    usable = [p for p in points if _finite(p.value) and _finite(p.career_months)]
    if len(usable) < 2:
        return {"slope_per_month": math.nan, "r2": math.nan, "n": len(usable)}
    xs = [p.career_months for p in usable]
    ys = [p.value for p in usable]
    slope, intercept = theil_sen(xs, ys)
    mean_y = statistics.mean(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else math.nan
    return {
        "slope_per_month": slope,
        "slope_per_year": slope * 12.0 if _finite(slope) else math.nan,
        "intercept": intercept,
        "r2": r2,
        "n": len(usable),
        "span_months": max(xs) - min(xs),
    }
