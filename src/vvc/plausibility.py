"""Is this clip plausibly the talent, around the time it was recorded?

The per-clip gate in `quality.py` cannot answer this. A clip carrying a guest's
voice is perfectly plausible *on its own* — normal spread, normal voicing, a
pitch a human could produce — and only looks wrong beside the rest of that
talent's career. So this is a corpus-level pass, run once a talent's series
exists.

Three design choices, each forced by measurement against human labels
(`data/outlier-review/labels.json`), and each contradicting a plausible guess:

1. **The threshold is in semitones, not standard deviations.** Normalising by
   each talent's own spread destroys the very separation it is meant to
   sharpen: a wide-range talent's contamination gets divided into invisibility,
   while a narrow-range talent's legitimate extreme gets divided into looking
   like contamination. On the labelled set the two populations overlap
   completely in spread units and separate cleanly in absolute ones. The reason
   is physical — a male voice sits roughly an octave below these talents no
   matter how variable the talent is — so the natural scale is absolute.

2. **The baseline follows the talent's own trend** rather than being a single
   career-wide number. Some talents genuinely drift most of an octave across a
   career. A fixed centre puts both ends of that drift near the band edge, so
   the gate trims the largest real trends in the corpus — which would be the
   worst failure available, since those trends are the finding.

3. **The band is asymmetric: tight below, loose above.** Every contamination
   observed so far is a lower voice, and below the baseline there is a clean
   gap between the worst legitimate clip and the mildest contaminated one.
   Above the baseline there is no gap at all — a deliberate character
   performance lands *closer* to the baseline than one talent's entirely
   ordinary wide range — so a tight upper bound buys nothing and destroys real
   data. The upper bound exists only to catch readings no voice would produce.

What this deliberately does not catch: a character performance (indistinguishable
from wide range by pitch alone), and an instrument tracked as a voice (which can
sit dead centre of normal). Both are documented limitations, not oversights.

See reference/qc.md.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from . import analysis
from .quality import verdict_any

REASON = "implausible_for_talent"

#: Semitones below the talent's own baseline at that date. Every labelled
#: contamination sits beyond this; every labelled legitimate clip sits well
#: inside it, with a clear gap between the two populations.
DEFAULT_LOW_SEMITONES = 6.0

#: Semitones above. Deliberately loose — see the module docstring. This is a
#: backstop for physically implausible readings, not a performance filter.
DEFAULT_HIGH_SEMITONES = 11.0

#: A baseline fitted to a handful of months is not a baseline. Below this, no
#: band is applied at all: rejecting against a guess would be inventing a
#: standard rather than measuring against one.
MIN_MONTHS = 10


@dataclass(frozen=True)
class Flag:
    id: str
    month: str
    value_hz: float
    baseline_hz: float
    semitones: float


def _months_between(a: str, b: str) -> int:
    return (int(b[:4]) - int(a[:4])) * 12 + (int(b[5:7]) - int(a[5:7]))


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _monthly_means(records: list[dict], feature_key: str) -> dict[str, float]:
    """Month -> mean of that month's QC-passing clips.

    QC-failing clips are excluded so junk cannot define what normal looks like,
    and months are averaged first so a densely sampled month cannot drag the
    baseline toward itself.
    """
    buckets: dict[str, list[float]] = {}
    for record in records:
        features = record.get("features") or {}
        if not verdict_any(features)[0]:
            continue
        value = features.get(feature_key)
        month = record.get("month")
        if month and _finite(value) and float(value) > 0:
            buckets.setdefault(month, []).append(float(value))
    return {month: statistics.mean(values) for month, values in buckets.items()}


def baseline(records: list[dict], *, feature_key: str = "median_f0"):
    """A function from month to the talent's expected value in Hz, or None.

    The trend is Theil-Sen over monthly means in semitone space, so a few
    contaminated months cannot tilt it and the fit is scale-free.
    """
    months = _monthly_means(records, feature_key)
    if len(months) < MIN_MONTHS:
        return None
    order = sorted(months)
    centre = statistics.median([months[m] for m in order])
    points = [
        analysis.Point(
            period=month,
            career_months=float(_months_between(order[0], month)),
            value=12.0 * math.log2(months[month] / centre),
        )
        for month in order
    ]
    trend = analysis.career_trend(points)
    if not _finite(trend.get("slope_per_month")) or not _finite(trend.get("intercept")):
        return None
    first = order[0]

    def at(month: str) -> float:
        offset = trend["intercept"] + trend["slope_per_month"] * float(
            _months_between(first, month)
        )
        return centre * (2.0 ** (offset / 12.0))

    return at


def implausible(
    records: list[dict],
    *,
    feature_key: str = "median_f0",
    low_semitones: float = DEFAULT_LOW_SEMITONES,
    high_semitones: float = DEFAULT_HIGH_SEMITONES,
) -> list[Flag]:
    """Clips too far from the talent's own baseline at their date."""
    at = baseline(records, feature_key=feature_key)
    if at is None:
        return []

    flags: list[Flag] = []
    for record in records:
        value = (record.get("features") or {}).get(feature_key)
        month = record.get("month")
        if not month or not _finite(value) or float(value) <= 0:
            continue
        expected = at(month)
        if expected <= 0:
            continue
        distance = 12.0 * math.log2(float(value) / expected)
        if distance < -low_semitones or distance > high_semitones:
            flags.append(
                Flag(
                    id=str(record.get("id", "")),
                    month=month,
                    value_hz=float(value),
                    baseline_hz=float(expected),
                    semitones=float(distance),
                )
            )
    return flags


def apply(
    records: list[dict],
    *,
    feature_key: str = "median_f0",
    low_semitones: float = DEFAULT_LOW_SEMITONES,
    high_semitones: float = DEFAULT_HIGH_SEMITONES,
) -> list[dict]:
    """Copies of ``records`` with implausible clips failed and the distance
    recorded, so the decision can be audited rather than merely trusted."""
    flags = {
        flag.id: flag
        for flag in implausible(
            records,
            feature_key=feature_key,
            low_semitones=low_semitones,
            high_semitones=high_semitones,
        )
    }
    out: list[dict] = []
    for record in records:
        row = dict(record)
        flag = flags.get(str(record.get("id", "")))
        if flag is not None:
            row["qc"] = {"pass": False, "reason": REASON}
            row["plausibility_semitones"] = flag.semitones
            row["plausibility_baseline_hz"] = flag.baseline_hz
        out.append(row)
    return out
