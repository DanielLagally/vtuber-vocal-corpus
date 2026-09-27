"""Which talents sound most alike, from measurements alone.

Two routes, deliberately kept apart so each can check the other:

**Measured metrics.** Talents are compared on their typical values. Every
difference is measured against how much that metric varies between one
talent's *own* clips — the pooled within-talent covariance — i.e. a
Mahalanobis distance. That one choice handles two problems at once: a metric
that jumps around from clip to clip counts for little and a stable one for a
lot, and several metrics reading the same underlying quality (tilt, alpha
ratio, Hammarberg index) cannot vote several times, because their correlation
is in the covariance. It is interpretable: the metrics on which a pair sits
unusually close — closer than talents typically differ — are *why* it matched.

**Embeddings** (``embed.py``): cosine similarity of talent centroids. Stronger,
but it cannot say why.

Both are reported as an absolute match — 0% a typical unrelated pair, 100% as
alike as a talent is to themselves — and as "closer than X% of all talent
pairs", so the two read the same way and neither needs a threshold picked by
hand. Nobody decides which
pairs "should" match; ``voice_validate`` checks the routes against each other
and against speaker-identification accuracy instead.

A one-off data point (``segment.py``) appears in neighbour lists, tagged, but
never shapes the pair distribution the scores are read against.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: Voice metrics the measured route compares on — the voice families of
#: site_data_v2, never the context/quality family, which describes the
#: recording rather than the speaker.
VOICE_METRICS: tuple[str, ...] = (
    "median_f0",
    "f0_iqr_semitones",
    "dynamism_semitones",
    "h1h2_db",
    "cpp_db",
    "harmonic_tilt_db_per_octave",
    "alpha_ratio_db",
    "hammarberg_db",
    "hnr_db",
    "jitter_local",
    "shimmer_local",
    "brightness_hz",
    "formant_dispersion_hz",
    "speaking_rate_syl_per_s",
)

#: Frequencies are compared as ratios (semitones), as a listener hears them:
#: 20 Hz between two low voices is a bigger step than between two high ones.
_LOG_SCALED = frozenset({"median_f0", "brightness_hz", "formant_dispersion_hz"})

#: Shrinkage of the within-talent covariance toward its diagonal. Keeps the
#: inverse stable when two metrics are nearly collinear, at the cost of letting
#: a near-duplicate count very slightly more than once.
SHRINKAGE = 0.1


def voice_value(metric: str, value: object) -> float:
    """A raw feature value on the scale the distance uses; ``nan`` if absent."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
        return math.nan
    value = float(value)
    if metric in _LOG_SCALED:
        return 12.0 * math.log2(value / 100.0) if value > 0 else math.nan
    return value


@dataclass(frozen=True)
class MetricSpace:
    metrics: tuple[str, ...]
    covariance: np.ndarray  # shrunk pooled within-talent covariance

    def _vector(self, features: dict) -> np.ndarray:
        return np.array([voice_value(m, features.get(m)) for m in self.metrics])

    def distance(self, a: dict, b: dict) -> float:
        """Mahalanobis distance over the metrics both sides have."""
        delta = self._vector(a) - self._vector(b)
        shared = np.isfinite(delta)
        if not shared.any():
            return math.nan
        d = delta[shared]
        cov = self.covariance[np.ix_(shared, shared)]
        return float(math.sqrt(max(0.0, float(d @ np.linalg.solve(cov, d)))))

    def closest_metrics(
        self,
        a: dict,
        b: dict,
        k: int = 3,
        spread: np.ndarray | None = None,
        among: Iterable[str] | None = None,
    ) -> list[str]:
        """The metrics on which the two sit closest — the reason a pair matched.

        With ``spread`` (``between_spread``), closeness is judged against how
        much talents differ on each metric, which is what makes a match
        notable. Without it, against within-talent noise; that flatters a
        metric that wanders a lot, since every pair then looks close on it.
        ``among`` limits the explanation to metrics a reader can relate to."""
        allowed = set(among) if among is not None else set(self.metrics)
        delta = self._vector(a) - self._vector(b)
        scale = spread if spread is not None else np.sqrt(np.diag(self.covariance))
        gaps = [
            (abs(delta[i]) / scale[i], i)
            for i in range(len(self.metrics))
            if math.isfinite(delta[i]) and scale[i] > 0 and self.metrics[i] in allowed
        ]
        return [self.metrics[i] for _, i in sorted(gaps)[:k]]


def between_spread(space: MetricSpace, typicals: Iterable[dict]) -> np.ndarray:
    """Per metric, the standard deviation of talents' typical values (on the
    distance scale): how much talents differ from each other."""
    rows = np.array([space._vector(t) for t in typicals])
    if rows.size == 0:
        return np.full(len(space.metrics), np.nan)
    with np.errstate(all="ignore"):
        return np.nanstd(rows, axis=0)


def fit_metric_space(
    clips: Iterable[tuple[str, dict]], *, metrics: Sequence[str] = VOICE_METRICS
) -> MetricSpace:
    """Pooled within-talent covariance from ``(talent, features)`` clips.

    Each clip is centred on its own talent's mean, so what remains is how one
    person varies from clip to clip — never how people differ, which is the
    signal. Only clips with every metric present contribute, so the covariance
    is one consistent matrix rather than a patchwork of pairwise estimates.
    """
    metrics = tuple(metrics)
    by_talent: dict[str, list[np.ndarray]] = {}
    for talent, features in clips:
        row = np.array([voice_value(m, features.get(m)) for m in metrics])
        if np.all(np.isfinite(row)):
            by_talent.setdefault(talent, []).append(row)
    residuals = [
        np.vstack(rows) - np.mean(rows, axis=0) for rows in by_talent.values() if len(rows) > 1
    ]
    if not residuals:
        raise ValueError("no talent has two complete clips; cannot estimate within-talent spread")
    stacked = np.vstack(residuals)
    dof = stacked.shape[0] - len(residuals)
    cov = stacked.T @ stacked / max(dof, 1)
    shrunk = (1 - SHRINKAGE) * cov + SHRINKAGE * np.diag(np.diag(cov))
    # A metric with no within-talent spread at all would make the matrix
    # singular; give it a tiny floor rather than an infinite weight.
    floor = 1e-9 * max(1.0, float(np.max(np.diag(shrunk))))
    shrunk = shrunk + floor * np.eye(len(metrics))
    return MetricSpace(metrics, shrunk)


def match_percent(value: float, *, zero: float, full: float) -> float:
    """An absolute match on a 0-100 scale: ``zero`` scores 0 (a typical
    unrelated pair), ``full`` scores 100 (as alike as a talent is to
    themselves). Works for similarities and distances alike, since only the
    direction from ``zero`` to ``full`` matters. Clipped to the scale;
    ``nan`` when the scale is undefined."""
    if not all(math.isfinite(v) for v in (value, zero, full)) or full == zero:
        return math.nan
    return float(min(100.0, max(0.0, 100.0 * (value - zero) / (full - zero))))


def neighbours(
    names: Sequence[str],
    score: Callable[[str, str], float],
    *,
    k: int = 8,
    one_off: Iterable[str] = (),
    higher_is_closer: bool = False,
    scale: tuple[float, float] | None = None,
) -> dict[str, list[dict]]:
    """Each talent's ``k`` closest others, closest first.

    ``closer_than_pct`` is the share of ordinary talent pairs further apart
    than this one. One-offs appear in other talents' lists, tagged
    ``one_off``, but are left out of that reference distribution, so they
    never move anyone else's score. With ``scale`` (zero, full), each row also
    carries an absolute ``match_pct`` (see ``match_percent``). An undefined
    score leaves the pair out.
    """
    one_off = set(one_off)
    names = list(names)
    pair_score: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = score(a, b)
            if value == value:
                pair_score[(a, b)] = float(value)

    def key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if names.index(a) < names.index(b) else (b, a)

    closeness = (lambda v: v) if higher_is_closer else (lambda v: -v)
    reference = sorted(
        closeness(v) for (a, b), v in pair_score.items() if a not in one_off and b not in one_off
    )

    def closer_than_pct(value: float) -> float:
        if not reference:
            return math.nan
        further = int(np.searchsorted(reference, closeness(value), side="left"))
        return 100.0 * further / len(reference)

    out: dict[str, list[dict]] = {}
    for name in names:
        rows = []
        for other in names:
            if other == name:
                continue
            value = pair_score.get(key(name, other))
            if value is None:
                continue
            row = {
                "name": other,
                "value": value,
                "closer_than_pct": closer_than_pct(value),
                "one_off": other in one_off,
            }
            if scale is not None:
                row["match_pct"] = match_percent(value, zero=scale[0], full=scale[1])
            rows.append(row)
        rows.sort(key=lambda r: (-closeness(r["value"]), r["name"]))
        out[name] = rows[:k]
    return out
