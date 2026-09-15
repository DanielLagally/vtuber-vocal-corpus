"""The measurement's own noise floor, and non-destructive review flags.

Two clips of the same talent in the same month share the speaker, the period
and the pipeline, so what separates them is what the instrument contributes
plus ordinary day-to-day variation. That number is the scale every other number
here has to be read against: a trend is only a finding if it is large relative
to it, and a difference between two talents is only a difference if it exceeds
it. Publishing a metric without it is publishing a number without its units.

The same pairing catches what QC structurally cannot. The gate checks one clip
against a plausibility band, so a clip read an octave low still passes — but it
cannot agree with its own same-month counterpart, and that disagreement is
visible without any audio. Flags are therefore advisory: they mark a clip for a
human, they never drop it and they are not a second gate.

Everything here runs on stored measurements alone — no audio, no re-measurement.

See reference/statistics.md and reference/qc.md.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from .quality import verdict_any

#: A gate maps stored features to ``(passed, reason)``. Injectable so a caller
#: can pin one schema's rule explicitly; the default dispatches per record.
Gate = Callable[[dict], tuple[bool, str | None]] | None

#: Metrics a noise floor is reported for by default. Pitch first because it is
#: the only one robust enough to the processing chain to carry a conclusion;
#: the rest are reported so their noise is visible next to their claims.
DEFAULT_FEATURE_KEYS = (
    "median_f0",
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

#: Half an octave. Two clips of one person weeks apart do not differ by this
#: much for any reason that is about the person; at or beyond it, one of the
#: two readings is wrong even though both may pass every gate.
DEFAULT_SEMITONE_THRESHOLD = 6.0


@dataclass(frozen=True)
class Pair:
    """Two QC-passing clips of one talent from one month."""

    month: str
    ids: tuple[str, str]
    values: tuple[float, float]
    feature_key: str = "median_f0"

    @property
    def abs_diff(self) -> float:
        return abs(self.values[0] - self.values[1])

    @property
    def abs_semitones(self) -> float:
        """Scale-free distance. Undefined when either value is non-positive —
        a ratio needs two magnitudes, and a zero or negative reading is not
        one."""
        lo, hi = self.values
        if lo <= 0.0 or hi <= 0.0:
            return math.nan
        return abs(12.0 * math.log2(hi / lo))


@dataclass(frozen=True)
class Flag:
    """A same-month pair that disagrees by more than a musically plausible
    interval. Advisory only."""

    month: str
    ids: tuple[str, str]
    values: tuple[float, float]
    semitones: float
    feature_key: str = "median_f0"
    reason: str = field(default="same_month_discordance")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _passing(
    records: list[dict], feature_key: str, gate: Gate = None
) -> list[tuple[str, str, float]]:
    """(month, id, value) for clips that pass the CURRENT gate.

    The stored ``qc`` block is deliberately ignored: it is a convenience copy
    that goes stale the moment a threshold moves, and a noise floor computed
    over clips the live rule would reject measures the wrong thing.

    The default gate dispatches on the record's own schema, because a v2
    corpus holds re-measured and legacy records side by side.
    """
    gate = gate or verdict_any
    out: list[tuple[str, str, float]] = []
    for record in records:
        features = record.get("features") or {}
        passed, _ = gate(features)
        if not passed:
            continue
        value = features.get(feature_key)
        if not _finite(value):
            continue
        month = record.get("month")
        if not month:
            continue
        out.append((month, str(record.get("id", "")), float(value)))
    return out


def within_month_pairs(
    records: list[dict], *, feature_key: str = "median_f0", gate: Gate = None
) -> list[Pair]:
    """Every unordered pair of QC-passing clips sharing a month.

    All pairs, not just consecutive ones: each disagreement is independent
    evidence about the instrument, and dropping some of them would discard it.
    Clips in different months are never paired — a difference across months
    could be real change, which is the thing we are trying to measure, not the
    noise we are trying to characterise.
    """
    by_month: dict[str, list[tuple[str, float]]] = {}
    for month, video_id, value in _passing(records, feature_key, gate):
        by_month.setdefault(month, []).append((video_id, value))

    pairs: list[Pair] = []
    for month in sorted(by_month):
        for (id_a, val_a), (id_b, val_b) in itertools.combinations(by_month[month], 2):
            pairs.append(
                Pair(
                    month=month,
                    ids=(id_a, id_b),
                    values=(val_a, val_b),
                    feature_key=feature_key,
                )
            )
    return pairs


def _summarise(pairs: list[Pair]) -> dict:
    diffs = [pair.abs_diff for pair in pairs]
    semitones = [
        pair.abs_semitones for pair in pairs if not math.isnan(pair.abs_semitones)
    ]
    return {
        "n_pairs": len(pairs),
        "median_abs_diff": statistics.median(diffs) if diffs else None,
        "p90_abs_diff": _percentile(diffs, 90.0) if diffs else None,
        "median_abs_semitones": statistics.median(semitones) if semitones else None,
    }


def noise_floor(
    records: list[dict],
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    gate: Gate = None,
) -> dict[str, dict]:
    """Per-metric same-month disagreement for ONE talent: their noise floor.

    Reported in the metric's own units and, where the metric is a frequency,
    in semitones as well — an absolute Hz disagreement means something
    different for a high-pitched voice than a low-pitched one, so only the
    semitone form compares across talents.

    A metric with no pairs reports ``None``, never ``0.0``. Zero would read as
    a perfect instrument; the truth is that we do not know.
    """
    return {
        key: _summarise(within_month_pairs(records, feature_key=key, gate=gate))
        for key in feature_keys
    }


def corpus_noise_floor(
    records_by_talent: dict[str, list[dict]],
    *,
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS,
    gate: Gate = None,
) -> dict[str, dict]:
    """The corpus-wide noise floor, pooled from per-talent pairs.

    Pairing happens strictly inside one talent and then the pairs are pooled.
    Concatenating every talent's records first would pair two different people
    who happened to stream in the same month, and the difference between two
    people is the signal this corpus exists to measure — not its noise.
    """
    pooled: dict[str, list[Pair]] = {key: [] for key in feature_keys}
    for records in records_by_talent.values():
        for key in feature_keys:
            pooled[key].extend(within_month_pairs(records, feature_key=key, gate=gate))
    return {key: _summarise(pairs) for key, pairs in pooled.items()}


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Deliberately not interpolated: with the handful
    of pairs a single talent-month yields, an interpolated value invents a
    disagreement nobody observed."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def discordance_flags(
    records: list[dict],
    *,
    feature_key: str = "median_f0",
    semitone_threshold: float = DEFAULT_SEMITONE_THRESHOLD,
    gate: Gate = None,
) -> list[Flag]:
    """Same-month pairs disagreeing by more than ``semitone_threshold``.

    Non-destructive by construction: this reads records and returns findings.
    It does not mutate, re-verdict, or remove anything. An octave error is the
    archetypal case — both clips pass every gate, and only their mutual
    disagreement reveals that one of them is wrong.
    """
    flags: list[Flag] = []
    for pair in within_month_pairs(records, feature_key=feature_key, gate=gate):
        semitones = pair.abs_semitones
        if math.isnan(semitones) or semitones < semitone_threshold:
            continue
        flags.append(
            Flag(
                month=pair.month,
                ids=pair.ids,
                values=pair.values,
                semitones=semitones,
                feature_key=feature_key,
            )
        )
    return flags
