"""The v2 quality gate.

One rule, one implementation, recomputed by every consumer rather than read
back from a stored verdict — otherwise moving a threshold updates some views
and not others, which is exactly what happened in v1.

What changed from v1, and why (docs/v1/LIMITATIONS.md has the full account):

- **The range gate has both bounds and neither duplicates a tracker
  parameter.** v1's upper bound equalled the tracker's own search ceiling, so
  the tracker could never report a value that would trip it; and there was no
  lower bound at all, so a 76 Hz median passed cleanly.
- **Spread is judged in semitones.** An absolute Hz threshold is proportionally
  far more permissive for a high-pitched voice, which is backwards for a corpus
  whose subject is high-pitched voices.
- **Contamination is checked directly.** This is the gate v1 lacked entirely,
  and it targets the failure that actually corrupts this corpus.

On that last point, because it drove the whole design: the implausibly low
readings are not octave errors. Three independent trackers — Praat
autocorrelation, Praat filtered autocorrelation, CREPE — agree on them, and the
clips carry a large voiced mass far below the talent's range that their
same-month counterparts do not have. The audio genuinely contains a lower
voice. Raising the pitch floor until the number looked plausible would have
manufactured a result from audio that is not the talent; the honest response is
to detect it and record it as a gap.

A clean pitch track still does not establish clean formants, brightness, or
voice quality. Passing this gate is not a blessing on every feature of a
record — the per-feature constraints in reference/measurement.md still apply.

See reference/qc.md.
"""

from __future__ import annotations

from dataclasses import dataclass

REASON_MISSING = "f0_missing"
REASON_VOICED = "voiced_fraction"
REASON_SPREAD = "f0_spread"
REASON_CONTAMINATED = "contaminated"
REASON_RANGE = "f0_out_of_range"


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds, in one recorded object rather than as module constants.

    Every bound here is calibrated against the measured distribution, not
    chosen for tidiness, and each sits in an observed gap rather than through
    a dense part of the data.
    """

    #: Below this, a clip has too little voiced content for its other
    #: statistics to mean anything.
    voiced_fraction_min: float = 0.15

    #: Within-clip pitch spread. One person talking does not range this widely;
    #: beyond it, something else is in the clip or the track is unstable.
    f0_spread_max_semitones: float = 14.0

    #: Share of voiced frames below the contamination band. Healthy clips sit
    #: near zero and contaminated ones sit far above; the threshold is placed
    #: in the gap between those populations, not at the edge of either.
    fraction_below_max: float = 0.25

    #: A generous backstop, deliberately wider than any voice in the corpus.
    #: The contamination gate does the real work; this only catches readings no
    #: human speaking voice would produce.
    f0_min_hz: float = 100.0
    f0_max_hz: float = 700.0


DEFAULT_CONFIG = QualityConfig()


def _number(value: object) -> float | None:
    """Reject bools explicitly: `True` is an int in Python and would sail
    through every numeric comparison as 1.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value:  # NaN
        return None
    return float(value)


def verdict(
    features: dict,
    *,
    config: QualityConfig = DEFAULT_CONFIG,
    require_contamination: bool = False,
) -> tuple[bool, str | None]:
    """``(passed, reason)``. The first failing gate decides the reason.

    ``require_contamination`` fails a record that carries no contamination
    measure at all. v2 records always carry one; a record that does not was
    produced by a different pipeline and must not be waved through a check it
    was never measured against.
    """
    median = _number(features.get("median_f0"))
    if median is None:
        return False, REASON_MISSING

    voiced = _number(features.get("voiced_fraction"))
    if voiced is None or voiced < config.voiced_fraction_min:
        return False, REASON_VOICED

    spread = _number(features.get("f0_iqr_semitones"))
    if spread is None or spread >= config.f0_spread_max_semitones:
        # A record carrying only v1's Hz spread lands here too, deliberately:
        # the semitone threshold cannot be applied to a number measured on a
        # different scale, and guessing a conversion would be inventing data.
        return False, REASON_SPREAD

    below = _number(features.get("fraction_below_160hz"))
    if below is None:
        if require_contamination:
            return False, REASON_CONTAMINATED
    elif below >= config.fraction_below_max:
        return False, REASON_CONTAMINATED

    if not (config.f0_min_hz <= median <= config.f0_max_hz):
        return False, REASON_RANGE

    return True, None


def verdict_any(features: dict) -> tuple[bool, str | None]:
    """Apply whichever gate matches the schema this record was measured under.

    A v2 corpus legitimately holds both: clips whose audio survived were
    re-measured and carry `f0_iqr_semitones`, while clips whose audio is gone
    keep their v1 features verbatim. Anything reading across such a file — the
    noise floor, the exports — needs one entry point that does not silently
    fail every legacy record.

    The dispatch is on the spread field because that is the one genuinely
    incompatible change: Hz and semitone spreads are different scales, and
    converting between them without the quartiles would be inventing data.
    """
    if "f0_iqr_semitones" in features:
        return verdict(features)
    from .qc import qc_verdict

    return qc_verdict(features)


def requalify(
    records: list[dict],
    *,
    config: QualityConfig = DEFAULT_CONFIG,
    require_contamination: bool = False,
) -> list[dict]:
    """Copies of ``records`` with their stored verdict recomputed.

    Features are never touched and the input is never mutated: this refreshes a
    convenience copy from stored numbers, it does not re-measure anything.
    """
    out: list[dict] = []
    for record in records:
        row = dict(record)
        features = row.get("features")
        if isinstance(features, dict):
            passed, reason = verdict(
                features, config=config, require_contamination=require_contamination
            )
            row["qc"] = {"pass": passed, "reason": reason}
        out.append(row)
    return out
