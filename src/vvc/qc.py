"""The single QC rule (PLAN "QC (current)").

A measurement fails QC iff:

- ``median_f0`` is non-finite (no voiced frames), or
- ``voiced_fraction`` is non-finite or ``< 0.15`` (too little voiced
  content for the median/IQR to be a trustworthy statistic — a handful
  of voiced frames out of a 90 s clip can post a deceptively TIGHT IQR
  purely from having almost no data to spread across), or
- ``f0_iqr`` is non-finite or ``>= 200`` (BGM/melody/octave junk), or
- ``median_f0`` ``>= 600`` — the numpy ACF tracker cap ``_FMAX``
  (features.py). 600 is a junk indicator (octave errors), not "she
  cannot speak that high"; nothing below it is a pitch ceiling.

ONE implementation lives here and is used by both vvc.measure
(record ``qc`` block) and vvc.series (QC-series filtering), so
the persisted verdict and the plotted QC set can never drift apart.
"""

from __future__ import annotations

import math

IQR_QC_MAX = 200.0
F0_QC_MAX = 600.0
VOICED_FRACTION_QC_MIN = 0.15

REASON_MISSING = "f0_missing"
REASON_VOICED = "voiced_fraction"
REASON_IQR = "f0_iqr"
REASON_HIGH = "f0_high"


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def qc_verdict(features: dict) -> tuple[bool, str | None]:
    """Return ``(pass, reason)`` for a features dict.

    ``reason`` is ``None`` on a pass, else ``"f0_missing"``,
    ``"voiced_fraction"``, ``"f0_iqr"``, or ``"f0_high"``. Precedence
    when several conditions hold: missing median first (nothing else
    can be trusted without it), then too little voiced content, then
    IQR junk, then high median.
    """
    median = features.get("median_f0")
    if not _finite_number(median):
        return False, REASON_MISSING
    voiced = features.get("voiced_fraction")
    if not _finite_number(voiced) or float(voiced) < VOICED_FRACTION_QC_MIN:
        return False, REASON_VOICED
    iqr = features.get("f0_iqr")
    if not _finite_number(iqr) or float(iqr) >= IQR_QC_MAX:
        return False, REASON_IQR
    if float(median) >= F0_QC_MAX:
        return False, REASON_HIGH
    return True, None


def requalify(records: list[dict]) -> list[dict]:
    """Recompute the persisted ``qc`` block of every record from its
    already-stored ``features`` — no audio re-measurement. For applying
    a QC rule change (like the voiced_fraction floor) to an existing
    measurements file without re-running the pipeline. Every other key
    is passed through unchanged; a record with no ``features`` at all
    is passed through unchanged too (nothing to requalify)."""
    out: list[dict] = []
    for record in records:
        features = record.get("features")
        if features is None:
            out.append(record)
            continue
        qc_pass, qc_reason = qc_verdict(features)
        out.append({**record, "qc": {"pass": qc_pass, "reason": qc_reason}})
    return out
