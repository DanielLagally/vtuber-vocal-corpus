"""The single QC rule (PLAN "QC (current)").

A measurement fails QC iff:

- ``f0_iqr`` is non-finite or ``>= 200`` (BGM/melody/octave junk), or
- ``median_f0`` is non-finite (no voiced frames), or
- ``median_f0`` ``>= 600`` — the numpy ACF tracker cap ``_FMAX``
  (features.py). 600 is a junk indicator (octave errors), not "she
  cannot speak that high"; nothing below it is a pitch ceiling.

ONE implementation lives here and is used by both vanalysis.measure
(record ``qc`` block) and vanalysis.series (QC-series filtering), so
the persisted verdict and the plotted QC set can never drift apart.
"""

from __future__ import annotations

import math

IQR_QC_MAX = 200.0
F0_QC_MAX = 600.0

REASON_MISSING = "f0_missing"
REASON_IQR = "f0_iqr"
REASON_HIGH = "f0_high"


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def qc_verdict(features: dict) -> tuple[bool, str | None]:
    """Return ``(pass, reason)`` for a features dict.

    ``reason`` is ``None`` on a pass, else ``"f0_missing"``,
    ``"f0_iqr"``, or ``"f0_high"``. Precedence when several conditions
    hold: missing median first, then IQR junk, then high median.
    """
    median = features.get("median_f0")
    if not _finite_number(median):
        return False, REASON_MISSING
    iqr = features.get("f0_iqr")
    if not _finite_number(iqr) or float(iqr) >= IQR_QC_MAX:
        return False, REASON_IQR
    if float(median) >= F0_QC_MAX:
        return False, REASON_HIGH
    return True, None
