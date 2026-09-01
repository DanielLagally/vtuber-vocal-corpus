"""Product tests for the monthly F0/IQR series, the QC rule, and plots.

User-visible rules (synthetic data ONLY — fabricated measurement dicts;
never Cover/hololive audio, never downloads):

1. The monthly F0 series is a list of (month, median_f0) pairs sorted by
   month ("YYYY-MM" strings sort chronologically). A month whose
   median_f0 is non-finite is a GAP in the F0 series — excluded, never
   invented as 0 Hz.
2. That same non-finite-median month still appears in the IQR series
   when its f0_iqr is finite (the gap is per-series, not per-entry).
3. The all-months F0 series keeps QC-failing points — they are never
   silently deleted from the "all" view.
4. The QC series (``f0_series(entries, qc=True)``) drops exactly the
   points with f0_iqr >= 200 or a non-finite f0_iqr. IQR >= 200 is the
   ONLY QC rule — there is NO median-F0 ceiling anywhere: a 700 Hz
   median with a tight IQR survives QC.
5. write_plots(entries, out_dir) writes PNG plot files into out_dir.
   Existence only here — no image comparison.

Entry shape is the measurement record persisted by vanalysis.measure:
id / month / score / window / features{median_f0, f0_iqr,
voiced_fraction} / qc{pass, reason} / model.
"""

import math
from pathlib import Path

from vanalysis import series


# ---------------------------------------------------------------- helpers


def _entry(month: str, median_f0: float, f0_iqr: float, vid: str) -> dict:
    """A measurement record shaped exactly like measure persists it; the
    qc block is filled consistently with the only QC rule (IQR >= 200)."""
    qc_pass = math.isfinite(f0_iqr) and f0_iqr < 200
    return {
        "id": vid,
        "month": month,
        "score": 70.0,
        "window": {"start_s": 0.0, "end_s": 90.0},
        "features": {
            "median_f0": median_f0,
            "f0_iqr": f0_iqr,
            "voiced_fraction": 0.6,
        },
        "qc": {"pass": qc_pass, "reason": None if qc_pass else "f0_iqr"},
        "model": "bs_roformer_vocals_resurrection_unwa.ckpt",
    }


def _pairs(pairs_in) -> list[tuple]:
    """Normalize (month, value) pairs so the contract is about the pair
    contents and order, not the container class."""
    return [(month, value) for month, value in pairs_in]


# ------------------------------------------------------------------- tests


def test_f0_series_month_sorted_with_nan_gaps() -> None:
    """Rules 1: pairs are (month, median_f0) sorted by month across the
    year boundary, and the non-finite-median month is a gap — excluded,
    not zero-filled."""
    entries = [
        _entry("2025-01", 240.0, 35.0, "jan0000001"),
        _entry("2024-11", 220.0, 30.0, "nov0000001"),
        _entry("2024-12", math.nan, 25.0, "dec0000001"),
    ]
    assert _pairs(series.f0_series(entries)) == [
        ("2024-11", 220.0),
        ("2025-01", 240.0),
    ]


def test_iqr_series_keeps_nan_median_entry_with_finite_iqr() -> None:
    """Rule 2: the December entry has no median but a finite IQR — it must
    still appear in the IQR series (gap applies per-series)."""
    entries = [
        _entry("2025-01", 240.0, 35.0, "jan0000001"),
        _entry("2024-11", 220.0, 30.0, "nov0000001"),
        _entry("2024-12", math.nan, 25.0, "dec0000001"),
    ]
    assert _pairs(series.iqr_series(entries)) == [
        ("2024-11", 30.0),
        ("2024-12", 25.0),
        ("2025-01", 35.0),
    ]


def test_all_months_f0_series_keeps_qc_failing_points() -> None:
    """Rule 3: the all-months view keeps the IQR-junk point alongside the
    clean one (all vs QC must stay side-by-side)."""
    entries = [
        _entry("2025-02", 220.0, 40.0, "clean00001"),
        _entry("2025-03", 240.0, 260.0, "junkish001"),
    ]
    assert _pairs(series.f0_series(entries)) == [
        ("2025-02", 220.0),
        ("2025-03", 240.0),
    ]


def test_qc_series_drops_iqr_at_or_above_200_and_nonfinite() -> None:
    """Rule 4: the QC series drops IQR >= 200 (boundary included) and
    non-finite IQR; only the tight-IQR point survives."""
    entries = [
        _entry("2025-02", 220.0, 150.0, "keep000001"),
        _entry("2025-03", 230.0, 200.0, "edge199999"),
        _entry("2025-04", 240.0, 260.0, "junkish002"),
        _entry("2025-05", 250.0, math.nan, "noiqr00001"),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == [("2025-02", 220.0)]


def test_qc_series_has_no_f0_ceiling() -> None:
    """Rule 4: IQR >= 200 is the ONLY QC rule — a 700 Hz median (above the
    old 500 Hz pitch ceiling) with a tight IQR must survive QC."""
    entries = [_entry("2025-06", 700.0, 45.0, "highbutok1")]
    assert _pairs(series.f0_series(entries, qc=True)) == [("2025-06", 700.0)]


def test_write_plots_creates_png_files(tmp_path: Path) -> None:
    """Rule 5: write_plots over 3 synthetic entries (one clean, one IQR
    junk, one missing median) produces PNG files in out_dir — gaps and QC
    points must not crash plotting. Existence only."""
    entries = [
        _entry("2025-01", 220.0, 35.0, "clean00003"),
        _entry("2025-02", 240.0, 260.0, "junkish003"),
        _entry("2025-03", math.nan, 30.0, "gap0000001"),
    ]
    series.write_plots(entries, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, f"write_plots must produce PNG files in {tmp_path}"
