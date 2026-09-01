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
   points failing the shared QC rule (PLAN): ``f0_iqr >= 200`` OR
   median_f0 non-finite OR ``median_f0 >= 600`` (the numpy ACF tracker
   cap — a junk/octave-error indicator, NOT "she cannot speak that
   high"). A 700 Hz tight-IQR median is tracker junk and is dropped
   from the QC view; the all-months view still keeps it visible.
5. write_plots(entries, out_dir) writes PNG plot files into out_dir.
   Existence only here — no image comparison.
6. The quarterly series (``f0_quarterly``) aggregates clip medians per
   calendar quarter ("YYYY-Qn"): the MEAN of that quarter's clip
   median_f0 values, plus min and max of the same values, and n,
   sorted by quarter. A clip with a non-finite median is excluded
   ENTIRELY from its quarterly point (it cannot plot — it contributes
   to nothing, not even n); a quarter left with no plottable clips is
   simply absent (a gap, never an invented point). A n=1 quarter is a
   point with min == max == the single value (the plot decides band vs
   bare point; the aggregation stays pure).
7. The qc=True variant filters via the shared QC rule BEFORE
   aggregating (nan-median, IQR >= 200, and median >= 600 all drop
   first); the all-clip variant includes QC-failing clips but still
   excludes non-finite medians entirely.
8. write_quarterly_plots(entries, out_dir) writes exactly two PNGs:
   f0_quarterly_all.png and f0_quarterly_qc.png. The min–max band is
   drawn only where n >= 2 (n=1 renders a bare point — anecdotal,
   PLAN L36); existence only here — no image comparison.

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
    qc block is filled consistently with the shared QC rule (fail iff
    f0_iqr >= 200, median non-finite, or median >= 600)."""
    if not math.isfinite(median_f0):
        qc_pass, qc_reason = False, "f0_missing"
    elif not math.isfinite(f0_iqr) or f0_iqr >= 200.0:
        qc_pass, qc_reason = False, "f0_iqr"
    elif median_f0 >= 600.0:
        qc_pass, qc_reason = False, "f0_high"
    else:
        qc_pass, qc_reason = True, None
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
        "qc": {"pass": qc_pass, "reason": qc_reason},
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


def test_qc_series_drops_median_f0_at_or_above_600() -> None:
    """Rule 4 (PLAN, flipped from the old no-ceiling rule): median F0 >=
    600 (the numpy ACF tracker cap _FMAX) is an octave-junk indicator —
    a 700 Hz median with a tight IQR FAILS QC and is dropped from the QC
    series, while the all-months series still keeps it visible so the
    cleaning stays auditable."""
    entries = [_entry("2025-06", 700.0, 45.0, "highbutok1")]
    assert _pairs(series.f0_series(entries, qc=True)) == [], (
        "a >=600 Hz median is tracker junk and must be dropped from QC"
    )
    assert _pairs(series.f0_series(entries)) == [("2025-06", 700.0)], (
        "the all-months view still shows the QC-failing point"
    )


def test_qc_series_applies_full_rule_keeps_599_drops_nan_and_high() -> None:
    """Rule 4 (full rule in one view): the QC series keeps a 599 Hz
    tight-IQR point, drops a NaN-median point and a 615 Hz point; the
    all-months series keeps both finite-median points (fails are gaps,
    never deletions, and NaN medians are gaps in every F0 series)."""
    entries = [
        _entry("2025-01", 599.0, 45.0, "keep599001"),
        _entry("2025-02", math.nan, 30.0, "nanmed0001"),
        _entry("2025-03", 615.0, 45.0, "high615001"),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == [("2025-01", 599.0)]
    assert _pairs(series.f0_series(entries)) == [
        ("2025-01", 599.0),
        ("2025-03", 615.0),
    ]


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


# ------------------------------------------------------------- quarterly


def test_f0_quarterly_mean_min_max_n_sorted_nan_excluded_entirely() -> None:
    """Rule 6: per calendar quarter the mean/min/max/n of the clip
    medians, sorted by quarter; the nan-median June clip is excluded
    ENTIRELY (n stays 2, mean/min/max untouched — it cannot plot, it
    must not be averaged in or counted)."""
    entries = [
        _entry("2020-04", 220.0, 35.0, "apr000001"),
        _entry("2020-05", 240.0, 45.0, "may000001"),
        _entry("2019-11", 200.0, 35.0, "nov000001"),
        _entry("2020-06", math.nan, 30.0, "junnan0001"),
    ]
    assert series.f0_quarterly(entries) == [
        {"quarter": "2019-Q4", "mean": 200.0, "min": 200.0, "max": 200.0, "n": 1},
        {"quarter": "2020-Q2", "mean": 230.0, "min": 220.0, "max": 240.0, "n": 2},
    ]


def test_f0_quarterly_single_clip_point_min_equals_max() -> None:
    """Rule 6 (pinned choice): a n=1 quarter carries min == max == the
    single value — the aggregation stays pure; whether it renders as a
    band or a bare point is the plot's decision."""
    entries = [_entry("2020-06", 250.0, 35.0, "jun000001")]
    assert series.f0_quarterly(entries) == [
        {"quarter": "2020-Q2", "mean": 250.0, "min": 250.0, "max": 250.0, "n": 1}
    ]


def test_f0_quarterly_quarter_without_plottable_clips_is_absent() -> None:
    """Rule 6: a quarter whose clips all have nan medians is simply
    absent (a gap — never an invented point); so is an entry with no
    month at all (it cannot be placed on a timeline)."""
    entries = [
        _entry("2020-07", math.nan, 20.0, "julnan0001"),
        _entry("2020-08", math.nan, 20.0, "augnan0001"),
        _entry(None, 250.0, 35.0, "nomonth001"),
    ]
    assert series.f0_quarterly(entries) == []


def test_f0_quarterly_qc_filters_before_aggregating() -> None:
    """Rule 7: the qc=True variant applies the shared QC rule BEFORE
    aggregating — iqr >= 200, nan-median, and median >= 600 all drop,
    leaving the two clean clips; the all-clip variant keeps every
    finite-median clip (700 Hz included) so the cleaning stays visible."""
    entries = [
        _entry("2021-01", 220.0, 35.0, "jan-pass01"),
        _entry("2021-02", 240.0, 45.0, "feb-pass01"),
        _entry("2021-03", 230.0, 260.0, "mar-iqr001"),
        _entry("2021-01", 700.0, 40.0, "jan-high01"),
        _entry("2021-02", math.nan, 30.0, "feb-nan001"),
    ]
    assert series.f0_quarterly(entries, qc=True) == [
        {"quarter": "2021-Q1", "mean": 230.0, "min": 220.0, "max": 240.0, "n": 2}
    ]
    assert series.f0_quarterly(entries) == [
        {
            "quarter": "2021-Q1",
            "mean": 347.5,
            "min": 220.0,
            "max": 700.0,
            "n": 4,
        }
    ]


def test_f0_quarterly_all_fail_quarter_is_gap_in_qc_only() -> None:
    """Rule 7 (gap semantics, mirrors the real 2024-Q3/Q4 & 2025-Q3
    quarters): a quarter where every clip fails QC is absent from the QC
    series (gaps, not zeros) but present in the all-clip series."""
    entries = [
        _entry("2024-07", 300.0, 260.0, "jul-iqr001"),
        _entry("2024-08", 320.0, 300.0, "aug-iqr001"),
        _entry("2024-09", 280.0, math.nan, "sep-noiqr01"),
    ]
    assert series.f0_quarterly(entries, qc=True) == []
    assert [p["quarter"] for p in series.f0_quarterly(entries)] == ["2024-Q3"]


def test_write_quarterly_plots_creates_exactly_two_pngs(tmp_path: Path) -> None:
    """Rule 8: write_quarterly_plots writes exactly f0_quarterly_all.png
    and f0_quarterly_qc.png — n=1 quarters, all-failing quarters and nan
    gaps must not crash plotting (existence only, no image comparison)."""
    entries = [
        _entry("2020-04", 220.0, 35.0, "apr000001"),
        _entry("2020-05", 240.0, 45.0, "may000001"),
        _entry("2020-06", 250.0, 35.0, "jun000001"),
        _entry("2020-07", 300.0, 260.0, "jul-iqr001"),
        _entry("2020-10", math.nan, 30.0, "octnan0001"),
    ]
    series.write_quarterly_plots(entries, tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "f0_quarterly_all.png",
        "f0_quarterly_qc.png",
    }
