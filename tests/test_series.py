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
9. Multi-clip months (STATE R3): when several records share a month,
   the monthly series collapses to ONE point per month — the plain
   float mean (NO rounding) of that month's values for the series at
   hand. f0_series: the mean of finite median_f0 values; with
   qc=True the shared QC rule filters FIRST, then the survivors are
   averaged — a month with no surviving value is a gap in that
   series (but can still be present in the all-clip series).
   iqr_series: the mean of finite f0_iqr values. A single-record
   month is unchanged: mean of one is the value itself.

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


# ---------------------------------------------------- multi-clip months


def test_f0_series_multi_clip_month_mean_all_and_qc() -> None:
    """Rule 9: two records share 2025-07 — one QC-pass (median 300) and
    one QC-fail (iqr 250, median 500, still finite). The all-series
    month point is the mean of the finite medians (400.0); the QC
    series filters FIRST and averages only the survivor (300.0). One
    point per month, not one per record."""
    entries = [
        _entry("2025-07", 300.0, 40.0, "julpass001"),
        _entry("2025-07", 500.0, 250.0, "julfail001"),
    ]
    assert _pairs(series.f0_series(entries)) == [("2025-07", 400.0)]
    assert _pairs(series.f0_series(entries, qc=True)) == [("2025-07", 300.0)]


def test_f0_series_multi_clip_month_two_qc_pass_mean() -> None:
    """Rule 9: two same-month QC-pass records (medians 300 and 340) —
    both the all-series and the QC series collapse to the single month
    point 320.0."""
    entries = [
        _entry("2025-07", 300.0, 40.0, "julpass003"),
        _entry("2025-07", 340.0, 45.0, "julpass004"),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == [("2025-07", 320.0)]
    assert _pairs(series.f0_series(entries)) == [("2025-07", 320.0)]


def test_f0_series_all_fail_month_gap_in_qc_present_in_all() -> None:
    """Rule 9: a month whose records all fail QC is a gap in the QC
    series but still present in the all-clip series (mean of its finite
    medians) — gaps are per-series, never zero-filled."""
    entries = [
        _entry("2025-07", 300.0, 260.0, "juliqr002"),
        _entry("2025-07", 320.0, 280.0, "juliqr004"),
        _entry("2025-08", 320.0, 300.0, "augiqr002"),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == []
    assert _pairs(series.f0_series(entries)) == [
        ("2025-07", 310.0),
        ("2025-08", 320.0),
    ]


def test_f0_series_single_record_month_unchanged_mean_of_one() -> None:
    """Rule 9: single-record months behave exactly as before, next to a
    multi-clip month — mean of one is the value itself, month order
    ascending."""
    entries = [
        _entry("2025-02", 300.0, 40.0, "febclip001"),
        _entry("2025-01", 220.0, 35.0, "jansolo002"),
        _entry("2025-02", 340.0, 45.0, "febclip002"),
    ]
    assert _pairs(series.f0_series(entries)) == [
        ("2025-01", 220.0),
        ("2025-02", 320.0),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == [
        ("2025-01", 220.0),
        ("2025-02", 320.0),
    ]


def test_f0_series_month_mean_is_plain_float_no_rounding() -> None:
    """Rule 9 (rounding pin): the month point is the plain float mean —
    (300 + 341) / 2 = 320.5 must survive exactly, not be rounded."""
    entries = [
        _entry("2025-07", 300.0, 40.0, "julpass005"),
        _entry("2025-07", 341.0, 45.0, "julpass006"),
    ]
    assert _pairs(series.f0_series(entries, qc=True)) == [
        ("2025-07", (300.0 + 341.0) / 2)
    ]


def test_iqr_series_multi_clip_month_mean_of_finite_iqrs() -> None:
    """Rule 9: iqr_series collapses same-month records to the mean of
    their finite f0_iqr values (270 and 230 -> 250.0); a non-finite IQR
    record contributes nothing (mean of the remaining finite one); a
    NaN-median record with a finite IQR still counts (per-series gaps —
    existing rule 2, now per-month); single-record months are the bare
    value."""
    entries = [
        _entry("2025-07", 300.0, 250.0, "juliqr003"),
        _entry("2025-07", 300.0, math.nan, "julnan002"),
        _entry("2025-08", 300.0, 270.0, "augiqr003"),
        _entry("2025-08", 300.0, 230.0, "augiqr004"),
        _entry("2025-09", math.nan, 25.0, "sepnan003"),
    ]
    assert _pairs(series.iqr_series(entries)) == [
        ("2025-07", 250.0),
        ("2025-08", 250.0),
        ("2025-09", 25.0),
    ]


# ---------------------------------------------------------------- yearly


def test_f0_yearly_median_of_clip_medians_min_max_n_sorted() -> None:
    """Rule 10 (PLAN L36 year rule): per calendar year the MEDIAN of that
    year's clip medians (not the mean — clips 300/340/520 give 340, a mean
    would give ~386.7), plus between-clip min/max and n, sorted by year.
    The nan-median clip is excluded ENTIRELY (it cannot plot)."""
    entries = [
        _entry("2024-03", 300.0, 45.0, "mar0000001"),
        _entry("2024-07", 340.0, 50.0, "jul0000001"),
        _entry("2024-11", 520.0, 90.0, "nov0000001"),
        _entry("2023-05", 280.0, 40.0, "may0000001"),
        _entry("2024-09", math.nan, 30.0, "sepnan0001"),
    ]
    assert series.f0_yearly(entries) == [
        {"year": "2023", "median": 280.0, "min": 280.0, "max": 280.0, "n": 1},
        {
            "year": "2024",
            "median": 340.0,
            "min": 300.0,
            "max": 520.0,
            "n": 3,
        },
    ]


def test_f0_yearly_is_clip_level_not_month_mean_first() -> None:
    """Rule 10: the year aggregates CLIP medians directly — two passing
    clips in one month (300, 340) plus another month (400) give a year
    median of 340; aggregating month-means first would give 360. The clip
    is the sample (PLAN), not the month."""
    entries = [
        _entry("2024-01", 300.0, 40.0, "janclip0001"),
        _entry("2024-01", 340.0, 45.0, "janclip0002"),
        _entry("2024-02", 400.0, 45.0, "febclip0001"),
    ]
    assert series.f0_yearly(entries, qc=True) == [
        {"year": "2024", "median": 340.0, "min": 300.0, "max": 400.0, "n": 3}
    ]


def test_f0_yearly_qc_filters_first_all_fail_year_is_gap() -> None:
    """Rule 10 + rule 7 semantics: qc=True applies the shared QC rule
    BEFORE aggregating — a year whose clips all fail (iqr >= 200 / >=600 /
    nan) is absent from the QC series (a gap, never an invented point) and
    present in the all-clip series (finite medians only)."""
    entries = [
        _entry("2024-01", 300.0, 260.0, "janiqr0001"),
        _entry("2024-06", 615.0, 140.0, "juncap0001"),
        _entry("2024-10", math.nan, 30.0, "octnan0001"),
        _entry("2025-02", 320.0, 45.0, "febpass0001"),
    ]
    assert series.f0_yearly(entries, qc=True) == [
        {"year": "2025", "median": 320.0, "min": 320.0, "max": 320.0, "n": 1}
    ]
    assert series.f0_yearly(entries) == [
        {"year": "2024", "median": 300.0, "min": 300.0, "max": 615.0, "n": 2},
        {"year": "2025", "median": 320.0, "min": 320.0, "max": 320.0, "n": 1},
    ]


def test_f0_yearly_single_clip_and_unplaceable_entries() -> None:
    """Rule 10: a n=1 year carries min == max == the single value; an
    entry with no month at all cannot be placed on a timeline and is
    excluded from every yearly point."""
    entries = [
        _entry("2025-05", 310.0, 45.0, "may0000002"),
        _entry(None, 400.0, 45.0, "nomonth0001"),
    ]
    assert series.f0_yearly(entries, qc=True) == [
        {"year": "2025", "median": 310.0, "min": 310.0, "max": 310.0, "n": 1}
    ]


def test_write_yearly_plot_creates_exactly_one_png(tmp_path: Path) -> None:
    """Rule 10 (plot): write_yearly_plot writes exactly f0_yearly.png —
    the QC view with per-year sample size (n bars) and between-clip
    min–max spread; years, gaps and n=1 years must not crash plotting
    (existence only, no image comparison)."""
    entries = [
        _entry("2024-03", 300.0, 45.0, "mar0000001"),
        _entry("2024-07", 340.0, 50.0, "jul0000001"),
        _entry("2025-02", 320.0, 45.0, "feb0000001"),
        _entry("2025-04", 300.0, 260.0, "apriqr0001"),
        _entry("2026-01", math.nan, 30.0, "jannan0001"),
    ]
    series.write_yearly_plot(entries, tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == {"f0_yearly.png"}


def test_new_run_dir_never_reuses_a_directory(tmp_path: Path) -> None:
    """CLAUDE.md: plots are a permanent record. Two calls in the same
    process (same base, same label) must return two distinct, already-
    created directories — the second call must not silently reuse or
    overwrite the first."""
    first = series.new_run_dir(tmp_path, label="luna")
    second = series.new_run_dir(tmp_path, label="luna")
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
    assert first.parent == tmp_path / "runs"
    assert second.parent == tmp_path / "runs"


def test_new_run_dir_no_label_uses_bare_timestamp(tmp_path: Path) -> None:
    run_dir = series.new_run_dir(tmp_path)
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.is_dir()
    assert "-" not in run_dir.name  # %Y%m%dT%H%M%S has no dashes


def test_write_multi_talent_plot_creates_two_pngs_no_crash_on_gap(
    tmp_path: Path,
) -> None:
    """write_multi_talent_plot(talents, out_dir) writes exactly
    f0_quarterly_multi.png and f0_yearly_multi.png. Talents with
    non-overlapping quarters/years (one has no QC-pass data at all in
    a year/quarter the other does) must not crash — that's a real gap,
    not an error."""
    talent_a = [
        _entry("2024-01", 300.0, 45.0, "a0000001"),
        _entry("2025-01", 310.0, 45.0, "a0000002"),
    ]
    talent_b = [
        _entry("2025-01", 250.0, 45.0, "b0000001"),
    ]
    series.write_multi_talent_plot(
        {"Talent A": talent_a, "Talent B": talent_b}, tmp_path
    )
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "f0_quarterly_multi.png",
        "f0_yearly_multi.png",
    }


def test_write_feature_yearly_plot_creates_one_png_for_any_feature_key(
    tmp_path: Path,
) -> None:
    """write_feature_yearly_plot generalizes the yearly plot to any
    numeric feature (brightness, dynamism, jitter, ...) via
    feature_key; QC-pass still gates on the shared F0/IQR rule, not on
    the plotted feature itself. Existence only, no image comparison —
    matches this repo's plot-testing convention."""
    entries = []
    for month, brightness in (("2024-03", 1900.0), ("2024-07", 2100.0), ("2025-02", 2000.0)):
        entry = _entry(month, 300.0, 45.0, f"id{month.replace('-', '')}")
        entry["features"]["brightness_hz"] = brightness
        entries.append(entry)

    series.write_feature_yearly_plot(
        entries, tmp_path,
        feature_key="brightness_hz", filename="brightness_yearly.png",
        subject="Brightness by Year", subtitle="Spectral centroid.",
        unit_label="Brightness (Hz)",
    )
    assert {p.name for p in tmp_path.glob("*.png")} == {"brightness_yearly.png"}
