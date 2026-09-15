"""Month-first aggregation for the v2 site (reference/statistics.md).

Three rules from statistics.md drive every test here: aggregation is always
month-first (a quarter/year is built from its months' values, never pooled
clips), one estimator — the ordinary median — is used at every level, and
uncertainty is a bootstrap CI over the values that produced an aggregate,
never min-max (which is a labelled range instead).
"""

from __future__ import annotations

import math

import pytest

from vvc import series_v2


def _rec(vid, month, f0, *, passed=True):
    return {
        "id": vid,
        "month": month,
        "features": {
            "median_f0": f0,
            "f0_iqr_semitones": 4.0,
            "voiced_fraction": 0.5,
            "fraction_below_160hz": 0.01,
        },
        "qc": {"pass": passed, "reason": None if passed else "f0_spread"},
    }


class TestMonthlySeries:
    def test_one_point_per_month_using_the_ordinary_median(self):
        records = [
            _rec("a", "2020-01", 300.0),
            _rec("b", "2020-01", 310.0),
            _rec("c", "2020-01", 296.0),
            _rec("d", "2020-02", 400.0),
        ]
        points = series_v2.monthly_series(records, "median_f0")
        by_month = {p["period"]: p for p in points}
        assert by_month["2020-01"]["median"] == pytest.approx(300.0)
        assert by_month["2020-01"]["n"] == 3
        assert by_month["2020-02"]["median"] == pytest.approx(400.0)

    def test_a_failing_clip_is_excluded_not_zeroed(self):
        records = [_rec("a", "2020-01", 300.0), _rec("b", "2020-01", 9999.0, passed=False)]
        # the failing clip's own qc.pass is ignored — the gate is recomputed —
        # so make it fail the recomputed gate too via an implausible spread
        records[1]["features"]["f0_iqr_semitones"] = 99.0
        points = series_v2.monthly_series(records, "median_f0")
        assert len(points) == 1
        assert points[0]["n"] == 1
        assert points[0]["median"] == pytest.approx(300.0)

    def test_a_month_with_no_passing_clips_is_a_gap_not_a_point(self):
        records = [_rec("a", "2020-01", 300.0, passed=False)]
        records[0]["features"]["f0_iqr_semitones"] = 99.0
        assert series_v2.monthly_series(records, "median_f0") == []

    def test_single_clip_month_has_no_ci_but_has_a_point(self):
        points = series_v2.monthly_series([_rec("a", "2020-01", 300.0)], "median_f0")
        assert points[0]["n"] == 1
        assert math.isnan(points[0]["ci_low"])
        assert math.isnan(points[0]["ci_high"])

    def test_min_max_is_reported_separately_from_the_ci(self):
        records = [_rec("a", "2020-01", 290.0), _rec("b", "2020-01", 310.0)]
        point = series_v2.monthly_series(records, "median_f0")[0]
        assert point["min"] == pytest.approx(290.0)
        assert point["max"] == pytest.approx(310.0)
        # min-max is never presented as the CI: the two keys must be able to differ
        assert {"min", "max", "ci_low", "ci_high"} <= point.keys()


class TestQuarterlyAndYearlySeries:
    def test_a_densely_sampled_month_does_not_outweigh_a_sparse_one(self):
        """One month posts ten clips at 300 Hz, the other one clip at 340 Hz,
        both in the same quarter. Pooling clips directly would give a median
        near 300 (ten votes to one); month-first aggregation treats the two
        months as two equally-weighted samples."""
        records = [_rec(f"a{i}", "2020-01", 300.0) for i in range(10)]
        records.append(_rec("b", "2020-02", 340.0))
        points = series_v2.quarterly_series(records, "median_f0")
        assert len(points) == 1
        assert points[0]["median"] == pytest.approx(320.0)  # median of [300, 340]
        assert points[0]["n"] == 2  # two MONTHS contributed, not eleven clips

    def test_yearly_rolls_up_quarters_worth_of_months(self):
        records = [
            _rec("a", "2020-01", 300.0),
            _rec("b", "2020-06", 320.0),
            _rec("c", "2020-11", 340.0),
        ]
        points = series_v2.yearly_series(records, "median_f0")
        assert len(points) == 1
        assert points[0]["period"] == "2020"
        assert points[0]["median"] == pytest.approx(320.0)
        assert points[0]["n"] == 3

    def test_gap_months_leave_a_gap_not_an_interpolation(self):
        records = [_rec("a", "2020-01", 300.0), _rec("b", "2020-07", 300.0)]
        quarters = {p["period"] for p in series_v2.quarterly_series(records, "median_f0")}
        assert quarters == {"2020-Q1", "2020-Q3"}


class TestTypicalAndTrend:
    def test_typical_is_the_median_of_month_medians(self):
        records = [
            _rec("a", "2020-01", 300.0),
            _rec("b", "2020-01", 300.0),
            _rec("c", "2020-02", 340.0),
        ]
        summary = series_v2.typical_and_trend(records, "median_f0")
        assert summary["typical"] == pytest.approx(320.0)  # median([300, 340])

    def test_empty_corpus_slice_is_a_gap_not_a_zero(self):
        summary = series_v2.typical_and_trend([], "median_f0")
        assert math.isnan(summary["typical"])
        assert summary["trend"] == {}

    def test_trend_uses_the_talents_own_first_observed_month_as_career_zero(self):
        records = [
            _rec("a", "2020-01", 300.0),
            _rec("b", "2020-07", 306.0),
            _rec("c", "2021-01", 312.0),
        ]
        summary = series_v2.typical_and_trend(records, "median_f0")
        assert summary["trend"]["n"] == 3
        assert summary["trend"]["slope_per_year"] > 0

    def test_dispatches_the_gate_across_v1_and_v2_schema_records(self):
        """A talent's file can hold re-measured (semitone-spread) and legacy
        (Hz-spread) records side by side; both schemas must be gate-able."""
        legacy = {
            "id": "z",
            "month": "2020-03",
            "features": {"median_f0": 305.0, "f0_iqr": 50.0, "voiced_fraction": 0.6},
            "qc": {"pass": True, "reason": None},
        }
        records = [_rec("a", "2020-01", 300.0), legacy]
        points = series_v2.monthly_series(records, "median_f0")
        assert {p["period"] for p in points} == {"2020-01", "2020-03"}
