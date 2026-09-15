"""Flat exports: the corpus in a form someone can actually check.

Four non-destructive tables, so a reader can reproduce any published figure
from the measurements rather than taking prose on trust.

The correlation table carries the rule worth stating outright: between-talent
and within-talent correlations are reported separately and never pooled. "Do
higher-pitched people sound brighter?" and "when this person's pitch rises,
does their brightness rise?" are different questions with different answers,
and a pooled coefficient is a mixture of the two that answers neither.
"""

from __future__ import annotations

import csv
import math

import pytest

from vvc import exports


def _rec(vid, month, f0, brightness=2000.0, *, passed=True):
    return {
        "id": vid,
        "month": month,
        "topic_id": "talk",
        "available_at": f"{month}-15T00:00:00.000Z",
        "features": {
            "median_f0": f0,
            "brightness_hz": brightness,
            "f0_iqr_semitones": 4.0,
            "voiced_fraction": 0.5,
            "fraction_below_160hz": 0.01,
        },
        "qc": {"pass": passed, "reason": None if passed else "f0_spread"},
        "legacy": False,
    }


@pytest.fixture
def by_talent():
    return {
        "alpha": [
            _rec("a1", "2020-01", 300.0, 2000.0),
            _rec("a2", "2020-01", 310.0, 2100.0),
            _rec("a3", "2020-02", 290.0, 1900.0),
            _rec("a4", "2020-03", 280.0, 1850.0),
        ],
        "beta": [
            _rec("b1", "2020-01", 200.0, 1500.0),
            _rec("b2", "2020-02", 210.0, 1600.0),
            _rec("b3", "2020-03", 205.0, 1550.0),
        ],
    }


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestClipsCsv:
    def test_one_row_per_measurement(self, by_talent, tmp_path):
        path = exports.write_clips_csv(by_talent, tmp_path / "clips.csv")
        assert len(_read(path)) == 7

    def test_carries_talent_qc_and_stream_type(self, by_talent, tmp_path):
        path = exports.write_clips_csv(by_talent, tmp_path / "clips.csv")
        row = _read(path)[0]
        assert row["talent"] == "alpha"
        assert row["qc_pass"] in {"True", "False"}
        assert row["topic_id"] == "talk"

    def test_includes_failing_clips_so_exclusions_are_visible(self, tmp_path):
        data = {"alpha": [_rec("a1", "2020-01", 300.0), _rec("a2", "2020-01", 90.0, passed=False)]}
        path = exports.write_clips_csv(data, tmp_path / "clips.csv")
        assert len(_read(path)) == 2


class TestTalentMonthCsv:
    def test_one_row_per_talent_month(self, by_talent, tmp_path):
        path = exports.write_talent_month_csv(by_talent, tmp_path / "tm.csv")
        rows = _read(path)
        assert len(rows) == 6  # alpha 3 months + beta 3 months

    def test_reports_clip_count_and_within_month_disagreement(self, by_talent, tmp_path):
        path = exports.write_talent_month_csv(by_talent, tmp_path / "tm.csv")
        row = next(r for r in _read(path) if r["talent"] == "alpha" and r["month"] == "2020-01")
        assert row["n_pass"] == "2"
        assert float(row["within_month_abs_diff"]) == pytest.approx(10.0)

    def test_a_single_clip_month_has_no_disagreement_rather_than_zero(
        self, by_talent, tmp_path
    ):
        """Zero would claim perfect agreement from one measurement."""
        path = exports.write_talent_month_csv(by_talent, tmp_path / "tm.csv")
        row = next(r for r in _read(path) if r["talent"] == "beta" and r["month"] == "2020-01")
        assert row["within_month_abs_diff"] == ""


class TestTalentSummaryCsv:
    def test_one_row_per_talent(self, by_talent, tmp_path):
        path = exports.write_talent_summary_csv(by_talent, tmp_path / "ts.csv")
        assert {r["talent"] for r in _read(path)} == {"alpha", "beta"}

    def test_reports_the_trend_with_its_fit_quality(self, by_talent, tmp_path):
        """A slope without a fit figure invites seeing a trend that is not there."""
        path = exports.write_talent_summary_csv(by_talent, tmp_path / "ts.csv")
        row = _read(path)[0]
        assert "median_f0_slope_per_year" in row
        assert "median_f0_trend_r2" in row

    def test_reports_coverage_and_the_noise_floor(self, by_talent, tmp_path):
        path = exports.write_talent_summary_csv(by_talent, tmp_path / "ts.csv")
        row = _read(path)[0]
        assert row["months_covered"]
        assert "noise_floor_median_f0" in row


class TestCorrelationsCsv:
    def test_reports_between_and_within_separately(self, by_talent, tmp_path):
        path = exports.write_correlations_csv(by_talent, tmp_path / "c.csv")
        row = _read(path)[0]
        assert "between_talent_r" in row
        assert "within_talent_r" in row

    def test_never_emits_a_pooled_coefficient(self, by_talent, tmp_path):
        """A pooled r mixes two different questions and answers neither."""
        path = exports.write_correlations_csv(by_talent, tmp_path / "c.csv")
        with open(path, encoding="utf-8") as handle:
            header = handle.readline()
        assert "pooled" not in header.lower()

    def test_one_row_per_metric_pair_not_per_ordered_pair(self, by_talent, tmp_path):
        path = exports.write_correlations_csv(
            by_talent, tmp_path / "c.csv", metrics=("median_f0", "brightness_hz")
        )
        assert len(_read(path)) == 1


class TestVerdictsAreRecomputed:
    """v1 shipped this bug: its plots recomputed the QC verdict while its site
    export read the stored one, so the two could disagree about which clips
    existed. The exports must agree with the noise floor, which recomputes."""

    def test_a_stale_stored_pass_does_not_get_a_clip_into_the_aggregates(self, tmp_path):
        stale = _rec("bad", "2020-01", 300.0)
        stale["features"]["f0_iqr_semitones"] = 40.0  # the live rule fails this
        stale["qc"] = {"pass": True, "reason": None}  # the stored copy lies
        good = _rec("ok", "2020-01", 310.0)

        path = exports.write_talent_month_csv({"alpha": [stale, good]}, tmp_path / "tm.csv")
        row = _read(path)[0]
        assert row["n_clips"] == "2"
        assert row["n_pass"] == "1"

    def test_a_stale_stored_fail_does_not_hide_a_passing_clip(self, tmp_path):
        hidden = _rec("hidden", "2020-01", 300.0)
        hidden["qc"] = {"pass": False, "reason": "stale"}
        path = exports.write_talent_month_csv({"alpha": [hidden]}, tmp_path / "tm.csv")
        assert _read(path)[0]["n_pass"] == "1"


class TestPearson:
    def test_perfect_positive_relationship(self):
        assert exports.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_too_few_points_is_nan_not_zero(self):
        assert math.isnan(exports.pearson([1.0], [2.0]))

    def test_no_variance_is_nan_not_one(self):
        assert math.isnan(exports.pearson([3, 3, 3], [1, 2, 3]))
