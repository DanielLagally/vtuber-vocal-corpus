"""Reliability: the measurement's own noise floor, and cross-clip review flags.

Two clips of the same talent in the same month are the same person, the same
period, and the same pipeline — so the difference between them is what the
instrument contributes. Without that number, no trend and no between-talent
difference can be called large or small, because there is nothing to call it
large or small relative to.

The same pairing catches the failure QC cannot: a clip whose pitch is read an
octave off still lands inside every plausibility gate, but it cannot agree with
its own same-month counterpart. Flagging is non-destructive — it marks data for
a human, it never drops it.
"""

from __future__ import annotations

import math

import pytest

from vvc import reliability


def _rec(month: str, f0: float, *, iqr: float = 100.0, voiced: float = 0.5, vid: str = "x"):
    return {
        "id": vid,
        "month": month,
        "features": {
            "median_f0": f0,
            "f0_iqr": iqr,
            "voiced_fraction": voiced,
        },
        "qc": {"pass": True, "reason": None},
    }


class TestWithinMonthPairs:
    def test_a_month_with_one_clip_contributes_no_pair(self):
        """One clip cannot disagree with itself — it is not evidence about noise."""
        pairs = reliability.within_month_pairs([_rec("2020-01", 300.0)])
        assert pairs == []

    def test_a_month_with_two_clips_contributes_one_pair(self):
        pairs = reliability.within_month_pairs(
            [_rec("2020-01", 300.0, vid="a"), _rec("2020-01", 320.0, vid="b")]
        )
        assert len(pairs) == 1
        assert pairs[0].month == "2020-01"
        assert pairs[0].values == (300.0, 320.0)

    def test_every_unordered_pair_in_a_month_is_counted(self):
        """Three clips make three pairs, not two: all disagreements are evidence."""
        pairs = reliability.within_month_pairs(
            [
                _rec("2020-01", 300.0, vid="a"),
                _rec("2020-01", 320.0, vid="b"),
                _rec("2020-01", 340.0, vid="c"),
            ]
        )
        assert len(pairs) == 3

    def test_clips_in_different_months_are_never_paired(self):
        """Across months, a difference could be real change — not instrument noise."""
        pairs = reliability.within_month_pairs(
            [_rec("2020-01", 300.0, vid="a"), _rec("2020-02", 320.0, vid="b")]
        )
        assert pairs == []

    def test_qc_failing_clips_are_excluded(self):
        """A known-bad clip would inflate the noise floor with junk we already reject."""
        bad = _rec("2020-01", 300.0, iqr=500.0, vid="bad")  # fails the IQR gate
        pairs = reliability.within_month_pairs([bad, _rec("2020-01", 320.0, vid="ok")])
        assert pairs == []

    def test_qc_is_recomputed_not_read_from_the_stored_verdict(self):
        """Stored verdicts go stale when a threshold moves; the rule does not."""
        stale = _rec("2020-01", 300.0, iqr=500.0, vid="stale")
        stale["qc"] = {"pass": True, "reason": None}  # lies
        pairs = reliability.within_month_pairs([stale, _rec("2020-01", 320.0, vid="ok")])
        assert pairs == []

    def test_non_finite_values_are_skipped(self):
        pairs = reliability.within_month_pairs(
            [_rec("2020-01", math.nan, vid="a"), _rec("2020-01", 320.0, vid="b")]
        )
        assert pairs == []


class TestNoiseFloor:
    def test_reports_the_typical_same_month_disagreement(self):
        records = [
            _rec("2020-01", 300.0, vid="a"),
            _rec("2020-01", 310.0, vid="b"),  # 10 Hz apart
            _rec("2020-02", 300.0, vid="c"),
            _rec("2020-02", 330.0, vid="d"),  # 30 Hz apart
        ]
        floor = reliability.noise_floor(records)["median_f0"]
        assert floor["n_pairs"] == 2
        assert floor["median_abs_diff"] == pytest.approx(20.0)

    def test_reports_spread_in_semitones_too(self):
        """Hz disagreement is pitch-scale dependent; semitones compare across voices."""
        records = [_rec("2020-01", 200.0, vid="a"), _rec("2020-01", 400.0, vid="b")]
        floor = reliability.noise_floor(records)["median_f0"]
        assert floor["median_abs_semitones"] == pytest.approx(12.0)

    def test_a_corpus_with_no_pairs_reports_no_floor_rather_than_zero(self):
        """Zero would read as a perfect instrument. Unknown must stay unknown."""
        floor = reliability.noise_floor([_rec("2020-01", 300.0)])["median_f0"]
        assert floor["n_pairs"] == 0
        assert floor["median_abs_diff"] is None


class TestDiscordanceFlags:
    def test_flags_a_same_month_pair_that_disagrees_by_more_than_the_threshold(self):
        """Nobody's speaking pitch halves and recovers inside one month."""
        records = [
            _rec("2020-01", 300.0, vid="high"),
            _rec("2020-01", 150.0, vid="low"),  # exactly an octave down
        ]
        flags = reliability.discordance_flags(records, semitone_threshold=6.0)
        assert len(flags) == 1
        assert set(flags[0].ids) == {"high", "low"}
        assert flags[0].semitones == pytest.approx(12.0)

    def test_does_not_flag_ordinary_month_to_month_variation(self):
        records = [_rec("2020-01", 300.0, vid="a"), _rec("2020-01", 320.0, vid="b")]
        assert reliability.discordance_flags(records, semitone_threshold=6.0) == []

    def test_flagging_never_removes_a_record(self):
        """Flags mark data for a human; they are not a second QC gate."""
        records = [_rec("2020-01", 300.0, vid="high"), _rec("2020-01", 150.0, vid="low")]
        before = len(records)
        reliability.discordance_flags(records)
        assert len(records) == before
        assert all(r["qc"]["pass"] for r in records)

    def test_flagged_ids_are_reported_so_the_suspect_clip_can_be_found(self):
        records = [_rec("2020-03", 400.0, vid="v1"), _rec("2020-03", 100.0, vid="v2")]
        flags = reliability.discordance_flags(records)
        assert flags[0].month == "2020-03"
        assert flags[0].values == (400.0, 100.0)
