"""Is this clip plausibly the talent, around the time it was recorded?

The per-clip gate cannot answer this: a clip carrying a guest's voice is
perfectly plausible *on its own*, and only looks wrong beside the rest of that
talent's career. So this is a corpus-level pass, run after a talent's series
exists.

Three design choices, each forced by measurement against human labels
(`data/outlier-review/labels.json`):

1. **The threshold is in semitones, not standard deviations.** Normalising by
   each talent's own spread destroys the separation: a wide-range talent's
   contamination gets divided into invisibility while a narrow-range talent's
   legitimate extreme gets divided into looking like contamination. What makes
   a clip wrong is a fact on an absolute scale — a male voice sits about an
   octave below these talents however variable they are.

2. **The baseline follows the talent's trend rather than being one career-wide
   number.** Some talents genuinely drift most of an octave across a career; a
   fixed centre puts both ends of that near the band edge, so the gate starts
   trimming the largest real trends in the corpus — the worst thing it could do.

3. **The band is asymmetric, tight below and loose above.** Every observed
   contamination is a lower voice. Above the baseline there is no separation to
   exploit: a character performance sits *closer* to the baseline than one
   talent's perfectly ordinary wide range, so a tight upper bound only destroys
   real data.
"""

from __future__ import annotations

import pytest

from vvc import plausibility


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
        "qc": {"pass": passed, "reason": None},
    }


def _steady(n=36, f0=300.0, start_year=2020):
    return [
        _rec(f"c{i}", f"{start_year + i // 12:04d}-{i % 12 + 1:02d}", f0)
        for i in range(n)
    ]


class TestBand:
    def test_a_clip_far_below_the_baseline_is_implausible(self):
        records = _steady() + [_rec("bad", "2021-06", 140.0)]
        flagged = {f.id for f in plausibility.implausible(records)}
        assert "bad" in flagged

    def test_the_same_margin_above_is_not_flagged(self):
        """Asymmetric on purpose: contamination is a lower voice, and above the
        baseline a performance and a wide natural range are indistinguishable."""
        below = 300.0 * 2 ** (-8 / 12)
        above = 300.0 * 2 ** (8 / 12)
        low = _steady() + [_rec("low", "2021-06", below)]
        high = _steady() + [_rec("high", "2021-06", above)]
        assert "low" in {f.id for f in plausibility.implausible(low)}
        assert "high" not in {f.id for f in plausibility.implausible(high)}

    def test_an_extreme_high_reading_is_still_caught(self):
        records = _steady() + [_rec("way_up", "2021-06", 300.0 * 2 ** (14 / 12))]
        assert "way_up" in {f.id for f in plausibility.implausible(records)}

    def test_reports_the_distance_so_a_decision_can_be_audited(self):
        records = _steady() + [_rec("bad", "2021-06", 150.0)]
        flag = next(f for f in plausibility.implausible(records) if f.id == "bad")
        assert flag.semitones < -11.0
        assert flag.baseline_hz == pytest.approx(300.0, abs=5.0)


class TestFollowsDrift:
    def test_a_steadily_drifting_talent_keeps_both_ends_of_the_drift(self):
        """Some talents really do move most of an octave over a career. A
        career-wide centre would put both ends near the band edge and start
        trimming the real trend."""
        records = [
            _rec(f"d{i}", f"{2020 + i // 12:04d}-{i % 12 + 1:02d}",
                 300.0 * 2 ** ((i * 9.0 / 47) / 12))
            for i in range(48)
        ]
        assert plausibility.implausible(records) == []

    def test_contamination_is_still_caught_inside_a_drifting_series(self):
        records = [
            _rec(f"d{i}", f"{2020 + i // 12:04d}-{i % 12 + 1:02d}",
                 300.0 * 2 ** ((i * 9.0 / 47) / 12))
            for i in range(48)
        ]
        records.append(_rec("guest", "2022-01", 130.0))
        assert "guest" in {f.id for f in plausibility.implausible(records)}


class TestGuards:
    def test_a_talent_with_too_little_history_gets_no_band(self):
        """A baseline fitted to a handful of clips is not a baseline, and
        rejecting against it would be inventing a standard."""
        records = [_rec("a", "2020-01", 300.0), _rec("b", "2020-02", 140.0)]
        assert plausibility.implausible(records) == []

    def test_the_baseline_ignores_qc_failing_clips(self):
        """Junk must not be allowed to define what normal looks like."""
        records = _steady() + [_rec(f"junk{i}", f"2021-{i + 1:02d}", 120.0, passed=False)
                               for i in range(6)]
        records.append(_rec("bad", "2022-06", 150.0))
        assert "bad" in {f.id for f in plausibility.implausible(records)}

    def test_non_finite_readings_are_skipped_not_flagged(self):
        records = _steady() + [_rec("missing", "2021-06", float("nan"))]
        assert "missing" not in {f.id for f in plausibility.implausible(records)}


class TestApply:
    def test_applying_marks_records_without_mutating_the_input(self):
        records = _steady() + [_rec("bad", "2021-06", 140.0)]
        out = plausibility.apply(records)
        assert records[-1]["qc"]["pass"] is True
        marked = next(r for r in out if r["id"] == "bad")
        assert marked["qc"]["pass"] is False
        assert marked["qc"]["reason"] == plausibility.REASON

    def test_applying_leaves_plausible_records_alone(self):
        out = plausibility.apply(_steady())
        assert all(r["qc"]["pass"] for r in out)

    def test_the_distance_is_recorded_on_the_record(self):
        out = plausibility.apply(_steady() + [_rec("bad", "2021-06", 140.0)])
        marked = next(r for r in out if r["id"] == "bad")
        assert marked["plausibility_semitones"] < -6.0
