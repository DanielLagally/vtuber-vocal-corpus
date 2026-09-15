"""The v2 quality gate.

v1's gate had four checks and the wrong four (docs/v1/LIMITATIONS.md):

- its upper F0 bound equalled the tracker's own search ceiling, so the tracker
  could never report a value that would trip it — the gate was unreachable;
- it had no lower bound at all, so a 76 Hz median passed cleanly;
- its spread threshold was absolute Hz, which is a weaker constraint on a high
  voice than a low one — backwards for a corpus about high-pitched voices;
- and it checked nothing for the failure that actually corrupts this corpus.

That last point is the important one. The implausibly low readings are **not**
octave errors: Praat autocorrelation, Praat filtered autocorrelation and CREPE
independently agree on them, and the clips carry a large voiced mass far below
the talent's range that their same-month counterparts do not. They are
contaminated — another speaker, game or video dialogue, music. Raising the
pitch floor until the number looked plausible would have manufactured a result
from audio that is not the talent. Detection is the only honest response.

v1's gate is untouched: it still has to reproduce the published corpus.
"""

from __future__ import annotations

import math

import pytest

from vvc import acoustics, quality


def _features(**overrides) -> dict:
    base = {
        "median_f0": 300.0,
        "f0_iqr_semitones": 5.0,
        "voiced_fraction": 0.5,
        "fraction_below_160hz": 0.02,
    }
    base.update(overrides)
    return base


class TestGates:
    def test_a_healthy_clip_passes(self):
        assert quality.verdict(_features()) == (True, None)

    def test_a_missing_median_fails_first(self):
        passed, reason = quality.verdict(_features(median_f0=math.nan))
        assert (passed, reason) == (False, quality.REASON_MISSING)

    def test_too_little_voice_fails(self):
        """Almost no voiced content can post a deceptively tight spread purely
        from having too few points to spread across."""
        passed, reason = quality.verdict(_features(voiced_fraction=0.02))
        assert (passed, reason) == (False, quality.REASON_VOICED)

    def test_an_implausibly_wide_spread_fails(self):
        passed, reason = quality.verdict(_features(f0_iqr_semitones=24.0))
        assert (passed, reason) == (False, quality.REASON_SPREAD)

    def test_contaminated_audio_fails_on_its_own_gate(self):
        """The v1 blind spot: a clip whose voiced mass sits mostly below the
        talent's range is another speaker, not a low reading."""
        passed, reason = quality.verdict(_features(median_f0=130.0, fraction_below_160hz=0.7))
        assert (passed, reason) == (False, quality.REASON_CONTAMINATED)

    def test_both_ends_of_the_plausible_range_are_checked(self):
        assert quality.verdict(_features(median_f0=40.0))[1] == quality.REASON_RANGE
        assert quality.verdict(_features(median_f0=900.0))[1] == quality.REASON_RANGE


class TestScaleFreeSpread:
    def test_spread_is_judged_in_semitones_not_hz(self):
        """Same musical spread, different registers — same verdict."""
        low = _features(median_f0=200.0, f0_iqr_semitones=5.0)
        high = _features(median_f0=400.0, f0_iqr_semitones=5.0)
        assert quality.verdict(low)[0] == quality.verdict(high)[0] is True

    def test_hz_spread_alone_is_not_consulted(self):
        """A record carrying only the old Hz spread must not silently pass a
        check it was never measured against."""
        passed, reason = quality.verdict(
            {"median_f0": 300.0, "f0_iqr_hz": 190.0, "voiced_fraction": 0.5}
        )
        assert passed is False
        assert reason == quality.REASON_SPREAD


class TestRangeGateIsNotDeadCode:
    def test_the_range_gate_does_not_duplicate_a_tracker_bound(self):
        """v1's ceiling equalled the tracker's, so it could never fire. A
        threshold that duplicates a tracker parameter is not a quality gate."""
        config = quality.DEFAULT_CONFIG
        tracker = acoustics.DEFAULT_CONFIG
        assert config.f0_max_hz != tracker.ceiling
        assert config.f0_min_hz != tracker.floor

    def test_the_range_gate_sits_inside_what_the_tracker_can_report(self):
        """Otherwise it is unreachable in the other direction."""
        config = quality.DEFAULT_CONFIG
        tracker = acoustics.DEFAULT_CONFIG
        assert tracker.floor < config.f0_min_hz < config.f0_max_hz < tracker.ceiling


class TestVerdictAny:
    """A v2 corpus holds both schemas: clips whose audio survived were
    re-measured, clips whose audio is gone keep their v1 features. Anything
    reading across such a file needs one entry point that does not silently
    fail every legacy record."""

    def test_a_v2_record_is_judged_by_the_v2_gate(self):
        assert quality.verdict_any(_features()) == (True, None)

    def test_a_v1_record_is_judged_by_the_v1_gate(self):
        v1_shaped = {"median_f0": 300.0, "f0_iqr": 120.0, "voiced_fraction": 0.5}
        assert quality.verdict_any(v1_shaped)[0] is True

    def test_a_v1_record_is_not_failed_merely_for_lacking_v2_fields(self):
        """The bug this prevents: a legacy record vanishing from every
        aggregate because it was measured before the schema changed."""
        v1_shaped = {"median_f0": 300.0, "f0_iqr": 120.0, "voiced_fraction": 0.5}
        assert quality.verdict_any(v1_shaped)[1] is None

    def test_the_two_schemas_are_told_apart_by_the_spread_field(self):
        """Hz and semitone spreads are different scales; converting without the
        quartiles would be inventing data, so dispatch is on that field."""
        v1_shaped = {"median_f0": 300.0, "f0_iqr": 250.0, "voiced_fraction": 0.5}
        assert quality.verdict_any(v1_shaped)[0] is False


class TestRequalify:
    def test_recomputes_a_stored_verdict_without_touching_features(self):
        records = [{"features": _features(), "qc": {"pass": False, "reason": "stale"}}]
        out = quality.requalify(records)
        assert out[0]["qc"] == {"pass": True, "reason": None}
        assert out[0]["features"] == _features()

    def test_does_not_mutate_the_input(self):
        records = [{"features": _features(), "qc": {"pass": False, "reason": "stale"}}]
        quality.requalify(records)
        assert records[0]["qc"]["reason"] == "stale"

    def test_a_record_without_features_is_left_alone_not_failed(self):
        records = [{"id": "x"}]
        assert quality.requalify(records) == [{"id": "x"}]


class TestMissingInputs:
    def test_a_missing_contamination_measure_does_not_silently_pass(self):
        """v2 records always carry it; a record that does not was measured by
        something else, and must not be waved through a check it never had."""
        passed, reason = quality.verdict(
            {"median_f0": 300.0, "f0_iqr_semitones": 5.0, "voiced_fraction": 0.5},
            require_contamination=True,
        )
        assert passed is False
        assert reason == quality.REASON_CONTAMINATED

    def test_booleans_are_not_accepted_as_numbers(self):
        passed, _ = quality.verdict(_features(median_f0=True))
        assert passed is False
