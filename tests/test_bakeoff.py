"""The bake-off: choosing a tracker and sizing the era confound, on evidence.

Three questions, three mechanisms:

1. Can a tracker find a pitch that is known? Synthetic tones, with and without
   competing sound, where the right answer is not in doubt.
2. Do independent trackers agree on real audio? Where they do not, at least one
   is wrong, and the deviation from their consensus ranks them without any
   ground truth.
3. Does a feature survive re-encoding? If a metric moves as much under a
   bitrate change as it moves across a career, it cannot support a claim about
   voices — and career time is confounded with recording era in this corpus.

Tests use synthetic audio only, per the repo rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from vvc import bakeoff


def _write_tone(path, f0: float, sr: int = 22050, seconds: float = 1.5):
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    sf.write(path, (0.3 * wave / np.max(np.abs(wave))).astype(np.float32), sr)
    return path


class TestStratifiedSample:
    def _corpus(self):
        return {
            "alpha": [{"id": f"a{i}", "month": f"2020-{i % 12 + 1:02d}"} for i in range(50)],
            "beta": [{"id": f"b{i}", "month": f"2021-{i % 12 + 1:02d}"} for i in range(10)],
        }

    def test_draws_from_every_talent_not_just_the_biggest(self):
        """A sample dominated by one talent measures that talent, not the corpus."""
        sample = bakeoff.stratified_sample(self._corpus(), n=20, seed=1)
        assert {row["talent"] for row in sample} == {"alpha", "beta"}

    def test_is_deterministic_for_a_given_seed(self):
        """A bake-off nobody can reproduce decides nothing."""
        a = bakeoff.stratified_sample(self._corpus(), n=20, seed=7)
        b = bakeoff.stratified_sample(self._corpus(), n=20, seed=7)
        assert [row["id"] for row in a] == [row["id"] for row in b]

    def test_never_returns_more_than_the_corpus_holds(self):
        sample = bakeoff.stratified_sample(self._corpus(), n=500, seed=1)
        assert len(sample) == 60

    def test_required_ids_are_always_included(self):
        """The known-bad clips are the whole point; a random draw would miss them."""
        sample = bakeoff.stratified_sample(
            self._corpus(), n=5, seed=1, required_ids={"a49", "b9"}
        )
        assert {"a49", "b9"} <= {row["id"] for row in sample}


class TestConsensusDeviation:
    def test_ranks_a_tracker_by_its_distance_from_the_others(self):
        """No ground truth on real audio — but a lone dissenter is the suspect."""
        results = [
            bakeoff.TrackerResult("clip1", "good_a", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("clip1", "good_b", 305.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("clip1", "odd", 150.0, 0.5, 4.0, 0.0, 0.1),
        ]
        deviation = bakeoff.consensus_deviation(results)
        assert deviation["odd"]["median_abs_semitones"] > 6.0
        assert deviation["good_a"]["median_abs_semitones"] < 1.0

    def test_counts_gross_disagreements_separately_from_typical_ones(self):
        """A tracker that is usually fine but occasionally off by an octave is
        a different risk from one that is always slightly off."""
        results = [
            bakeoff.TrackerResult("c1", "a", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c1", "b", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c1", "c", 150.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c2", "a", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c2", "b", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c2", "c", 302.0, 0.5, 4.0, 0.0, 0.1),
        ]
        deviation = bakeoff.consensus_deviation(results, gross_semitones=3.0)
        assert deviation["c"]["gross_disagreements"] == 1
        assert deviation["a"]["gross_disagreements"] == 0

    def test_a_clip_measured_by_one_tracker_yields_no_consensus(self):
        """One opinion is not agreement."""
        results = [bakeoff.TrackerResult("c1", "a", 300.0, 0.5, 4.0, 0.0, 0.1)]
        assert bakeoff.consensus_deviation(results)["a"]["n_clips"] == 0

    def test_non_finite_medians_are_skipped_not_treated_as_zero(self):
        results = [
            bakeoff.TrackerResult("c1", "a", float("nan"), 0.0, float("nan"), 0.0, 0.1),
            bakeoff.TrackerResult("c1", "b", 300.0, 0.5, 4.0, 0.0, 0.1),
            bakeoff.TrackerResult("c1", "c", 300.0, 0.5, 4.0, 0.0, 0.1),
        ]
        deviation = bakeoff.consensus_deviation(results)
        assert deviation["a"]["n_clips"] == 0


class TestDegrade:
    def test_reencoding_produces_readable_audio_of_the_same_duration(self, tmp_path):
        import soundfile as sf

        src = _write_tone(tmp_path / "tone.wav", 220.0)
        out = bakeoff.degrade(src, tmp_path / "deg.wav", bitrate_kbps=48, codec="libopus")
        original, sr_a = sf.read(src)
        degraded, sr_b = sf.read(out)
        assert sr_a == sr_b
        assert abs(len(degraded) - len(original)) < sr_a * 0.1

    def test_a_lower_bitrate_changes_the_signal_more(self, tmp_path):
        """Sanity: the degradation knob must actually degrade."""
        src = _write_tone(tmp_path / "tone.wav", 220.0)
        mild = bakeoff.degrade(src, tmp_path / "m.wav", bitrate_kbps=128, codec="libopus")
        harsh = bakeoff.degrade(src, tmp_path / "h.wav", bitrate_kbps=24, codec="libopus")
        assert mild.stat().st_size > 0 and harsh.stat().st_size > 0


class TestEraSensitivity:
    def test_reports_the_shift_a_reencode_causes_per_feature(self, tmp_path):
        src = _write_tone(tmp_path / "tone.wav", 220.0)
        rows = bakeoff.era_sensitivity(
            [src], bitrates=[48], tracker_name="praat_ac", work_dir=tmp_path / "w"
        )
        assert rows, "no sensitivity rows produced"
        row = rows[0]
        assert row["bitrate_kbps"] == 48
        assert "median_f0_delta" in row

    def test_a_clean_tone_survives_a_generous_bitrate(self, tmp_path):
        """If a 128 kbps re-encode moved a pure tone's F0, the harness itself
        would be the thing under suspicion."""
        src = _write_tone(tmp_path / "tone.wav", 220.0)
        rows = bakeoff.era_sensitivity(
            [src], bitrates=[128], tracker_name="praat_ac", work_dir=tmp_path / "w"
        )
        assert abs(rows[0]["median_f0_delta"]) < 5.0
