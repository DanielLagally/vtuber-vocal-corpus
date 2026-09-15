"""CLI surface for the v2 work: build-v2, export, analyse.

The rule that matters most here is containment. v1's measurement files feed the
published site and are the baseline every v2 comparison is made against, so a
v2 build that touched them would destroy the only reference point the project
has. That is asserted directly rather than assumed from code reading.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vvc import __main__ as cli


def _rec(vid, month, f0):
    return {
        "id": vid,
        "month": month,
        "score": 40.0,
        "window": {"start_s": 15.0, "end_s": 105.0},
        "features": {
            "median_f0": f0,
            "f0_iqr": 100.0,
            "voiced_fraction": 0.5,
            "brightness_hz": 2000.0,
        },
        "qc": {"pass": True, "reason": None},
        "model": "m.ckpt",
        "tracker": "praat_ac",
    }


@pytest.fixture
def project(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    records = [
        _rec("v1", "2020-01", 300.0),
        _rec("v2", "2020-01", 312.0),
        _rec("v3", "2020-02", 295.0),
        _rec("v4", "2020-03", 288.0),
        _rec("v5", "2020-04", 305.0),
        _rec("v6", "2020-05", 290.0),
    ]
    (measurements / "alpha_monthly.json").write_text(json.dumps(records), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


class TestBuildV2:
    def test_never_modifies_the_v1_measurement_files(self, project):
        """The single most important containment rule in the v2 work."""
        source = project / "measurements" / "alpha_monthly.json"
        before = source.read_bytes()

        cli.main(
            [
                "build-v2",
                "--measurements-dir", str(project / "measurements"),
                "--out-dir", str(project / "v2"),
                "--data-dir", str(project / "data"),
                "--cache-dir", str(project / "cache"),
            ]
        )

        assert source.read_bytes() == before

    def test_writes_to_a_separate_directory(self, project):
        cli.main(
            [
                "build-v2",
                "--measurements-dir", str(project / "measurements"),
                "--out-dir", str(project / "v2"),
                "--data-dir", str(project / "data"),
                "--cache-dir", str(project / "cache"),
            ]
        )
        assert (project / "v2" / "alpha.json").is_file()

    def test_refuses_to_write_a_corpus_when_extraction_mostly_fails(
        self, project, monkeypatch
    ):
        """A build once measured nothing and still exited zero, because every
        clip that failed to measure was recorded as merely lacking audio."""
        import numpy as np
        import soundfile as sf

        windows = project / "data" / "windows"
        windows.mkdir(parents=True)
        for vid in ("v1", "v2"):
            sf.write(windows / f"{vid}_raw90.wav", np.zeros(16000, "float32"), 16000)

        from vvc import corpus as corpus_module

        def all_fail(*args, **kwargs):
            paths = args[0]
            return corpus_module.ExtractionResult(
                {}, len(paths), {str(p): "RuntimeError: boom" for p in paths}
            )

        # The CLI imports vvc.corpus inside the command, so patch the module.
        monkeypatch.setattr(corpus_module, "extract_many", all_fail)

        with pytest.raises(SystemExit):
            cli.main(
                [
                    "build-v2",
                    "--measurements-dir", str(project / "measurements"),
                    "--out-dir", str(project / "v2"),
                    "--data-dir", str(project / "data"),
                    "--cache-dir", str(project / "cache"),
                ]
            )

    def test_clips_without_local_audio_become_legacy_records(self, project):
        """No audio exists in this fixture, so every record must be marked —
        not dropped, and not re-measured from nothing."""
        cli.main(
            [
                "build-v2",
                "--measurements-dir", str(project / "measurements"),
                "--out-dir", str(project / "v2"),
                "--data-dir", str(project / "data"),
                "--cache-dir", str(project / "cache"),
            ]
        )
        built = json.loads((project / "v2" / "alpha.json").read_text(encoding="utf-8"))
        assert len(built) == 6
        assert all(row["legacy"] for row in built)


class TestConditionalSeparation:
    """Separation is lossy: on a clip with nothing to remove it only adds
    error. v2 therefore measures the raw window when the residual stem says
    there was little competing sound, and records which treatment was used —
    a corpus whose treatment varies is usable only when the treatment is known.
    """

    @staticmethod
    def _stems(root: Path, vid: str, voice_amp: float, other_amp: float):
        import numpy as np
        import soundfile as sf

        stems = root / "stems_fast"
        stems.mkdir(parents=True, exist_ok=True)
        windows = root / "windows"
        windows.mkdir(parents=True, exist_ok=True)
        t = np.arange(16000) / 16000
        sf.write(stems / f"{vid}_raw90_(vocals)_m.wav",
                 (voice_amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
        sf.write(stems / f"{vid}_raw90_(other)_m.wav",
                 (other_amp * np.sin(2 * np.pi * 90 * t)).astype("float32"), 16000)
        sf.write(windows / f"{vid}_raw90.wav",
                 (voice_amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)

    def test_a_clean_clip_is_measured_raw(self, tmp_path):
        self._stems(tmp_path, "clean", voice_amp=0.5, other_amp=0.005)
        path, separated, ratio = cli._resolve_v2_audio(
            "clean", tmp_path, threshold_db=-15.0
        )
        assert separated is False
        assert path.name == "clean_raw90.wav"
        assert ratio < -15.0

    def test_a_noisy_clip_is_measured_from_the_separated_stem(self, tmp_path):
        self._stems(tmp_path, "noisy", voice_amp=0.3, other_amp=0.3)
        path, separated, ratio = cli._resolve_v2_audio(
            "noisy", tmp_path, threshold_db=-15.0
        )
        assert separated is True
        assert "(vocals)" in path.name
        assert ratio > -15.0

    def test_a_clip_with_no_residual_stem_is_separated_rather_than_assumed_clean(
        self, tmp_path
    ):
        """Absent evidence must not read as evidence of a clean clip."""
        import numpy as np
        import soundfile as sf

        stems = tmp_path / "stems_fast"
        stems.mkdir(parents=True)
        t = np.arange(16000) / 16000
        sf.write(stems / "lonely_raw90_(vocals)_m.wav",
                 (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
        path, separated, _ = cli._resolve_v2_audio(
            "lonely", tmp_path, threshold_db=-15.0
        )
        assert separated is True
        assert "(vocals)" in path.name


class TestExport:
    def test_writes_all_four_tables(self, project):
        out = project / "exports"
        cli.main(
            [
                "export", "--v1",
                "--measurements-dir", str(project / "measurements"),
                "--v2-dir", str(project / "nonexistent"),
                "-o", str(out),
            ]
        )
        for name in ("clips.csv", "talent_month.csv", "talent_summary.csv", "correlations.csv"):
            assert (out / name).is_file(), f"{name} missing"

    def test_clips_table_has_one_row_per_measurement(self, project):
        out = project / "exports"
        cli.main(
            [
                "export", "--v1",
                "--measurements-dir", str(project / "measurements"),
                "--v2-dir", str(project / "nonexistent"),
                "-o", str(out),
            ]
        )
        with (out / "clips.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == 6


class TestAnalyse:
    def test_reports_trends_and_period_residuals(self, project, tmp_path):
        out = tmp_path / "analysis.json"
        cli.main(
            [
                "analyse", "--v1",
                "--measurements-dir", str(project / "measurements"),
                "--v2-dir", str(project / "nonexistent"),
                "--min-months", "3",
                "-o", str(out),
            ]
        )
        report = json.loads(out.read_text(encoding="utf-8"))
        assert "alpha" in report["career_trends"]
        assert "period_residuals" in report

    def test_states_the_identification_limit_in_the_report(self, project, tmp_path):
        """A consumer of this JSON must not read the residuals as a clean
        separation of career from calendar effects, because that separation is
        not identifiable."""
        out = tmp_path / "analysis.json"
        cli.main(
            [
                "analyse", "--v1",
                "--measurements-dir", str(project / "measurements"),
                "--v2-dir", str(project / "nonexistent"),
                "--min-months", "3",
                "-o", str(out),
            ]
        )
        note = json.loads(out.read_text(encoding="utf-8"))["identification_note"].lower()
        assert "not separately identifiable" in note

    def test_is_read_only(self, project, tmp_path):
        source = project / "measurements" / "alpha_monthly.json"
        before = source.read_bytes()
        cli.main(
            [
                "analyse", "--v1",
                "--measurements-dir", str(project / "measurements"),
                "--v2-dir", str(project / "nonexistent"),
                "--min-months", "3",
                "-o", str(tmp_path / "a.json"),
            ]
        )
        assert source.read_bytes() == before
