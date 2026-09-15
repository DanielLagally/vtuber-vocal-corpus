"""Product tests for the v2 site's data export (site_data_v2.py).

Synthetic data only — fabricated measurement dicts, never Cover/hololive
audio. User-visible rules:

1. build_site_data_v2(registry, ...) emits exactly the 5 families from
   reference/statistics.md's "Presenting fewer things than there are
   columns" section, in that order, each carrying its member metric keys
   and a robust flag/role matching reference/measurement.md's validity
   table.
2. Every talent's per-metric entry carries typical/ci/trend/percentile/
   noise_floor plus monthly/quarterly/yearly series — built by series_v2,
   never by v1's series.py (mean at month/quarter, median_low at year).
3. Percentile requires >=2 talents with a finite typical value on that
   metric; with fewer, the metric is simply absent from that talent's
   percentile, never a fabricated 50.0 or a crash.
4. A talent whose v2 file is missing entirely (build-v2 never ran for
   them) still appears, built from their v1 records directly, rather
   than silently vanishing from the site.
5. write_site_data_v2 writes ``window.SITE_DATA_V2 = {...};``, matching
   v1's script-tag convention (no fetch, no CORS issue on file://).

Talent colors are deliberately NOT assigned here: v1's app.js already has a
stable, deterministic color assignment (a curated brand-color map plus an
alphabetically-sorted fallback palette) and v2's app.js reuses that same
logic client-side rather than duplicating brand-color data into Python.
"""

from __future__ import annotations

import json

import pytest

from vvc import site_data_v2


def _v2_rec(vid, month, f0, *, passed=True, jitter=0.01):
    return {
        "id": vid,
        "month": month,
        "features": {
            "median_f0": f0,
            "f0_iqr_semitones": 4.0,
            "voiced_fraction": 0.5,
            "fraction_below_160hz": 0.01,
            "jitter_local": jitter,
            "shimmer_local": 0.05,
            "hnr_db": 15.0,
            "brightness_hz": 2000.0,
            "dynamism_semitones": 1.0,
            "loudness_dynamics_db": 6.0,
            "f1_hz": 700.0,
            "f2_hz": 1500.0,
            "f3_hz": 2600.0,
            "f4_hz": 3600.0,
            "background_ratio_db": -20.0,
        },
        "qc": {"pass": passed, "reason": None if passed else "f0_spread"},
        "legacy": False,
    }


def _v1_rec(vid, month, f0):
    return {
        "id": vid,
        "month": month,
        "features": {"median_f0": f0, "f0_iqr": 80.0, "voiced_fraction": 0.5},
        "qc": {"pass": True, "reason": None},
    }


@pytest.fixture
def registry():
    return {
        "data/measurements/alpha_monthly.json": "Alpha Talent",
        "data/measurements/beta_monthly.json": "Beta Talent",
    }


def _loaders(v2_store, v1_store):
    def v2_loader(path):
        if path not in v2_store:
            raise FileNotFoundError(path)
        return v2_store[path]

    def v1_loader(path):
        return v1_store[path]

    return v2_loader, v1_loader


class TestFamilies:
    def test_five_families_in_statistics_md_order(self, registry):
        v2_loader, v1_loader = _loaders({}, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        keys = [f["key"] for f in data["families"]]
        assert keys == [
            "pitch_level",
            "pitch_movement",
            "periodicity_noise",
            "spectrum_resonance",
            "context_quality",
        ]

    def test_experimental_families_are_flagged_not_robust(self, registry):
        v2_loader, v1_loader = _loaders({}, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        by_key = {f["key"]: f for f in data["families"]}
        assert by_key["pitch_level"]["robust"] is True
        assert by_key["periodicity_noise"]["robust"] is False
        assert by_key["spectrum_resonance"]["robust"] is False
        assert "hnr_db" in by_key["periodicity_noise"]["metrics"]
        assert set(by_key["spectrum_resonance"]["metrics"]) >= {
            "brightness_hz", "f1_hz", "f2_hz", "f3_hz", "f4_hz",
        }


class TestPerTalentMetrics:
    def test_metric_entry_carries_series_trend_and_noise_floor(self, registry):
        v2_store = {
            "data/measurements/v2/alpha.json": [
                _v2_rec("a1", "2020-01", 300.0),
                _v2_rec("a2", "2020-02", 310.0),
                _v2_rec("a3", "2020-03", 320.0),
            ],
            "data/measurements/v2/beta.json": [
                _v2_rec("b1", "2020-01", 250.0),
                _v2_rec("b2", "2020-01", 254.0),
            ],
        }
        v2_loader, v1_loader = _loaders(v2_store, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        alpha = data["talents"]["Alpha Talent"]["metrics"]["median_f0"]
        assert alpha["typical"] == pytest.approx(310.0)
        assert alpha["trend"]["n"] == 3
        assert len(alpha["monthly"]) == 3
        assert len(alpha["yearly"]) == 1
        assert alpha["noise_floor"]["n_pairs"] == 0  # no same-month pairs for alpha

        beta = data["talents"]["Beta Talent"]["metrics"]["median_f0"]
        assert beta["noise_floor"]["n_pairs"] == 1

    def test_percentile_needs_at_least_two_comparable_talents(self, registry):
        v2_store = {
            "data/measurements/v2/alpha.json": [_v2_rec("a1", "2020-01", 300.0)],
            "data/measurements/v2/beta.json": [],
        }
        v2_loader, v1_loader = _loaders(v2_store, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        assert "percentile" not in data["talents"]["Alpha Talent"]["metrics"]["median_f0"]

    def test_percentile_present_with_two_comparable_talents(self, registry):
        v2_store = {
            "data/measurements/v2/alpha.json": [_v2_rec("a1", "2020-01", 300.0)],
            "data/measurements/v2/beta.json": [_v2_rec("b1", "2020-01", 200.0)],
        }
        v2_loader, v1_loader = _loaders(v2_store, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        assert data["talents"]["Alpha Talent"]["metrics"]["median_f0"]["percentile"] == 100.0
        assert data["talents"]["Beta Talent"]["metrics"]["median_f0"]["percentile"] == 0.0


class TestLegacyFallback:
    def test_a_talent_missing_its_v2_file_still_appears_via_v1(self, registry):
        v2_store = {
            "data/measurements/v2/alpha.json": [_v2_rec("a1", "2020-01", 300.0)],
            # beta's v2 file is entirely missing — build-v2 never ran for them
        }
        v1_store = {
            "data/measurements/alpha_monthly.json": [],
            "data/measurements/beta_monthly.json": [_v1_rec("b1", "2020-01", 250.0)],
        }
        v2_loader, v1_loader = _loaders(v2_store, v1_store)
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        assert "Beta Talent" in data["talents"]
        assert data["talents"]["Beta Talent"]["metrics"]["median_f0"]["typical"] == pytest.approx(250.0)
        assert data["talents"]["Beta Talent"]["legacy_fallback"] is True
        assert data["talents"]["Alpha Talent"]["legacy_fallback"] is False


class TestWriteSiteDataV2:
    def test_writes_a_window_assignment_not_bare_json(self, tmp_path, registry):
        v2_loader, v1_loader = _loaders({}, {p: [] for p in registry})
        out = tmp_path / "data.js"
        site_data_v2.write_site_data_v2(
            registry, out, v2_loader=v2_loader, v1_loader=v1_loader
        )
        text = out.read_text(encoding="utf-8")
        assert text.startswith("window.SITE_DATA_V2 = ")
        assert text.rstrip().endswith(";")
        payload = json.loads(text[len("window.SITE_DATA_V2 = ") : text.rindex(";")])
        assert "talents" in payload
