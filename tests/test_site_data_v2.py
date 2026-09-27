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
import math

import numpy as np

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
    def test_families_in_statistics_md_order(self, registry):
        v2_loader, v1_loader = _loaders({}, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        keys = [f["key"] for f in data["families"]]
        assert keys == [
            "pitch_level",
            "pitch_movement",
            "voice_source",
            "periodicity_noise",
            "spectrum_resonance",
            "tempo",
            "context_quality",
        ]

    def test_voice_source_family_carries_the_airy_vs_full_measures(self, registry):
        """The breathiness/tilt measures are shown, labelled experimental
        until their invariance to separation is established."""
        v2_loader, v1_loader = _loaders({}, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader
        )
        by_key = {f["key"]: f for f in data["families"]}
        assert by_key["voice_source"]["robust"] is False
        assert set(by_key["voice_source"]["metrics"]) == {
            "cpp_db", "h1h2_db", "harmonic_tilt_db_per_octave",
            "alpha_ratio_db", "hammarberg_db",
        }
        assert list(by_key["tempo"]["metrics"]) == ["speaking_rate_syl_per_s"]
        assert "formant_dispersion_hz" in by_key["spectrum_resonance"]["metrics"]

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


class TestOneOff:
    """A one-off data point (a single stream, e.g. a talent speaking another
    language) is shown on its own card but must not reshape the corpus-wide
    rankings everyone else is placed in."""

    def _data(self, registry, with_one_off: bool):
        reg = dict(registry)
        v2_store = {
            "data/measurements/v2/alpha.json": [_v2_rec("a1", "2020-01", 300.0)],
            "data/measurements/v2/beta.json": [_v2_rec("b1", "2020-01", 200.0)],
        }
        if with_one_off:
            reg["data/measurements/gamma_monthly.json"] = "Gamma (English)"
            v2_store["data/measurements/v2/gamma.json"] = [
                {**_v2_rec(f"g@{s}", "2020-01", 400.0), "one_off": True}
                for s in (0, 90)
            ]
        v2_loader, v1_loader = _loaders(v2_store, {p: [] for p in reg})
        return site_data_v2.build_site_data_v2(
            reg, v2_loader=v2_loader, v1_loader=v1_loader
        )

    def test_one_off_does_not_shift_other_talents_percentiles(self, registry):
        before = self._data(registry, with_one_off=False)
        after = self._data(registry, with_one_off=True)
        for name in ("Alpha Talent", "Beta Talent"):
            assert (
                after["talents"][name]["metrics"]["median_f0"]["percentile"]
                == before["talents"][name]["metrics"]["median_f0"]["percentile"]
            )

    def test_one_off_is_placed_among_the_ranked_talents_not_ranked_with_them(
        self, registry
    ):
        """Its percentile says where it would fall among the ranked talents
        (above both here), so its profile snapshot is still readable."""
        data = self._data(registry, with_one_off=True)
        gamma = data["talents"]["Gamma (English)"]
        assert gamma["one_off"] is True
        assert gamma["metrics"]["median_f0"]["percentile"] == 100.0
        assert data["talents"]["Alpha Talent"]["one_off"] is False

    def test_one_off_does_not_enter_the_corpus_noise_floor(self, registry):
        before = self._data(registry, with_one_off=False)
        after = self._data(registry, with_one_off=True)
        assert after["corpus_noise_floor"] == before["corpus_noise_floor"]


class TestNeighbours:
    """Each talent's closest voices, by the measured route and — when local
    embeddings exist — by the embedding route. Only names and scores are
    published; never an embedding."""

    def _store(self):
        def talent(prefix, f0, jitter):
            return [
                _v2_rec(f"{prefix}{i}", f"2020-0{1 + i}", f0 + 3 * i, jitter=jitter + 0.001 * i)
                for i in range(4)
            ]

        return {
            "data/measurements/v2/alpha.json": talent("a", 300.0, 0.010),
            "data/measurements/v2/beta.json": talent("b", 305.0, 0.011),
            "data/measurements/v2/gamma.json": talent("g", 200.0, 0.030),
        }

    def _registry(self):
        return {
            "data/measurements/alpha_monthly.json": "Alpha",
            "data/measurements/beta_monthly.json": "Beta",
            "data/measurements/gamma_monthly.json": "Gamma",
        }

    def _embeddings(self, store):
        from vvc.embed import Embeddings

        direction = {"alpha": [1.0, 0.0, 0.0], "beta": [0.9, 0.1, 0.0], "gamma": [0.0, 0.0, 1.0]}
        out = {}
        for path, records in store.items():
            slug = path.rsplit("/", 1)[1].removesuffix(".json")
            out[slug] = Embeddings(
                tuple(r["id"] for r in records),
                np.array([direction[slug] for _ in records]),
                "fake",
            )
        return out

    def test_measured_neighbours_are_ranked_and_explained(self):
        store = self._store()
        v2_loader, v1_loader = _loaders(store, {p: [] for p in self._registry()})
        data = site_data_v2.build_site_data_v2(
            self._registry(), v2_loader=v2_loader, v1_loader=v1_loader
        )
        near = data["talents"]["Alpha"]["neighbours"]["measured"]
        assert [n["name"] for n in near] == ["Beta", "Gamma"]
        assert 0.0 <= near[0]["closer_than_pct"] <= 100.0
        assert near[0]["closest_metrics"]
        assert "voice" not in data["talents"]["Alpha"]["neighbours"]

    def test_embedding_neighbours_publish_scores_never_vectors(self):
        store = self._store()
        v2_loader, v1_loader = _loaders(store, {p: [] for p in self._registry()})
        data = site_data_v2.build_site_data_v2(
            self._registry(),
            v2_loader=v2_loader,
            v1_loader=v1_loader,
            embeddings=self._embeddings(store),
        )
        near = data["talents"]["Alpha"]["neighbours"]["voice"]
        assert near[0]["name"] == "Beta"
        assert set(near[0]) == {
            "name", "similarity", "match_pct", "closer_than_pct", "one_off", "closest_metrics"
        }
        for talent in data["talents"].values():
            assert not any("embedding" in key or "centroid" in key for key in talent)


class TestNoiseUnits:
    """The comparison radar places each talent on each voice metric as a
    distance from the corpus median in units of within-talent variation —
    real magnitudes, so two talents overlap only when genuinely close."""

    def _data(self, extra=None):
        def talent(prefix, f0):
            return [_v2_rec(f"{prefix}{i}", f"2020-0{1 + i}", f0 + 4 * i) for i in range(4)]

        store = {
            "data/measurements/v2/alpha.json": talent("a", 200.0),
            "data/measurements/v2/beta.json": talent("b", 250.0),
            "data/measurements/v2/gamma.json": talent("g", 300.0),
        }
        registry = {
            "data/measurements/alpha_monthly.json": "Alpha",
            "data/measurements/beta_monthly.json": "Beta",
            "data/measurements/gamma_monthly.json": "Gamma",
        }
        if extra:
            store["data/measurements/v2/delta.json"] = [
                {**_v2_rec(f"d@{i}", "2020-01", 600.0), "one_off": True} for i in range(3)
            ]
            registry["data/measurements/delta_monthly.json"] = "Delta"
        v2_loader, v1_loader = _loaders(store, {p: [] for p in registry})
        return site_data_v2.build_site_data_v2(registry, v2_loader=v2_loader, v1_loader=v1_loader)

    def test_the_median_talent_sits_at_zero_and_the_others_either_side(self):
        data = self._data()
        units = {n: t["metrics"]["median_f0"]["noise_units"] for n, t in data["talents"].items()}
        assert units["Beta"] == pytest.approx(0.0)
        assert units["Alpha"] < 0 < units["Gamma"]

    def test_a_one_off_does_not_move_the_centre(self):
        before = self._data()
        after = self._data(extra=True)
        assert after["talents"]["Beta"]["metrics"]["median_f0"]["noise_units"] == pytest.approx(
            before["talents"]["Beta"]["metrics"]["median_f0"]["noise_units"]
        )
        assert after["talents"]["Delta"]["metrics"]["median_f0"]["noise_units"] > 0

    def test_radar_axes_are_voice_metrics_only(self):
        data = self._data()
        assert "median_f0" in data["radar_metrics"]
        assert not {"voiced_fraction", "loudness_dynamics_db", "background_ratio_db"} & set(
            data["radar_metrics"]
        )


class TestLanguageCompensation:
    """A speaker model hears language as well as voice: left alone, every
    Japanese-speaking talent's closest voices are Japanese speakers. Each
    language group's mean embedding is removed before comparing, so what is
    left is the voice."""

    def _build(self):
        from vvc.embed import Embeddings

        # Language dominates the raw vectors; the voice is a small second axis.
        spec = {
            "jp_airy": ("Hololive JP Talent", [1.0, 0.0, 0.3]),
            "jp_full": ("Hololive JP Talent", [1.0, 0.0, -0.3]),
            "en_airy": ("Hololive EN Talent", [0.0, 1.0, 0.3]),
            "en_full": ("Hololive EN Talent", [0.0, 1.0, -0.3]),
        }
        store, registry, embeddings = {}, {}, {}
        for slug, (_, vec) in spec.items():
            recs = [_v2_rec(f"{slug}{i}", f"2020-0{1 + i}", 300.0 + i) for i in range(3)]
            store[f"data/measurements/v2/{slug}.json"] = recs
            registry[f"data/measurements/{slug}_monthly.json"] = slug
            embeddings[slug] = Embeddings(tuple(r["id"] for r in recs), np.array([vec] * 3), "fake")
        v2_loader, v1_loader = _loaders(store, {p: [] for p in registry})
        return site_data_v2.build_site_data_v2(
            registry,
            v2_loader=v2_loader,
            v1_loader=v1_loader,
            embeddings=embeddings,
            language_of={"jp_airy": "JP", "jp_full": "JP", "en_airy": "EN", "en_full": "EN"},
        )

    def test_the_closest_voice_crosses_languages_when_the_voice_matches(self):
        data = self._build()
        assert data["talents"]["jp_airy"]["neighbours"]["voice"][0]["name"] == "en_airy"
        assert data["talents"]["en_full"]["neighbours"]["voice"][0]["name"] == "jp_full"


def test_dev_is_speaks_japanese_for_compensation():
    assert site_data_v2.language_group("DEV_IS") == "JP"
    assert site_data_v2.language_group("EN") == "EN"
    assert site_data_v2.language_group("Graduated") is None


def test_a_one_off_can_declare_its_own_language():
    records = [{"one_off": True, "language": "en"}, {"one_off": True, "language": "en"}]
    assert site_data_v2.declared_language(records) == "EN"
    assert site_data_v2.declared_language([{"id": "x"}]) is None



class TestVoiceMatchExport:
    def test_neighbours_carry_an_absolute_match_on_both_routes(self):
        t = TestNeighbours()
        store = t._store()
        v2_loader, v1_loader = _loaders(store, {p: [] for p in t._registry()})
        data = site_data_v2.build_site_data_v2(
            t._registry(),
            v2_loader=v2_loader,
            v1_loader=v1_loader,
            embeddings=t._embeddings(store),
        )
        voice = data["talents"]["Alpha"]["neighbours"]["voice"]
        by_name = {r["name"]: r for r in voice}
        # Beta's voice points almost the same way as Alpha's; Gamma's does not.
        assert by_name["Beta"]["match_pct"] > 90
        assert by_name["Gamma"]["match_pct"] < 10
        for row in data["talents"]["Alpha"]["neighbours"]["measured"]:
            assert 0.0 <= row["match_pct"] <= 100.0 or math.isnan(row["match_pct"])
        scale = data["similarity_scale"]
        assert set(scale) == {"voice", "measured"}
        assert set(scale["voice"]) == {"zero", "full"}

    def test_a_one_off_shows_up_in_the_lists_of_the_talents_it_matches(self):
        from vvc.embed import Embeddings

        t = TestNeighbours()
        store = t._store()
        registry = t._registry()
        store["data/measurements/v2/delta.json"] = [
            {**_v2_rec(f"d@{i}", "2020-01", 301.0), "one_off": True} for i in range(3)
        ]
        registry["data/measurements/delta_monthly.json"] = "Delta"
        embeddings = t._embeddings({k: v for k, v in store.items() if "delta" not in k})
        embeddings["delta"] = Embeddings(
            tuple(r["id"] for r in store["data/measurements/v2/delta.json"]),
            np.array([[1.0, 0.0, 0.0]] * 3),
            "fake",
        )
        v2_loader, v1_loader = _loaders(store, {p: [] for p in registry})
        data = site_data_v2.build_site_data_v2(
            registry, v2_loader=v2_loader, v1_loader=v1_loader, embeddings=embeddings
        )
        alpha = {r["name"]: r for r in data["talents"]["Alpha"]["neighbours"]["voice"]}
        assert alpha["Delta"]["one_off"] is True
        assert alpha["Beta"]["one_off"] is False
