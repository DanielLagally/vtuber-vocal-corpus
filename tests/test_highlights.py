"""Product tests for the Highlights page's data (highlights.py, exported by
site_data_v2 under ``highlights``).

Synthetic data only. User-visible rules:

1. Each end of a scale is its own positively named award, ranked by the
   talent's typical value.
2. Only talents with enough data place: one-offs, v1-only talents, and talents
   under the clip/month threshold never appear and never shift a ranking.
3. A podium place whose confidence interval overlaps the place above it is
   marked too close to call.
4. Every eligible talent gets exactly one signature: the award where they
   rank closest to the top, robust measures winning ties.
5. Voice-identity awards publish names and scores only, each pair once, and
   one-offs never take part.
"""

from __future__ import annotations

import numpy as np
import pytest

from vvc import site_data_v2

N_CLIPS = 24
N_MONTHS = 8


def _clip(vid, month, f0, **features):
    base = {
        "median_f0": f0,
        "f0_iqr_semitones": 4.0,
        "voiced_fraction": 0.5,
        "jitter_local": 0.01,
        "shimmer_local": 0.05,
        "hnr_db": 15.0,
        "brightness_hz": 2000.0,
        "dynamism_semitones": 0.3,
        "loudness_dynamics_db": 12.0,
        "f1_hz": 700.0,
        "f2_hz": 1500.0,
        "f3_hz": 2600.0,
        "f4_hz": 3600.0,
    }
    base.update(features)
    return {
        "id": vid,
        "month": month,
        "features": base,
        "qc": {"pass": True, "reason": None},
        "legacy": False,
    }


def _talent(prefix, f0, *, n=N_CLIPS, months=N_MONTHS, spread=0.0, slope=0.0, **features):
    """``n`` clips dealt across ``months`` consecutive months. ``spread``
    alternates each month's pitch above and below ``f0``; ``slope`` adds that
    many Hz per month."""
    out = []
    for i in range(n):
        m = i % months
        wobble = spread if m % 2 else -spread
        out.append(
            _clip(f"{prefix}{i:03d}", f"{2020 + m // 12}-{m % 12 + 1:02d}", f0 + wobble + slope * m, **features)
        )
    return out


def _build(talents: dict[str, list[dict]], *, embeddings=None, roster=None, v1_only=()):
    store, registry = {}, {}
    for name, records in talents.items():
        slug = name.lower().replace(" ", "_")
        registry[f"data/measurements/{slug}_monthly.json"] = name
        if name not in v1_only:
            store[f"data/measurements/v2/{slug}.json"] = records
    v1_store = {
        p: [
            {"id": f"v1{i}", "month": f"2020-{i + 1:02d}", "features": {"median_f0": 900.0},
             "qc": {"pass": True, "reason": None}}
            for i in range(N_MONTHS)
        ]
        for p in registry
    }

    def v2_loader(path):
        if path not in store:
            raise FileNotFoundError(path)
        return store[path]

    return site_data_v2.build_site_data_v2(
        registry,
        v2_loader=v2_loader,
        v1_loader=lambda p: v1_store[p],
        embeddings=embeddings,
        roster=roster,
    )


def _award(data, key):
    return next(a for a in data["highlights"]["awards"] if a["key"] == key)


def _names(award):
    return [row["name"] for row in award["ranking"]]


FOUR = {
    "Alpha": _talent("a", 200.0),
    "Beta": _talent("b", 250.0),
    "Gamma": _talent("g", 300.0),
    "Delta": _talent("d", 350.0),
}


class TestRanking:
    def test_each_end_of_a_scale_is_its_own_award_ranked_by_typical(self):
        data = _build(FOUR)
        assert _names(_award(data, "highest_voice")) == ["Delta", "Gamma", "Beta", "Alpha"]
        assert _names(_award(data, "deepest_voice")) == ["Alpha", "Beta", "Gamma", "Delta"]

    def test_pitch_comparison_is_in_semitones_from_the_typical_talent(self):
        data = _build(FOUR)
        top = _award(data, "highest_voice")["ranking"][0]
        assert top["value"] == pytest.approx(350.0)
        # The typical eligible talent sits at 275 Hz.
        assert top["delta"] == pytest.approx(12 * np.log2(350.0 / 275.0))
        assert _award(data, "highest_voice")["delta_unit"] == "semitones"

    def test_only_talents_with_enough_data_place_and_others_shift_nothing(self):
        before = _build(FOUR)
        crowded = dict(FOUR)
        crowded["Tiny"] = _talent("t", 600.0, n=3, months=3)
        crowded["Short"] = _talent("s", 600.0, months=2)
        crowded["Once"] = [{**c, "one_off": True} for c in _talent("o", 600.0)]
        crowded["Old"] = _talent("x", 600.0)
        after = _build(crowded, v1_only=("Old",))
        for award in after["highlights"]["awards"]:
            assert not {"Tiny", "Short", "Once", "Old"} & set(_names(award))
        assert _award(after, "highest_voice")["ranking"] == _award(before, "highest_voice")["ranking"]
        assert set(after["highlights"]["signatures"]) == set(FOUR)

    def test_overlapping_confidence_is_too_close_to_call(self):
        data = _build(
            {
                "Alpha": _talent("a", 350.0),
                "Beta": _talent("b", 300.0, spread=40.0),
                "Gamma": _talent("g", 302.0, spread=40.0),
            }
        )
        rows = _award(data, "highest_voice")["ranking"]
        assert [r["name"] for r in rows] == ["Alpha", "Gamma", "Beta"]
        assert rows[0]["too_close"] is False
        assert rows[1]["too_close"] is False
        assert rows[2]["too_close"] is True


class TestSignatures:
    def test_each_talent_is_signed_by_the_award_they_rank_highest_in(self):
        data = _build(
            {
                # Deepest pitch and warmest tone: a tie, and the robust measure wins.
                "Alpha": _talent("a", 200.0, brightness_hz=1500.0),
                "Beta": _talent("b", 250.0, brightness_hz=3000.0),
                "Gamma": _talent("g", 300.0),
                "Delta": _talent("d", 350.0),
            }
        )
        sig = data["highlights"]["signatures"]
        assert sig["Alpha"]["award"] == "deepest_voice"
        assert sig["Beta"]["award"] == "most_sparkling"
        assert sig["Delta"]["award"] == "highest_voice"
        assert sig["Delta"]["rank"] == 1
        assert sig["Delta"]["of"] == 4
        assert set(sig) == set(FOUR)


class TestThroughTheYears:
    def test_a_pitch_journey_compares_the_first_and_latest_year(self):
        months = 30
        data = _build(
            {
                "Alpha": _talent("a", 300.0, n=60, months=months, slope=-2.0),
                "Beta": _talent("b", 250.0, n=60, months=months),
                "Gamma": _talent("g", 280.0, n=60, months=months),
            }
        )
        journey = _award(data, "pitch_journey")
        # A flat career is no journey at all.
        assert _names(journey) == ["Alpha"]
        row = journey["ranking"][0]
        # First twelve months centre on 289 Hz, the latest twelve on 253 Hz.
        assert row["from_hz"] == pytest.approx(289.0)
        assert row["to_hz"] == pytest.approx(253.0)
        assert row["value"] == pytest.approx(12 * np.log2(253.0 / 289.0))

    def test_a_short_career_has_no_journey(self):
        data = _build(FOUR)
        assert "pitch_journey" not in {a["key"] for a in data["highlights"]["awards"]}

    def test_longest_record_counts_months(self):
        data = _build({"Alpha": _talent("a", 300.0, months=12), **{k: FOUR[k] for k in ("Beta", "Gamma")}})
        assert _names(_award(data, "longest_record"))[0] == "Alpha"


def _embeddings(talents, direction):
    """One fixed vector per talent."""
    from vvc.embed import Embeddings

    out = {}
    for name, records in talents.items():
        slug = name.lower().replace(" ", "_")
        vectors = np.array([direction[name]] * len(records), dtype=float)
        out[slug] = Embeddings(tuple(r["id"] for r in records), vectors, "fake")
    return out


class TestVoiceIdentity:
    TALENTS = {
        "Alpha": _talent("a", 300.0),
        "Beta": _talent("b", 305.0),
        "Gamma": _talent("g", 200.0),
        "Delta": _talent("d", 250.0),
    }
    DIRECTION = {
        "Alpha": [1.0, 0.0, 0.0, 0.0],
        "Beta": [0.95, 0.05, 0.0, 0.0],
        "Gamma": [0.0, 0.0, 1.0, 0.0],
        "Delta": [0.0, 1.0, 0.0, 0.0],
    }

    def _data(self, extra=None):
        talents = dict(self.TALENTS)
        direction = dict(self.DIRECTION)
        if extra:
            talents["Once"] = [{**c, "one_off": True} for c in _talent("o", 300.0)]
            direction["Once"] = [1.0, 0.0, 0.0, 0.0]
        return _build(talents, embeddings=_embeddings(talents, direction))

    def test_voice_twins_are_each_pair_once_by_match(self):
        twins = _award(self._data(), "voice_twins")["ranking"]
        assert set(twins[0]["members"]) == {"Alpha", "Beta"}
        pairs = [frozenset(t["members"]) for t in twins]
        assert len(pairs) == len(set(pairs))
        assert set(twins[0]) == {"name", "members", "value", "delta", "too_close"}

    def test_one_of_a_kind_is_the_voice_furthest_from_everyone(self):
        unique = _names(_award(self._data(), "one_of_a_kind"))
        assert unique[-1] in {"Alpha", "Beta"}
        assert unique[0] in {"Gamma", "Delta"}

    def test_one_offs_take_no_part(self):
        data = self._data(extra=True)
        for key in ("voice_twins", "one_of_a_kind"):
            for row in _award(data, key)["ranking"]:
                assert "Once" not in row.get("members", [row["name"]])

    def test_generations_are_ranked_by_how_alike_their_members_sound(self):
        talents = {f"{g}{i}": _talent(f"{g}{i}_", 250.0 + 10 * i) for g in "AB" for i in range(3)}
        direction = {
            **{f"A{i}": [1.0, 0.1 * i, 0.0, 0.0] for i in range(3)},
            **{f"B{i}": [0.0, 0.0, 0.0, 0.0] for i in range(3)},
        }
        direction["B0"], direction["B1"], direction["B2"] = [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]
        roster = [
            {"english_name": n, "group": "1st Generation" if n[0] == "A" else "2nd Generation"}
            for n in talents
        ]
        data = _build(talents, embeddings=_embeddings(talents, direction), roster=roster)
        assert _names(_award(data, "most_harmonious_generation"))[0] == "1st Generation"
        assert _names(_award(data, "most_varied_generation"))[0] == "2nd Generation"
        # Scored as the share of all talent pairs less alike than the average
        # member pair, so the less alike generation is not pinned at zero.
        harmonious = _award(data, "most_harmonious_generation")["ranking"]
        assert 0.0 < harmonious[1]["value"] < harmonious[0]["value"] <= 100.0
