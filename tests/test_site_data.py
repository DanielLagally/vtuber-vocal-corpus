"""Product tests for the interactive-site data export (site_data.py).

User-visible rules (synthetic data ONLY — fabricated measurement dicts;
never Cover/hololive audio, never downloads):

1. build_site_data(registry, loader=...) returns display names exactly as
   given in the registry, one "talents" entry per registry row, each
   carrying monthly_f0_all/monthly_f0_qc/quarterly_f0/yearly/qc_summary —
   all built from vvc.series's existing aggregation functions
   (this module does no aggregation math of its own beyond cute_mature).
2. yearly-key gating: a feature key (of the 7 candidates: median_f0,
   brightness_hz, dynamism_semitones, jitter_local, shimmer_local,
   hnr_db, loudness_dynamics_db) only appears under "yearly" for ANY
   talent if at least one record ANYWHERE in the registry has that key —
   a corpus with only median_f0 never emits an empty brightness_hz list.
3. cute_mature holds one point per talent that has at least one QC-pass
   clip: f0_mean/brightness_mean/dynamism_mean (plain means of that
   talent's QC-pass clips) plus a percentile — the equal-weight average
   of the z-scores of those three means across every included talent,
   turned into a percentile by rank (0 = lowest, 100 = highest) among
   included talents. A talent with a corpus_sd of 0 on some axis (all
   talents tied) gets a z-score of 0 on that axis, not a division error.
4. A talent with zero QC-pass clips is omitted from cute_mature entirely
   (never plotted at a fabricated (0, 0)), and does not influence the
   corpus mean/sd used to score the remaining talents.
5. With fewer than 2 talents holding QC-pass clips, cute_mature is empty
   (nothing to compare against).
6. write_site_data writes a ``window.SITE_DATA = {...};`` JS assignment,
   not bare JSON — a plain ``fetch("data.json")`` from a page opened via
   ``file://`` is blocked by CORS in every major browser (confirmed via
   headless Chromium against an earlier draft of this page), so the page
   must not depend on fetching a separate JSON file at all.
7. build_site_data(registry, roster=..., loader=...) attaches "group" (the
   exact Holodex generation/unit label, e.g. "4th Generation (holoForce)")
   and "branch" (a coarser EN/ID/DEV_IS/JP/Graduated bucket derived from
   that group string) to every talent entry, matched by english_name
   against the roster list. A handful of DEV_IS talents are registered
   under short given names (e.g. "Niko") that don't equal roster's full
   english_name ("Koganei Niko") — those go through a small known-alias
   table. A talent absent from the roster entirely (graduated members,
   who Holodex's active-only roster fetch excludes) gets group="Graduated"
   branch="Graduated" rather than silently vanishing from every
   generation/branch filtered view.
8. Branch bucketing from the group string: "English -*-" -> EN,
   "Indonesia *" -> ID, "DEV_IS *" -> DEV_IS, anything else present in the
   roster (JP numbered generations, GAMERS, mekPark) -> JP.
9. Without a roster argument (backward compatible — existing callers/tests
   above don't pass one), every talent still gets group="Unknown"
   branch="Unknown" rather than a missing key, so frontend filter code
   never has to special-case an absent field.
10. Every talent's "percentiles" dict carries a rank percentile (0-100,
    same population z-score-then-rank as cute_mature, but computed
    per-axis independently rather than combined) for each yearly feature
    key that has QC-pass data from at least 2 talents — one percentile
    per metric, not just the 3 cute/mature axes. A talent with zero
    QC-pass data on a given axis is omitted from that axis's percentiles
    (not a fabricated value).
11. An axis with fewer than 2 contributing talents (nothing to compare
    against) is entirely absent from every talent's percentiles dict —
    mirrors cute_mature's own "n<2 -> empty" rule, applied per-axis.

Entry shape matches vvc.measure's persisted record: id / month /
score / window / features{median_f0, f0_iqr, voiced_fraction,
brightness_hz, dynamism_semitones, jitter_local, shimmer_local, hnr_db,
loudness_dynamics_db} / qc{pass, reason} / model / tracker.
"""

from __future__ import annotations

import json

from vvc import site_data


# ---------------------------------------------------------------- helpers


def _entry(
    month: str,
    vid: str,
    *,
    median_f0: float,
    f0_iqr: float = 50.0,
    brightness_hz: float | None = None,
    dynamism_semitones: float | None = None,
    score: float = 70.0,
) -> dict:
    qc_pass = median_f0 == median_f0 and f0_iqr < 200.0 and median_f0 < 600.0
    features: dict = {
        "median_f0": median_f0,
        "f0_iqr": f0_iqr,
        "voiced_fraction": 0.6,
    }
    if brightness_hz is not None:
        features["brightness_hz"] = brightness_hz
    if dynamism_semitones is not None:
        features["dynamism_semitones"] = dynamism_semitones
    return {
        "id": vid,
        "month": month,
        "score": score,
        "window": {"start_s": 0.0, "end_s": 90.0},
        "features": features,
        "qc": {"pass": qc_pass, "reason": None if qc_pass else "f0_iqr"},
        "model": "bs_roformer_vocals_resurrection_unwa.ckpt",
        "tracker": "praat_ac",
    }


def _loader(store: dict[str, list[dict]]):
    def load(path: str) -> list[dict]:
        return store[path]

    return load


# ------------------------------------------------------------------- tests


def test_talent_display_names_and_series_pass_through() -> None:
    """Rule 1: registry display names come through unchanged, and the
    per-talent series are exactly what series.py's own functions return."""
    store = {
        "a.json": [
            _entry("2024-01", "vid0000001", median_f0=200.0),
            _entry("2024-02", "vid0000002", median_f0=210.0),
        ],
    }
    result = site_data.build_site_data({"a.json": "Talent A"}, loader=_loader(store))
    assert set(result["talents"]) == {"Talent A"}
    talent = result["talents"]["Talent A"]
    assert talent["monthly_f0_qc"] == [("2024-01", 200.0), ("2024-02", 210.0)]
    assert talent["qc_summary"] == {"qc_pass": 2, "total": 2}


def test_yearly_feature_keys_gated_on_registry_wide_presence() -> None:
    """Rule 2: brightness_hz is absent from EVERY talent's yearly dict
    when no record in the whole registry carries it; present (even as an
    empty list for the talent that lacks it) once ANY talent does."""
    store = {
        "a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)],
        "b.json": [_entry("2024-01", "vid0000002", median_f0=200.0)],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store)
    )
    assert "brightness_hz" not in result["talents"]["A"]["yearly"]
    assert "brightness_hz" not in result["talents"]["B"]["yearly"]
    assert "median_f0" in result["talents"]["A"]["yearly"]

    store2 = {
        "a.json": [_entry("2024-01", "vid0000001", median_f0=200.0, brightness_hz=1500.0)],
        "b.json": [_entry("2024-01", "vid0000002", median_f0=200.0)],
    }
    result2 = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store2)
    )
    assert "brightness_hz" in result2["talents"]["A"]["yearly"]
    assert "brightness_hz" in result2["talents"]["B"]["yearly"]
    assert result2["talents"]["B"]["yearly"]["brightness_hz"] == []


def test_cute_mature_percentile_hand_computed_three_talents() -> None:
    """Rule 3: a hand-verified 3-talent example. Talent-level means:
    A=(f0=210, bright=1100, dyn=1.0), B=(300, 1500, 2.0), C=(250, 1300, 1.5).
    Population z-scores per axis put A lowest, C middle, B highest on
    ALL THREE axes simultaneously, so the combined (equal-weight average)
    z-score ranks identically: A < C < B — giving clean rank percentiles
    0.0 / 50.0 / 100.0 with n=3 (rank / (n-1) * 100)."""
    store = {
        "a.json": [
            _entry("2024-01", "vidA0000001", median_f0=200.0, brightness_hz=1000.0, dynamism_semitones=1.0),
            _entry("2024-02", "vidA0000002", median_f0=220.0, brightness_hz=1200.0, dynamism_semitones=1.0),
        ],
        "b.json": [
            _entry("2024-01", "vidB0000001", median_f0=300.0, brightness_hz=1500.0, dynamism_semitones=2.0),
        ],
        "c.json": [
            _entry("2024-01", "vidC0000001", median_f0=250.0, brightness_hz=1300.0, dynamism_semitones=1.5),
        ],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B", "c.json": "C"}, loader=_loader(store)
    )
    cute = result["cute_mature"]
    assert set(cute) == {"A", "B", "C"}
    assert cute["A"]["f0_mean"] == 210.0
    assert cute["A"]["brightness_mean"] == 1100.0
    assert cute["A"]["dynamism_mean"] == 1.0
    assert cute["A"]["percentile"] == 0.0
    assert cute["C"]["percentile"] == 50.0
    assert cute["B"]["percentile"] == 100.0


def test_cute_mature_omits_talent_with_zero_qc_pass_clips() -> None:
    """Rules 4-5: an all-QC-fail talent never appears in cute_mature, and
    is excluded from the corpus stats used to score the survivors — with
    only one QC-passing talent left, cute_mature is empty (rule 5)."""
    store = {
        "a.json": [
            _entry("2024-01", "vidA0000001", median_f0=200.0, f0_iqr=250.0, brightness_hz=1000.0, dynamism_semitones=1.0),
        ],
        "b.json": [
            _entry("2024-01", "vidB0000001", median_f0=300.0, brightness_hz=1500.0, dynamism_semitones=2.0),
        ],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store)
    )
    assert result["cute_mature"] == {}


def test_cute_mature_zero_corpus_sd_gives_zero_zscore_not_a_crash() -> None:
    """Rule 3: two talents tied on every axis must not divide by zero —
    both z-score 0 on every axis, so combined z is 0 for both. With no
    discriminating signal at all (every included talent identical), both
    land at the midpoint, 50.0, rather than an arbitrary tie-break."""
    store = {
        "a.json": [
            _entry("2024-01", "vidA0000001", median_f0=250.0, brightness_hz=1300.0, dynamism_semitones=1.5),
        ],
        "b.json": [
            _entry("2024-01", "vidB0000001", median_f0=250.0, brightness_hz=1300.0, dynamism_semitones=1.5),
        ],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store)
    )
    assert result["cute_mature"]["A"]["percentile"] == 50.0
    assert result["cute_mature"]["B"]["percentile"] == 50.0


def _roster_entry(english_name: str, group: str) -> dict:
    return {"id": f"UC{english_name}", "english_name": english_name, "group": group}


def test_group_and_branch_attached_via_roster_english_name_match() -> None:
    """Rule 7: a direct english_name match picks up group verbatim, and
    rule 8: a JP numbered-generation group buckets to branch 'JP'."""
    store = {"a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)]}
    roster = [_roster_entry("Himemori Luna", "4th Generation (holoForce)")]
    result = site_data.build_site_data(
        {"a.json": "Himemori Luna"}, roster=roster, loader=_loader(store)
    )
    talent = result["talents"]["Himemori Luna"]
    assert talent["group"] == "4th Generation (holoForce)"
    assert talent["branch"] == "JP"


def test_dev_is_short_name_resolves_via_alias_table() -> None:
    """Rule 7: 'Niko' (the registry display name) has no english_name
    match on its own — it resolves via the known DEV_IS alias to roster's
    'Koganei Niko' — and rule 8: a DEV_IS group buckets to branch
    'DEV_IS'."""
    store = {"a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)]}
    roster = [_roster_entry("Koganei Niko", "DEV_IS FLOW GLOW")]
    result = site_data.build_site_data(
        {"a.json": "Niko"}, roster=roster, loader=_loader(store)
    )
    talent = result["talents"]["Niko"]
    assert talent["group"] == "DEV_IS FLOW GLOW"
    assert talent["branch"] == "DEV_IS"


def test_en_and_id_branches_derived_from_group_string() -> None:
    """Rule 8: English -*- and Indonesia * groups bucket to EN / ID."""
    store = {
        "a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)],
        "b.json": [_entry("2024-01", "vid0000002", median_f0=200.0)],
    }
    roster = [
        _roster_entry("Mori Calliope", "English -Myth-"),
        _roster_entry("Kaela Kovalskia", "Indonesia 3rd Gen (holoh3ro)"),
    ]
    result = site_data.build_site_data(
        {"a.json": "Mori Calliope", "b.json": "Kaela Kovalskia"},
        roster=roster,
        loader=_loader(store),
    )
    assert result["talents"]["Mori Calliope"]["branch"] == "EN"
    assert result["talents"]["Kaela Kovalskia"]["branch"] == "ID"


def test_talent_absent_from_roster_gets_graduated_bucket() -> None:
    """Rule 7: a graduated talent (Holodex's active-only roster fetch
    excludes them) gets group=branch='Graduated' instead of vanishing
    from filtered views."""
    store = {"a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)]}
    roster: list[dict] = [_roster_entry("Someone Else", "5th Generation (holoFive)")]
    result = site_data.build_site_data(
        {"a.json": "Kiryu Coco"}, roster=roster, loader=_loader(store)
    )
    talent = result["talents"]["Kiryu Coco"]
    assert talent["group"] == "Graduated"
    assert talent["branch"] == "Graduated"


def test_no_roster_argument_defaults_group_branch_to_unknown() -> None:
    """Rule 9: existing callers that never pass roster= (all tests above
    this one) still get group/branch keys, defaulted to 'Unknown' rather
    than omitted, so frontend filter code never special-cases a missing
    field."""
    store = {"a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)]}
    result = site_data.build_site_data({"a.json": "A"}, loader=_loader(store))
    assert result["talents"]["A"]["group"] == "Unknown"
    assert result["talents"]["A"]["branch"] == "Unknown"


def test_percentiles_computed_independently_per_yearly_axis() -> None:
    """Rule 10: median_f0 percentiles rank A < C < B (200/250/300), but
    brightness percentiles rank B < A < C (1500/1000/1300) — independent
    per-axis rankings, not the combined cute_mature order."""
    store = {
        "a.json": [
            _entry("2024-01", "vidA0000001", median_f0=200.0, brightness_hz=1000.0),
        ],
        "b.json": [
            _entry("2024-01", "vidB0000001", median_f0=300.0, brightness_hz=1500.0),
        ],
        "c.json": [
            _entry("2024-01", "vidC0000001", median_f0=250.0, brightness_hz=1300.0),
        ],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B", "c.json": "C"}, loader=_loader(store)
    )
    pct = {name: result["talents"][name]["percentiles"] for name in ("A", "B", "C")}
    assert pct["A"]["median_f0"] == 0.0
    assert pct["C"]["median_f0"] == 50.0
    assert pct["B"]["median_f0"] == 100.0
    assert pct["A"]["brightness_hz"] == 0.0
    assert pct["C"]["brightness_hz"] == 50.0
    assert pct["B"]["brightness_hz"] == 100.0


def test_percentile_axis_omits_talent_with_no_qc_pass_data_on_that_axis() -> None:
    """Rule 10: a talent with brightness_hz never set anywhere in its
    entries is simply absent from "percentiles"."brightness_hz", not a
    fabricated 0/50 value; its median_f0 percentile is unaffected."""
    store = {
        "a.json": [_entry("2024-01", "vidA0000001", median_f0=200.0, brightness_hz=1000.0)],
        "b.json": [_entry("2024-01", "vidB0000001", median_f0=300.0)],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store)
    )
    assert "brightness_hz" not in result["talents"]["B"]["percentiles"]
    assert "median_f0" in result["talents"]["B"]["percentiles"]


def test_percentile_axis_absent_entirely_when_under_two_talents_have_it() -> None:
    """Rule 11: with only one talent carrying brightness_hz, that axis
    doesn't appear in ANY talent's percentiles dict."""
    store = {
        "a.json": [_entry("2024-01", "vidA0000001", median_f0=200.0, brightness_hz=1000.0)],
        "b.json": [_entry("2024-01", "vidB0000001", median_f0=300.0)],
    }
    result = site_data.build_site_data(
        {"a.json": "A", "b.json": "B"}, loader=_loader(store)
    )
    assert "brightness_hz" not in result["talents"]["A"]["percentiles"]
    assert "brightness_hz" not in result["talents"]["B"]["percentiles"]


def test_write_site_data_emits_a_js_assignment_not_bare_json(tmp_path) -> None:
    """Rule 6: the written file is loadable via a plain <script> tag
    (window.SITE_DATA = {...};), not fetched as JSON."""
    store = {"a.json": [_entry("2024-01", "vid0000001", median_f0=200.0)]}
    out = tmp_path / "data.js"
    site_data.write_site_data({"a.json": "A"}, out, loader=_loader(store))
    text = out.read_text(encoding="utf-8")
    assert text.startswith("window.SITE_DATA = ")
    assert text.rstrip().endswith(";")
    body = text[len("window.SITE_DATA = ") : text.rindex(";")]
    parsed = json.loads(body)
    assert set(parsed["talents"]) == {"A"}
