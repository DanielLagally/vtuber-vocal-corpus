"""Product tests for the monthly measure step.

User-visible rules (synthetic stems ONLY — generated tones; fake windows
and picks; never Cover/hololive audio, never downloads):

1. ``measure.run_monthly(picks, windows_path, stems_dir, out_path, *,
   model_filename=None)`` measures each pick's stem and PERSISTS one
   JSON entry per measured pick to ``out_path``, returning the same
   list.
2. The persisted entry has EXACTLY these fields: ``id``, ``month``
   ("YYYY-MM" from the pick's available_at), ``score``
   (catalog.score_video of the pick), ``window`` (the windows.json
   entry, metadata ONLY), ``features`` {median_f0, f0_iqr,
   voiced_fraction} measured from the stem, ``qc`` {pass, reason} (the
   QC rule of rule 9), plus a non-empty ``model``
   provenance string naming the isolation model.
3. The stem for a pick is found in stems_dir under the name isolate
   writes for that id: "<id>_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"
   for the single-model ckpt, "<id>_(Vocals)_preset_vocal_balanced.wav"
   for the legacy preset.
4. A pick whose stem is missing is SKIPPED with a warning; the run still
   succeeds and persists the rest.
5. Re-running merges by id: no duplicate entries for the same id, newly
   measured ids are added.
6. Stems are the fast 90 s slices: lookup order is
   "<id>_raw90_(vocals)_<model>.wav" FIRST (the stem base is the 90 s
   slice "<id>_raw90"), falling back to the full-file-era
   "<id>_(vocals)_<model>.wav". Features are measured over the WHOLE
   stem — the stem IS the window; the full-file window offsets must
   never be applied as a slice onto the 90 s stem.
7. record["window"] is the windows.json entry persisted as METADATA
   ONLY (never clipped to the stem duration, never the slice features
   came from). Legacy 2-element array entries [start_s, end_s] are
   tolerated and normalized to the {"start_s": .., "end_s": ..} object
   form.
8. If BOTH the fast and the full-file stem exist, the fast stem is
   measured and the ambiguity is warned about.
9. QC (PLAN, one shared rule): a record fails QC iff ``f0_iqr >= 200``
   OR ``median_f0`` is non-finite OR ``median_f0 >= 600`` (600 is the
   numpy ACF tracker cap ``_FMAX`` — a junk indicator, not "she cannot
   speak that high"). The reason distinguishes the cause:
   "f0_missing" (non-finite median) / "f0_iqr" (IQR junk) /
   "f0_high" (median at or above the tracker cap), with precedence
   missing first, then IQR, then high. The record schema and merge
   behavior are unchanged.
"""

import array
import json
import math
import sys
import wave
from pathlib import Path

import pytest

from vanalysis import catalog, measure

SR = 16_000
MODEL_CKPT = "bs_roformer_vocals_resurrection_unwa.ckpt"
FAST_STEM_SUFFIX = "_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"
LEGACY_STEM_SUFFIX = "_(Vocals)_preset_vocal_balanced.wav"

# ---------------------------------------------------------------- synthesis


def _samples_to_bytes(samples: list[int]) -> bytes:
    buf = array.array("h", samples)
    if sys.byteorder == "big":
        buf.byteswap()
    return buf.tobytes()


def _sine(freq_hz: float, seconds: float, amp: float = 0.6) -> list[int]:
    peak = amp * 32767.0
    n = int(seconds * SR)
    fade = int(0.005 * SR)  # 5 ms fade to avoid clicks
    out = []
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        out.append(int(round(peak * env * math.sin(2.0 * math.pi * freq_hz * i / SR))))
    return out


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(SR)
        w.writeframes(_samples_to_bytes(samples))


# ---------------------------------------------------------------- helpers


def _pick(**overrides) -> dict:
    """A known-eligible chatting-stream pick row (score_video = 70.0)."""
    row = {
        "id": "vid0001xyz",
        "type": "stream",
        "channel_id": "UCfakechan0001",
        "available_at": "2025-03-05T12:00:00.000Z",
        "topic_id": "chatting",
        "mentions": [],
        "duration": 3600,
        "title": "zatsudan!",
    }
    row.update(overrides)
    return row


def _setup(
    tmp_path: Path, tones_per_id: dict[str, list[float]], *, legacy: bool = False
) -> tuple[Path, Path]:
    """Write a 0.5 s-per-tone synthetic stem per id (named like isolate
    writes it) plus a fake windows.json covering each whole stem."""
    stems_dir = tmp_path / "stems"
    windows: dict[str, dict[str, float]] = {}
    for vid, freqs in tones_per_id.items():
        samples: list[int] = []
        for freq in freqs:
            samples += _sine(freq, 0.5)
        suffix = LEGACY_STEM_SUFFIX if legacy else FAST_STEM_SUFFIX
        _write_wav(stems_dir / f"{vid}{suffix}", samples)
        windows[vid] = {"start_s": 0.0, "end_s": 0.5 * len(freqs)}
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(json.dumps(windows), encoding="utf-8")
    return stems_dir, windows_path


def _run(picks: list[dict], stems_dir: Path, windows_path: Path,
         out_path: Path, **kwargs) -> list[dict]:
    return measure.run_monthly(
        picks, windows_path, stems_dir, out_path, **kwargs
    )


# ------------------------------------------------------------------- tests


def test_measure_persists_exact_entry_shape(tmp_path: Path) -> None:
    """Rules 1+2+3: one pick + a 1 s 220 Hz stem (two half-second tones)
    -> exactly one persisted entry with exactly the contracted fields,
    month/score/window from the pick + windows.json, real features from
    the stem, QC pass, and a model provenance string naming the single
    RoFormer model."""
    stems_dir, windows_path = _setup(tmp_path, {"vid0001xyz": [220.0, 220.0]})
    out_path = tmp_path / "measurements.json"
    pick = _pick()

    entries = _run([pick], stems_dir, windows_path, out_path, model_filename=MODEL_CKPT)

    assert out_path.is_file(), "run_monthly must persist measurements to out_path"
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted == entries, "the file must hold exactly the returned entries"
    assert len(entries) == 1
    entry = entries[0]

    assert set(entry) == {"id", "month", "score", "window", "features", "qc", "model"}, (
        f"entry fields must be exactly the contracted set, got {sorted(entry)}"
    )
    assert entry["id"] == "vid0001xyz"
    assert entry["month"] == "2025-03", "month is YYYY-MM from the pick's available_at"
    assert entry["score"] == 70.0
    assert entry["score"] == catalog.score_video(pick)
    assert entry["window"] == {"start_s": 0.0, "end_s": 1.0}, (
        "window must be the windows.json values unchanged"
    )

    feats = entry["features"]
    assert set(feats) == {"median_f0", "f0_iqr", "voiced_fraction"}
    assert abs(feats["median_f0"] - 220.0) < 5.0, (
        f"median_f0 must be measured from the stem (~220 Hz), got {feats['median_f0']}"
    )
    assert math.isfinite(feats["f0_iqr"]), "a clean tone must have a finite IQR"
    assert 0.0 <= feats["voiced_fraction"] <= 1.0

    qc = entry["qc"]
    assert set(qc) == {"pass", "reason"}
    assert qc["pass"] is True, "a clean tone (IQR << 200) must pass QC"

    assert isinstance(entry["model"], str) and entry["model"], (
        "the entry must carry a model provenance string"
    )
    assert "bs_roformer_vocals_resurrection_unwa" in entry["model"], (
        f"provenance must name the isolation model, got {entry['model']!r}"
    )


def test_measure_qc_fails_on_wide_iqr_stem(tmp_path: Path) -> None:
    """Rule 2 (qc block): a stem with two tones half a second each (200 Hz
    then 450 Hz) has f0_iqr ~ 250 Hz — qc.pass False with a non-empty
    reason. The entry is still persisted, not dropped."""
    stems_dir, windows_path = _setup(tmp_path, {"junkvid0001": [200.0, 450.0]})
    out_path = tmp_path / "measurements.json"
    entries = _run([_pick(id="junkvid0001")], stems_dir, windows_path, out_path,
                   model_filename=MODEL_CKPT)
    assert len(entries) == 1, "QC-failing measurements are persisted, not dropped"
    qc = entries[0]["qc"]
    assert qc["pass"] is False, "f0_iqr >= 200 must fail QC"
    assert isinstance(qc["reason"], str) and qc["reason"], (
        f"a QC failure needs a reason string, got {qc['reason']!r}"
    )


# ------------------------------------------------------------- QC rule


def test_qc_verdict_reasons_and_precedence() -> None:
    """Rule 9 (the shared QC verdict, dict entries): fail iff f0_iqr >= 200
    OR median_f0 non-finite OR median_f0 >= 600 (the numpy ACF tracker cap
    _FMAX — a junk indicator, not a pitch ceiling) OR voiced_fraction < 0.15
    (too little voiced content for the median/IQR to be a trustworthy
    statistic — a handful of voiced frames can post a deceptively tight
    IQR purely from having almost no data). Reasons distinguish the cause
    with precedence missing -> voiced_fraction -> iqr -> high; a pass has
    reason None."""
    from vanalysis import qc  # noqa: local import so a red phase fails only here

    assert qc.qc_verdict({"median_f0": 220.0, "f0_iqr": 30.0, "voiced_fraction": 0.5}) == (
        True,
        None,
    )
    assert qc.qc_verdict({"median_f0": 599.5, "f0_iqr": 30.0, "voiced_fraction": 0.5}) == (
        True,
        None,
    )
    assert qc.qc_verdict(
        {"median_f0": 220.0, "f0_iqr": 30.0, "voiced_fraction": 0.15}
    ) == (True, None), "voiced_fraction exactly at the floor passes"
    assert qc.qc_verdict({"median_f0": math.nan, "f0_iqr": math.nan}) == (
        False,
        "f0_missing",
    )
    # precedence: a missing median wins over everything else
    assert qc.qc_verdict({"median_f0": math.nan, "f0_iqr": 500.0}) == (
        False,
        "f0_missing",
    )
    assert qc.qc_verdict(
        {"median_f0": math.nan, "f0_iqr": 30.0, "voiced_fraction": 0.05}
    ) == (False, "f0_missing")
    # too little voiced content: fails even with a deceptively tight IQR
    assert qc.qc_verdict(
        {"median_f0": 220.0, "f0_iqr": 5.0, "voiced_fraction": 0.05}
    ) == (False, "voiced_fraction")
    assert qc.qc_verdict({"median_f0": 220.0, "f0_iqr": 30.0}) == (
        False,
        "voiced_fraction",
    ), "a features dict with no voiced_fraction at all cannot be trusted either"
    # precedence: low voiced_fraction wins over IQR junk
    assert qc.qc_verdict(
        {"median_f0": 220.0, "f0_iqr": 250.0, "voiced_fraction": 0.05}
    ) == (False, "voiced_fraction")
    assert qc.qc_verdict({"median_f0": 220.0, "f0_iqr": math.nan, "voiced_fraction": 0.5}) == (
        False,
        "f0_iqr",
    )
    assert qc.qc_verdict({"median_f0": 220.0, "f0_iqr": 200.0, "voiced_fraction": 0.5}) == (
        False,
        "f0_iqr",
    )
    assert qc.qc_verdict({"median_f0": 615.0, "f0_iqr": 30.0, "voiced_fraction": 0.5}) == (
        False,
        "f0_high",
    )


def test_qc_requalify_recomputes_from_stored_features_only() -> None:
    """requalify(records): applies the CURRENT qc_verdict to each
    record's already-stored features (no re-measurement); every other
    key, including a stale/wrong qc block, is left as-is except qc
    itself; a record without features is passed through unchanged."""
    from vanalysis import qc

    records = [
        {
            "id": "stalepass01",
            "features": {"median_f0": 220.0, "f0_iqr": 30.0, "voiced_fraction": 0.05},
            "qc": {"pass": True, "reason": None},  # stale: predates the floor
        },
        {
            "id": "nofeatures01",
            "qc": {"pass": False, "reason": "f0_missing"},
        },
    ]
    out = qc.requalify(records)
    assert out[0]["id"] == "stalepass01"
    assert out[0]["qc"] == {"pass": False, "reason": "voiced_fraction"}
    assert out[0]["features"] == records[0]["features"]
    assert out[1] == records[1]


def test_measure_qc_fails_silent_stem_with_missing_reason(tmp_path: Path) -> None:
    """Rule 9 (measure): a stem with no voiced frames measures
    median_f0 = NaN — the record fails QC with reason "f0_missing", NOT
    "f0_iqr" (even though the IQR is NaN too: missing beats iqr)."""
    vid = "silence0001"
    stems_dir = tmp_path / "stems"
    _write_wav(stems_dir / f"{vid}{FAST_STEM_SUFFIX}", [0] * SR)  # 1 s silence
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 0.0, "end_s": 1.0}}), encoding="utf-8"
    )

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert len(entries) == 1, "the silent stem is still measured and persisted"
    assert math.isnan(entries[0]["features"]["median_f0"]), (
        "silence measures NaN median, never an invented Hz"
    )
    qc = entries[0]["qc"]
    assert qc["pass"] is False
    assert qc["reason"] == "f0_missing", (
        f"a NaN median is missing, not IQR junk — got {qc['reason']!r}"
    )


def _patch_features(monkeypatch, median_f0: float, f0_iqr: float) -> None:
    """Pin what the stem measures so the QC wiring of run_monthly is tested
    deterministically. (The ACF tracker cannot literally report an
    arbitrary >=600 Hz median from a synthetic tone — such medians arise
    from octave errors on real stems — so the features are pinned, not the
    audio.)"""
    monkeypatch.setattr(
        measure,
        "_stem_features",
        lambda stem: {
            "median_f0": median_f0,
            "f0_iqr": f0_iqr,
            "voiced_fraction": 0.6,
        },
    )


def test_measure_qc_fails_median_f0_at_or_above_600(
    tmp_path: Path, monkeypatch
) -> None:
    """Rule 9 (measure): a measured median_f0 >= 600 (tracker cap, octave
    junk) with a tight IQR fails QC with reason "f0_high". The entry is
    still persisted."""
    vid = "octavejunk1"
    stems_dir, windows_path = _setup(tmp_path, {vid: [220.0, 220.0]})
    _patch_features(monkeypatch, median_f0=615.0, f0_iqr=30.0)

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert len(entries) == 1, "QC-failing measurements are persisted, not dropped"
    qc = entries[0]["qc"]
    assert qc["pass"] is False
    assert qc["reason"] == "f0_high", (
        f"a median at/above the 600 tracker cap is 'high', got {qc['reason']!r}"
    )


def test_measure_qc_passes_median_f0_just_under_600(
    tmp_path: Path, monkeypatch
) -> None:
    """Rule 9 (measure): median_f0 just under 600 with a tight IQR passes
    QC — there is no ceiling below the tracker cap."""
    vid = "neartop0001"
    stems_dir, windows_path = _setup(tmp_path, {vid: [220.0, 220.0]})
    _patch_features(monkeypatch, median_f0=599.5, f0_iqr=30.0)

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    qc = entries[0]["qc"]
    assert qc["pass"] is True, "599.x Hz with a tight IQR must pass QC"
    assert qc["reason"] is None


def test_measure_qc_fails_iqr_exactly_200(tmp_path: Path, monkeypatch) -> None:
    """Rule 9 (measure, boundary preserved): f0_iqr = 200.0 exactly fails
    QC (fail is >=, not >) with reason "f0_iqr"."""
    vid = "edgeiqr0001"
    stems_dir, windows_path = _setup(tmp_path, {vid: [220.0, 220.0]})
    _patch_features(monkeypatch, median_f0=220.0, f0_iqr=200.0)

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    qc = entries[0]["qc"]
    assert qc["pass"] is False, "iqr = 200.0 exactly must fail (boundary is >=)"
    assert qc["reason"] == "f0_iqr"


def test_measure_missing_stem_skipped_with_warning(tmp_path: Path) -> None:
    """Rule 4: a pick without a stem is skipped with a warning; the run
    still succeeds and persists the remaining pick. Uses the legacy preset
    stem naming (model_filename=None)."""
    stems_dir, windows_path = _setup(
        tmp_path, {"present01": [220.0]}, legacy=True
    )
    picks = [_pick(id="present01"), _pick(id="absent9999")]

    with pytest.warns(UserWarning, match="absent9999"):
        entries = _run(picks, stems_dir, windows_path, tmp_path / "measurements.json")

    assert [e["id"] for e in entries] == ["present01"], (
        "only the pick with a stem may be measured"
    )


def test_measure_rerun_merges_by_id_without_duplicates(tmp_path: Path) -> None:
    """Rule 5: re-running over the same pick does not duplicate its entry;
    a second pick is merged in as a new id."""
    stems_dir, windows_path = _setup(
        tmp_path, {"mergeaaaa1": [220.0], "mergebbbb2": [240.0]}
    )
    out_path = tmp_path / "measurements.json"
    pick_a = _pick(id="mergeaaaa1")
    pick_b = _pick(id="mergebbbb2", available_at="2025-04-05T12:00:00.000Z")

    first = _run([pick_a], stems_dir, windows_path, out_path, model_filename=MODEL_CKPT)
    again = _run([pick_a], stems_dir, windows_path, out_path, model_filename=MODEL_CKPT)
    assert len(again) == 1 and again[0]["id"] == "mergeaaaa1", (
        "re-running the same pick must not duplicate the entry"
    )

    merged = _run([pick_a, pick_b], stems_dir, windows_path, out_path,
                  model_filename=MODEL_CKPT)
    ids = [e["id"] for e in merged]
    assert sorted(ids) == ["mergeaaaa1", "mergebbbb2"], (
        f"merge must add the new id without duplicating, got {ids}"
    )
    assert len(merged) == len(set(ids)), "one entry per id, no duplicates"
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted == merged, "the merged file is what is persisted"
    assert first[0] == next(e for e in merged if e["id"] == "mergeaaaa1"), (
        "the existing entry must survive the merge unchanged"
    )


# ------------------------------------------------------- fast-stem contract


def test_measure_finds_raw90_fast_stem(tmp_path: Path) -> None:
    """Rule 6 (lookup): the stems actually produced for the fast pipeline
    are named "<id>_raw90_(vocals)_<model>.wav" (the stem base is the 90 s
    slice "<id>_raw90", not the full file "<id>"). Such a stem IS found
    and measured — not skipped as missing."""
    vid = "fast90abc01"
    stems_dir = tmp_path / "stems"
    _write_wav(stems_dir / f"{vid}_raw90{FAST_STEM_SUFFIX}", _sine(220.0, 0.5))
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 0.0, "end_s": 0.5}}), encoding="utf-8"
    )

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert [e["id"] for e in entries] == [vid], (
        "the <id>_raw90_(vocals)_... fast stem must be found and measured"
    )
    feats = entries[0]["features"]
    assert abs(feats["median_f0"] - 220.0) < 5.0, (
        f"measured from the fast stem, got {feats['median_f0']}"
    )


def test_measure_falls_back_to_full_file_stem_name(tmp_path: Path) -> None:
    """Rule 6 (fallback): a stem named "<id>_(vocals)_<model>.wav" without
    the _raw90 marker (the pre-fast naming) is still found and measured."""
    vid = "fallbackb01"
    stems_dir = tmp_path / "stems"
    _write_wav(stems_dir / f"{vid}{FAST_STEM_SUFFIX}", _sine(220.0, 0.5))
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 0.0, "end_s": 0.5}}), encoding="utf-8"
    )

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert [e["id"] for e in entries] == [vid], (
        "the fallback <id>_(vocals)_... stem must still be found"
    )
    assert abs(entries[0]["features"]["median_f0"] - 220.0) < 5.0


def _tone_then_silence_samples(
    tone_hz: float, tone_s: float, total_s: float
) -> list[int]:
    return _sine(tone_hz, tone_s) + [0] * int(round((total_s - tone_s) * SR))


def test_measure_uses_whole_stem_not_window_reslice(tmp_path: Path) -> None:
    """Rule 6 (whole stem): features come from the WHOLE 90 s stem — the
    stem IS the window. A windows.json entry holding full-15-min offsets
    (60–150 s here) is metadata only; re-slicing the 90 s stem with it
    would land in the silent tail (median_f0 nan) — that must not
    happen."""
    vid = "wholestem1"
    stems_dir = tmp_path / "stems"
    _write_wav(
        stems_dir / f"{vid}{FAST_STEM_SUFFIX}",
        _tone_then_silence_samples(220.0, tone_s=30.0, total_s=90.0),
    )
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 60.0, "end_s": 150.0}}), encoding="utf-8"
    )

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert len(entries) == 1, "the stem must be measured"
    feats = entries[0]["features"]
    assert math.isfinite(feats["median_f0"]), (
        f"whole-stem measurement must see the 220 Hz first-30 s tone, got {feats}"
    )
    assert abs(feats["median_f0"] - 220.0) < 5.0, (
        f"features must reflect the whole stem (tone at its head), got {feats}"
    )


def test_window_metadata_is_the_windows_json_entry(tmp_path: Path) -> None:
    """Rule 7: record["window"] is the windows.json entry verbatim (object
    form), persisted as metadata only — NOT clipped to the stem duration
    and NOT the slice the features came from (features are whole-stem)."""
    vid = "metaonly001"
    stems_dir = tmp_path / "stems"
    _write_wav(
        stems_dir / f"{vid}{FAST_STEM_SUFFIX}",
        _tone_then_silence_samples(220.0, tone_s=30.0, total_s=90.0),
    )
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 60.0, "end_s": 150.0}}), encoding="utf-8"
    )

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert len(entries) == 1
    assert entries[0]["window"] == {"start_s": 60.0, "end_s": 150.0}, (
        f"window metadata must be the windows.json entry, got {entries[0]['window']}"
    )
    assert abs(entries[0]["features"]["median_f0"] - 220.0) < 5.0, (
        "while the features remain whole-stem"
    )


def test_legacy_array_window_entry_is_tolerated(tmp_path: Path) -> None:
    """Rule 7 (legacy): the 24-clip era wrote windows.json values as
    2-element arrays [start_s, end_s]. Reading them must not crash and the
    record's window metadata is normalized to the object form."""
    vid = "legacywin01"
    stems_dir = tmp_path / "stems"
    _write_wav(
        stems_dir / f"{vid}{FAST_STEM_SUFFIX}",
        _tone_then_silence_samples(220.0, tone_s=30.0, total_s=90.0),
    )
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(json.dumps({vid: [60.0, 150.0]}), encoding="utf-8")

    entries = _run(
        [_pick(id=vid)],
        stems_dir,
        windows_path,
        tmp_path / "measurements.json",
        model_filename=MODEL_CKPT,
    )

    assert len(entries) == 1, "a legacy array window entry must not crash the run"
    assert entries[0]["window"] == {"start_s": 60.0, "end_s": 150.0}, (
        f"array window must be normalized to the object form, "
        f"got {entries[0]['window']}"
    )
    assert abs(entries[0]["features"]["median_f0"] - 220.0) < 5.0


def test_measure_prefers_fast_stem_when_both_exist(tmp_path: Path) -> None:
    """Rule 8: when both "<id>_raw90_(vocals)_..." and
    "<id>_(vocals)_..." exist, the fast stem is measured and the
    ambiguity is warned about (distinguishable: the fast stem carries the
    220 Hz tone, the full-file-era stem 440 Hz)."""
    vid = "bothstem001"
    stems_dir = tmp_path / "stems"
    _write_wav(stems_dir / f"{vid}_raw90{FAST_STEM_SUFFIX}", _sine(220.0, 1.0))
    _write_wav(stems_dir / f"{vid}{FAST_STEM_SUFFIX}", _sine(440.0, 1.0))
    windows_path = tmp_path / "windows.json"
    windows_path.write_text(
        json.dumps({vid: {"start_s": 0.0, "end_s": 1.0}}), encoding="utf-8"
    )

    with pytest.warns(UserWarning, match=vid):
        entries = _run(
            [_pick(id=vid)],
            stems_dir,
            windows_path,
            tmp_path / "measurements.json",
            model_filename=MODEL_CKPT,
        )

    assert len(entries) == 1
    assert abs(entries[0]["features"]["median_f0"] - 220.0) < 5.0, (
        f"the fast (<id>_raw90) stem must win, got {entries[0]['features']}"
    )
