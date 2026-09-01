"""Product tests for the monthly measure step.

User-visible rules (synthetic stems ONLY — generated tones; fake windows
and picks; never Cover/hololive audio, never downloads):

1. ``measure.run_monthly(picks, windows_path, stems_dir, out_path, *,
   model_filename=None)`` measures each pick's window (windows.json:
   ``{video_id: {"start_s": .., "end_s": ..}}``) on its stem and PERSISTS
   one JSON entry per measured pick to ``out_path``, returning the same
   list.
2. The persisted entry has EXACTLY these fields: ``id``, ``month``
   ("YYYY-MM" from the pick's available_at), ``score``
   (catalog.score_video of the pick), ``window`` {start_s, end_s} (the
   windows.json values), ``features`` {median_f0, f0_iqr,
   voiced_fraction} measured from the stem, ``qc`` {pass, reason} (the
   only QC rule: IQR >= 200 fails), plus a non-empty ``model``
   provenance string naming the isolation model.
3. The stem for a pick is found in stems_dir under the name isolate
   writes for that id: "<id>_(vocals)_bs_roformer_vocals_resurrection_unwa.wav"
   for the single-model ckpt, "<id>_(Vocals)_preset_vocal_balanced.wav"
   for the legacy preset.
4. A pick whose stem is missing is SKIPPED with a warning; the run still
   succeeds and persists the rest.
5. Re-running merges by id: no duplicate entries for the same id, newly
   measured ids are added.
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
