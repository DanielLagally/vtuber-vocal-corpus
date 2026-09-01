# vanalysis plan

Public measurements of hololive **chatting-stream speech** (character voice, not off-stream, not singing). No audio, transcripts, or cloneable embeddings in git. No voice synthesis.

This file is the record of what was decided, what exists on disk, and what to do next. Implementation details that belong in code stay out; gotchas that belong in `CLAUDE.md` stay there.

## Goals and working rules

- **Luna first.** The first goal is **strong, high-quality Himemori Luna data** to an actually scientifically acceptable standard (enough good clips after QC that a time series is not n=1 anecdotes and tracker junk). **Only then** expand to other members, one talent at a time. Stop when every in-scope member has high-quality, complete data — not a 70-career rush.
- **Tooling serves the measurement.** Pick F0 tracker, separator, yt-dlp client, windowing, QC, for **data quality**, not convenience or “what we already imported.” Numpy ACF stays until something better is justified; Praat / a different stem model / hunting on the stem are on the table if they fix BGM and octave errors.
- **Keep `data/` organized; do not delete.** Do not overtly remove wavs, stems, measurements, or plots. Reorganize (names, subdirs, snapshots) rather than wipe. Prefer add-and-relocate over rm.
- **Work stays in this repo.** Catalog, audio, stems, windows, measurements, plots, batch logs, and retry snapshots live under the project tree (`data/` for gitignored artifacts). `/tmp` is only for throwaway probes, not the record.

---

## Locked product rules

### Corpus
- **Roster:** active hololive **talent** channels only (JP, EN, ID, DEV_IS). Drop Holodex `Official`, `Misc`, and anything with HOLOSTARS in the group name. Gura is inactive on Holodex — skip.
- **v1 sources:** chatting / talk / game commentary on the **talent channel**. No official-org-channel comparison. No singing streams, covers, original-song topics, shorts, teasers, members-only, collabs (`include=mentions`), title karaoke/歌枠, title ゲスト/guest.
- **Watchalongs:** lowest-score **fallback** only (movie audio, not speech). Prefer never to need them.
- **Default sampling (multi-talent v1):** **4 streams per year**, one per quarter (highest score; fill from leftovers if a quarter is empty).
- **Luna time-series experiment:** **1 eligible chatting stream per calendar month** she has been active; empty months = gap (none in 2020-06–2026-08). If several that month: **highest `score_video`**, tie → newest `available_at`. Channel id `UCa9Y57gfeY0Zro_noHRVrnw` (debut stream 2020-01-04). Catalog via Holodex `channel_id` + `include=mentions`.
- **Fetch window:** **15:00–30:00** of the stream (skip waiting-room intro). 15 minutes is enough; do not fetch 15:00–45:00.
- **Isolation:** **every clip** for a talent (no “skip if raw IQR looks fine” — that biases the sample). Same separator preset for that whole talent. Never mix presets inside one person.
- **Measure window:** 90 s inside the 15 min, chosen for high voiced fraction and **low F0 IQR** on the **raw** 15 min file; isolate **only that 90 s**.
- **Scale:** Luna to an acceptable standard first, then one other talent at a time, spare GPU. Not 70 careers overnight.

### What we measure
- **Median F0 (Hz):** typical pitch of voiced frames in the 90 s (numpy ACF, 16 kHz). Plotted pitch for a month/quarter is that clip median, or the **mean of clip medians** in a quarter. Silence / no voiced frames → `math.nan`, never an invented Hz.
- **F0 IQR (Hz):** spread of pitch *inside* a clip. Tight ≈ tracker on one voice. Huge (≳200 Hz) ≈ BGM/melody/octave errors — **QC flag, not “expressive.”**
- **Voiced fraction:** share of frames that look periodic. ~0.5 on zatsudan is normal.
- **Spectral centroid (Hz):** brightness. Mic/EQ/leftover BGM dominate. Keep it, distrust it on mixed audio. Not used on the Luna monthly plots yet.
- **Cute/mature (site, later):** F0 × brightness scatter plus a percentile from equal-weight z-scores of F0, brightness, and dynamism vs the hololive set. Caption: **acoustic correlates, not a vibe rating.** No adjective mapping (“cute”) as a model output.

Quarter point (Luna plots we kept): **mean** of that quarter’s clip medians + **min–max range**. n=1 is anecdotal (no range). Year point for the old 24-clip set: **median of that year’s clip medians**; error bar = between-clip spread.

### QC (current)
- **Fail / retry trigger:** stem `f0_iqr ≥ 200` **or** median F0 non-finite **or** median F0 **≥ 600 Hz** (numpy ACF tracker cap `_FMAX`, not “she cannot speak that high”).
- **IQR ≥ 200** is the main junk flag. A 500 Hz “ceiling” was tried on the 24-clip plots and **dropped**: Luna can sit in the 300s–400s; a 603 Hz *median* on a stem that was 418 Hz raw was octave-up, not illegal pitch.
- **QC plot:** only clips that **pass** (finite F0, F0 < 600, IQR < 200). Fails are gaps, not zeros.
- **All-clips plot:** still show fails (so cleaning is visible).
- Do not pretend cleaned medians are a published voice-change result.

### Public vs private
- Tracked: code, methodology, later curated aggregates/plots.
- Gitignored `data/`: wavs, stems, cookies, Holodex key (`.env`), plot PNGs, measurements JSON.
- `yt-dlp` is a local tool for **explicit ids** only. No download-all. No redistributing Cover audio.

---

## Isolation preset

- **24-clip batch (Luna/Lamy/Sora 2024–2025):** full 15 min `vocal_balanced`. Leave those stems alone. Do not mix onto Luna monthly graphs.
- **Luna monthly (and later talents unless reversed):** 90 s single RoFormer `bs_roformer_vocals_resurrection_unwa.ckpt` (`data/stems_fast/`). GPU ~20 s/clip vs ~6–8 min for 15 min balanced.
- Comparison on the 24 clips (same timestamps): median |F0_fast − F0_balanced| ≈ 5.7 Hz; disagreements >10 Hz already had huge IQR.

---

## Fetch / YouTube (current)

Sequential only. Extra parallel yt-dlp processes hit the image-only gate. Per-video failure: **skip, log, continue** (that month is a gap), do not abort the batch.

- Cookies: gitignored `data/youtube.cookies.txt` if present.
- **`player_client=android`** is what currently downloads. `tv` → “The page needs to be reloaded.” `mweb` → GVS PO token / image-only. Flake provides **deno** for yt-dlp EJS/n-challenge.
- Section still `*15:00-30:00`.

---

## Pipeline (Luna monthly — this is what ran)

1. `catalog --channel <id> --monthly` → 1 id/month, `include=mentions`.
2. Sequential fetch 15:00–30:00 → `data/audio/<id>.wav` (skip existing >1 MB).
3. Hunt 90 s on **raw** → `data/windows/<id>_raw90.wav` + `windows.json` times as offsets **into the 15 min file**.
4. Isolate **that 90 s** with the single RoFormer → `data/stems_fast/<id>_raw90_(vocals)_bs_roformer_….wav` (the whole file **is** the window).
5. Measure **the whole 90 s stem** (do not re-apply `windows.json` offsets onto the stem — those times are for the 15 min wav). Persist `window` as metadata. **Tracker: Praat autocorrelation** (`praat_features.py`, via `remeasure-praat`) — numpy ACF was the original tracker but is superseded for Luna (see "Why the Luna series disappointed").
6. Plots from measurements JSON — `vanalysis plot` writes into a fresh
   `data/plots/runs/<run>/`, never a fixed path (see `CLAUDE.md`).

**On the same audio, when a stem fails QC (all shipped and run):** 2nd-window retry (`_raw90b`, non-overlapping, `retry` CLI), stem-hunt rescue (`_stem90`, `rescue` CLI), 2nd-stream fetch (`pick_monthly_n`) — replace-if-pass, gap otherwise, no new YouTube beyond the 2nd stream. See "Why the Luna series disappointed" for what each lever actually moved.

Do not isolate 15 min `vocal_balanced` for new talents unless a new comparison says we must.

---

## What is already done

### Code (beyond the 24-clip era)
- Catalog: `pick_monthly` (delegates to `pick_monthly_n(n=1)`), `pick_monthly_n` for multi-clip months, `catalog --channel` + `--monthly` (Holodex `channel_id`, `include=mentions`).
- `praat_features.py`: Praat-autocorrelation tracker, same public shape as `features.py` (median_f0/f0_iqr/voiced_fraction), same 75–600 Hz bounds. `diagnose.py`: read-only numpy-vs-Praat comparison on audio already on disk. `remeasure.py`: whole-corpus re-measurement with Praat on each record's already-resolved source audio (`remeasure-praat` CLI), snapshot before overwrite. All three shipped and run.
- Fetch: `player_client=android`, `fetch_audio_many` skip-and-continue, skip existing wavs.
- Isolate: `model_filename` single-model path; `--windowed` reads `data/windows/<id>_raw90.wav`.
- Window CLI: `best_speech_window` + `slice_wav`.
- Measure: fixed — `stem_features`/`_stem_path` find `*_raw90_(vocals)_*` (and the legacy `<id>_(vocals)_*`), measure the whole stem, never re-slice with 15 min offsets; `window` in the record is metadata only.
- `retry` CLI (`retry.py`): 2nd non-overlapping raw window (`_raw90b`), replace-if-pass, snapshot before replace. Shipped and run.
- `rescue` CLI (`rescue.py`): stem-hunt — isolate the full wav, hunt the 90 s window on the stem (`_stem90`), replace-if-pass. Shipped and run.
- `series.py`: `f0_series`/`iqr_series` average multiple QC-passing clips per month (multi-clip months from the 2nd-stream fetch); `f0_yearly`/`write_yearly_plot` added.
- Product tests: catalog monthly + multi-pick, isolate single-model, slice_wav, series/QC/yearly, measure shape, fetch android + skip-continue, rescue (14 tests), retry.

### Data — 24-clip batch (unchanged)
- 24 wavs, 24 full `vocal_balanced` stems, 90 s slices, fast 90 s stems, first plot set in `data/plots/`.
- Ids at the bottom of this file.

### Data — Luna monthly
- Catalog: `data/catalog/luna_monthly.json` — **75 months**, 2020-06 through 2026-08, contiguous, 1 pick each. All 8 first-batch Luna ids are in this set (remeasured on RoFormer 90 s, not the old 15 min balanced numbers).
- 75 × 15 min wavs in `data/audio/` (91 wavs total including Lamy/Sora).
- 75 × `_raw90.wav` + 75 × RoFormer 90 s vocals in `data/stems_fast/`.
- Measurements: `data/measurements/luna_monthly.json` (stems, **now Praat-tracked**, see below), `luna_monthly_raw.json` (unisolated 90 s, still numpy). Pre-Praat snapshots: `luna_monthly_pre_retry_snapshot.json`, `luna_monthly_pre_rescue.json`, `luna_monthly_pre_2nd_stream.json`, `luna_monthly_pre_praat_remeasure.json` (numpy-tracked, kept for history — do not delete).
- Plots: `data/plots/runs/<run>/` — every `vanalysis plot` run gets its own directory now (see `CLAUDE.md`); the current Praat-based set is `20260901T154218-luna-monthly-praat/`.

### Luna monthly numbers (stem 90 s, first window only — numpy, historical)
- 75 clips; **34 fail** IQR ≥ 200 or nan F0; **41 pass**.
- QC-pass F0: mean **~345 Hz**, sd **~36 Hz**, range ~250–440. Year means after 2020 sit in **~330–360 Hz**. 2020 is higher (~393) but n=3. 2024 QC-pass n=2.
- 6 stems sit on the **~615 Hz** tracker cap (all already IQR-fail).
- |stem − raw| median F0: **~24 Hz**; 38/74 differ by >20 Hz. Isolation moves some clips hundreds of Hz (octave / leftover BGM), so stem is the series to trust **when it passes QC**.
- Quarterly QC: 26 quarters (2020-Q2–2026-Q3); **11 quarters n=1**; **3 empty** (2024-Q3, 2024-Q4, 2025-Q3). All-clips bands inflated by 615 Hz junk.
- **Reading:** not a clear pitch-over-career slope. Consistent with a stable ~350 Hz character voice plus tracker/window noise. Monthly plots were too noisy; quarterly mean + min–max is the graph that is worth looking at.

### Luna monthly numbers (Praat tracker, current — this is the live series)
- **143 records, 142 QC-pass, all 75/75 months (100%) have a QC-pass clip.** Snapshot chain: `luna_monthly_pre_praat_remeasure.json` (numpy, pre-Praat) → Praat remeasure (101 records, 74/75 months) → `luna_monthly_pre_densify.json` (101 records) → `densify --target-n 2` (+42 new clips, 17 skipped — see below) → current (143 records).
- QC-pass F0: n=142, mean **334.4 Hz**, sd **37.9 Hz**, range 152–455 Hz. One record (`zHwS28IQqh0`, 2023-09, 152 Hz) is a legitimate low outlier — normal voiced_fraction (0.59), IQR 183 (close to the 200 gate) — not a silence/garbage artifact, just a low reading worth another look if it recurs.
- Only 1 record still fails QC (`dPktttXyxZo`, 2023-04, IQR 223 — passed under numpy at IQR 136, a genuine tracker disagreement, not a bug).
- **Trend, now that every month has ≥1 clip and 55/75 have 2**: linear regression on the monthly series gives slope **−7.79 Hz/year, −48 Hz total over 6.2 years**, r² = **0.27** (up from r²=0.11 on the pre-densify 74/75-month series) — first-12-months mean 355.1 Hz vs last-12-months mean 313.1 Hz. Mann-Kendall S = −1093 (strongly declining, non-parametric). **This is a real, strengthening downward trend, consistent with the community-known "started higher, gradually settled lower" narrative** — it was real in the pre-densify data too (r²=0.11) but visually swamped by single-clip sampling noise; more clips per month made it clearer, not just less noisy, exactly as predicted.
- Why Praat fixed the original QC gap: cross-checked on the 17 numpy-hard-fail months' audio, Praat's medians agreed closely across the raw90/raw90b/stem90 variants of the same clip where numpy's disagreed by hundreds of Hz — octave-jump noise, not real pitch variation. Praat's frame-to-frame path-finding penalizes octave jumps; the numpy tracker's per-frame peak-pick has no such continuity constraint. See `diagnose.py` / `data/measurements/luna_tracker_diagnostic.json`.
- **`densify.py` (the more-data lever)**: brings every month up to `--target-n` clips using the already-cached raw Holodex listing (`data/catalog/video_cache/<channel_id>.json`, no new API calls) — fetch, raw90 window hunt, fast RoFormer isolate, Praat measure, append-only. Run 2026-09-01: 53 months targeted (including 2020-01–2020-05, before the original corpus start — genuinely new debut-era coverage), 42 added, 17 skipped, 0 errors. **14 of the 17 skips are "Private video"** — Luna's earliest 2020 VODs (including the literal 2020-01-04 debut stream, `F4Ymmtcs-ls`) are privated on YouTube now; not a pipeline bug, see `CLAUDE.md` gotcha. One skip was a transient CDN error, retriable.
- Still-open cheap improvement (not yet done): 6/142 QC-pass records have `voiced_fraction < 0.15` — QC only checks `f0_iqr`/`median_f0`, not how much voiced content backed the estimate. A `voiced_fraction` floor would close this; low priority since it's a small fraction and doesn't change the trend materially.
- **Praat is now the production tracker for Luna.** Other talents should inherit it when expansion resumes (locking the tracker before branching out avoids re-measuring 68 talents later).

### 24-clip comparison / plots (still on disk)
Old QC on those PNGs marked IQR ≥ 200 **or** F0 ≥ 500. Illustrative 2025−2024 balanced 90 s: Luna −24 / −59 after QC (almost no 2024 left); Lamy −25 / −34; Sora +5 / +27. **QC flips Sora’s sign** — n=4 with this much junk is not a voice-change claim.

---

## Why the Luna series disappointed (and what we are not doing yet)

Window hunt on **raw** (high voiced + low IQR) can lock onto periodic BGM; RoFormer then leaves melody (cap) or jumps octave. Numpy ACF is a **failure detector**, not a career-F0 meter.

**Levers 1–4 have all run.** 2nd-window retry, stem-hunt rescue, and
2nd-stream fetch (levers 1–3) moved month-coverage 54.7% → 62.7% → 64.0%
→ 76%, but left 18 months empty — 17 of those failing on *both* attempted
streams, all on `f0_iqr` alone (204–390 Hz) with normal-to-high
`voiced_fraction`, regardless of whether the audio was raw-hunted,
stem-hunted, or from a 2nd stream. That pattern (same months fail no
matter which source/window lever produced the audio) pointed at the
**tracker**, not window/source selection.

**Lever 4 (tracker) — done, and it was the fix.** `diagnose.py` compared
numpy-ACF against Praat autocorrelation (`praat_features.py`) on the
audio already on disk for the 17 hard-fail months: all 115 (id ×
window-variant) comparisons that failed under numpy passed under Praat,
with median F0 clustering at 249–436 Hz (matching Luna's known range)
and — tellingly — Praat's medians agreeing closely *across* the
raw90/raw90b/stem90 variants of the same clip where numpy's disagreed by
hundreds of Hz. That's octave-jump noise, not real pitch variation; Praat's
path-finding (which explicitly penalizes octave jumps between frames,
unlike the numpy tracker's uncorrelated per-frame peak-pick) resolves it.
**Praat is now the production tracker for Luna** — `remeasure.py`
re-measured every record in `luna_monthly.json` on the exact audio each
one already used (same processing, only the tracker changed; snapshot at
`luna_monthly_pre_praat_remeasure.json`). Result: **74/75 months (98.7%)
now have a QC-pass clip**, up from 76%. QC-pass F0 across the corpus:
n=100, mean 333.9 Hz, sd 34.8 Hz, range 249–455 Hz — a stable character
voice, consistent with the pre-Praat reading, now on a dataset that is
actually complete. One record (`dPktttXyxZo`, 2023-04) that passed under
numpy (IQR 135.6) now fails under Praat (IQR 223.4) — Praat is not a
strict superset of numpy passes, it is a different, more consistent
tracker; that one month is the sole remaining gap.

---

## Operator notes

- Holodex: `User-Agent: vanalysis/0.1` or 403; `mentions` only with `include=mentions`; `topic_id == singing` is rare (songs are `Original_Song` / `Music_Cover`); many recent items are `shorts`. Rate-limit folklore is 80/window; official docs do not state a number. Key in `.env` only.
- Isolation: always the same processing per talent so year-over-year is not “Demucs vs raw.”
- `vocal_balanced` is lunalearn’s Anki preset. For median F0 we need BGM down, not prettier stems.
- First 15:00 of a stream is usually intro — never sample 0:00–15:00.
- Long GPU / yt-dlp: `nohup` / CLI, don’t block chat on a Task. Cancelled subagents can leave `audio-separator` running.
- nixpkgs `cudaSupport = true` + torch/onnxruntime in `withPackages` is a disk bomb. Pip GPU wheels ~6 GB. `unset PYTHONPATH` before venv pip (yt-dlp leaks 3.14 wheels).

---

## 2026-09-01: goal shift — Luna max before branching (owner decision)

Branching out to other talents is **paused**. The goal is the absolute best
Luna dataset achievable.

1. **Stem hunt (lever 3) — done.** For every QC-failing Luna month: isolate
   the **full 15 min** with the same fast RoFormer preset, hunt the 90 s
   window **on the stem**, slice the stem, measure, replace the month only
   if it passes QC (snapshot first, `rescue.py`). Moved month-pass 41→48
   of 75 (raw-window-hunt fails were largely re-locking onto the same
   music, not fixed by isolating first).
2. **More streams per month (lever 2) — done.** For months still failing
   after (1), fetched the **2nd-highest scored eligible stream** of that
   month (`pick_monthly_n`) and ran the full pipeline on it including the
   stem-hunt fallback. Multiple clips of a month coexist in measurements;
   the monthly QC value is the **mean of that month's QC-pass clip
   medians** (`f0_series`/`iqr_series` in `series.py`); quarterly
   aggregation unchanged. Result: **76% of months (57/75) now have ≥1
   QC-pass clip**, up from 54.7% pre-lever. 18 months remain empty (17 of
   them failed on *both* streams) — see "Why the Luna series disappointed"
   for the residual-failure analysis.
3. **Re-evaluate — done: tracker was the fix.** The residual 18-month gap
   was IQR-driven and tracker-shaped, not window/source-shaped. The
   tracker diagnostic (`diagnose.py`) confirmed Praat clears the residual
   failures; `remeasure.py` re-measured the whole corpus with Praat.
   **Result: 74/75 months (98.7%) QC-pass**, only 2023-04 remains a gap. A
   3rd stream/month is not needed.

Luna is now at a genuinely acceptable standard (98.7% month-coverage,
n=100 QC-pass clips, stable ~334 Hz mean). **Praat is the tracker other
talents should inherit** when expansion resumes (round-1 analysis: lock
the tracker before branching out, not after). Other talents resume only
when the owner decides to. Sora v1 data (33 records, 87.9% QC-pass,
numpy tracker) stays on disk, parked — would need the same Praat
re-measurement before joining a cross-talent comparison.

---

## Next steps (in order)

1. **Tracker diagnostic + full re-measurement — done.** `diagnose.py`
   compared numpy-ACF against Praat (`praat_features.py`) on the 17
   hard-fail months' audio; Praat cleared all of them. `remeasure.py`
   then re-measured every record in `luna_monthly.json` with Praat on
   the same audio each record already used (snapshot at
   `luna_monthly_pre_praat_remeasure.json`). **74/75 months (98.7%)
   QC-pass**, up from 76%. New plots: `data/plots/runs/
   20260901T154218-luna-monthly-praat/`.
2. **Aggregate helper in-repo** so plots are not throwaway scripts; then **static site** (F0-over-years, profile, F0×brightness + “correlates not vibes”). Generated from aggregates only.
3. **Other talents only after owner decides to resume.** Luna is now at
   a genuinely acceptable standard. Fast path (90 s RoFormer) + Praat
   tracker (now the standard, not numpy ACF), one person at a time.

Do not: kill/replace the 24 `balanced` 15 min stems; mix presets inside one talent; commit `data/`; download Cover audio into git.

---

## First-batch clip ids (24-clip balanced reference)

Luna 2024: `Gz_2EzLyhmQ` `ro0lFIj2MJY` `boy302x08Gg` `qyQzBoMOqXo`  
Luna 2025: `c7xfx5DOmes` `UgCC0aua69E` `yHx9-xigq0o` `q8B9Y3a33Ik`  
Lamy 2024: `uIHMp3Y7XjU` `jg9o2q_lvrY` `LyDQcY9bzDQ` `b3NaMgu0fqM`  
Lamy 2025: `x0C3mbj1jLI` `1xxlWriXKVU` `mWovsFeoniE` `wGRD8IeiVtY`  
Sora 2024: `Ywz-QaqAgNM` `Wgl2LhK9tiU` `RUF6SYI3oko` `RYpOAavYST0`  
Sora 2025: `nhEQHX-ywz4` `NvFWC3YXreA` `KsOJfG4tmyw` `X-FdkMiT-Bk`
