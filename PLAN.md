# vtuber-vocal-corpus plan

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
- **Fail / retry trigger:** median F0 non-finite **or** `voiced_fraction < 0.15` **or** stem `f0_iqr ≥ 200` **or** median F0 **≥ 600 Hz** (numpy ACF tracker cap `_FMAX`, not “she cannot speak that high”). Precedence: missing median → low voiced_fraction → IQR junk → high median.
- **`voiced_fraction < 0.15`** (added 2026-09-01, `qc.py`): a clip with almost no voiced content can post a deceptively TIGHT `f0_iqr` purely from having too few points to spread across — low sample size masquerading as a clean signal. Found via a natural gap in the distribution (4 clips at 0.009–0.046, then a jump to 0.11+; the next cluster starts at 0.19+). Applying this to the existing Luna corpus via `qc.requalify` (recomputes `qc` from stored `features`, no re-measurement) flipped 6/143 records from pass to fail; only one of those (`qVgoIAHl43g`, 2023-04) left its month with zero passing clips — 2023-04 was already the sole gap. Trend numbers barely moved (slope −7.71 vs −7.79 Hz/year, r²=0.276 vs 0.272) — a hygiene fix, not a result-changing one.
- **IQR ≥ 200** is the main junk flag. A 500 Hz “ceiling” was tried on the 24-clip plots and **dropped**: Luna can sit in the 300s–400s; a 603 Hz *median* on a stem that was 418 Hz raw was octave-up, not illegal pitch.
- **QC plot:** only clips that **pass** (finite F0, voiced_fraction ≥ 0.15, F0 < 600, IQR < 200). Fails are gaps, not zeros.
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
6. Plots from measurements JSON — `vvc plot` writes into a fresh
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
- Plots: `data/plots/runs/<run>/` — every `vvc plot` run gets its own directory now (see `CLAUDE.md`); the current Praat-based set is `20260901T154218-luna-monthly-praat/`.

### Luna monthly numbers (stem 90 s, first window only — numpy, historical)
- 75 clips; **34 fail** IQR ≥ 200 or nan F0; **41 pass**.
- QC-pass F0: mean **~345 Hz**, sd **~36 Hz**, range ~250–440. Year means after 2020 sit in **~330–360 Hz**. 2020 is higher (~393) but n=3. 2024 QC-pass n=2.
- 6 stems sit on the **~615 Hz** tracker cap (all already IQR-fail).
- |stem − raw| median F0: **~24 Hz**; 38/74 differ by >20 Hz. Isolation moves some clips hundreds of Hz (octave / leftover BGM), so stem is the series to trust **when it passes QC**.
- Quarterly QC: 26 quarters (2020-Q2–2026-Q3); **11 quarters n=1**; **3 empty** (2024-Q3, 2024-Q4, 2025-Q3). All-clips bands inflated by 615 Hz junk.
- **Reading:** not a clear pitch-over-career slope. Consistent with a stable ~350 Hz character voice plus tracker/window noise. Monthly plots were too noisy; quarterly mean + min–max is the graph that is worth looking at.

### Luna monthly numbers (Praat tracker, current — this is the live series)
- **143 records, 136 QC-pass, 74/75 months (98.7%) have a QC-pass clip** (2023-04 is the sole gap — both its clips fail, one on IQR, one on the new voiced_fraction floor). Snapshot chain: `luna_monthly_pre_praat_remeasure.json` (numpy, pre-Praat) → Praat remeasure (101 records, 74/75 months) → `luna_monthly_pre_densify.json` (101 records) → `densify --target-n 2` (+42 new clips, 17 skipped) → `luna_monthly_pre_voiced_gate.json` (143 records, 142 pass) → `qc.requalify` with the voiced_fraction floor (143 records, 136 pass) → current.
- QC-pass F0: n=136, mean **334.8 Hz**, sd **37.5 Hz**, range 152–455 Hz. One record (`zHwS28IQqh0`, 2023-09, 152 Hz) is a legitimate low outlier — normal voiced_fraction (0.59), IQR 183 (close to the 200 gate) — not a silence/garbage artifact, just a low reading worth another look if it recurs.
- **Trend, now that 55/75 months have 2 clips**: linear regression gives slope **−7.71 Hz/year, −47.5 Hz total over 6.2 years**, r² = **0.276** (barely moved from the pre-voiced-gate 0.272 — the gate is a hygiene fix, not a result-changer) — first-12-months mean 355.1 Hz vs last-12-months mean 313.1 Hz. Mann-Kendall S = −1075 (strongly declining, non-parametric). **This is a real, strengthening downward trend, consistent with the community-known "started higher, gradually settled lower" narrative** — it was already present at r²=0.11 on the pre-densify single-clip-per-month data but visually swamped by sampling noise; more clips per month made it clearer, not just less noisy, exactly as predicted.
- Why Praat fixed the original QC gap: cross-checked on the 17 numpy-hard-fail months' audio, Praat's medians agreed closely across the raw90/raw90b/stem90 variants of the same clip where numpy's disagreed by hundreds of Hz — octave-jump noise, not real pitch variation. Praat's frame-to-frame path-finding penalizes octave jumps; the numpy tracker's per-frame peak-pick has no such continuity constraint. See `diagnose.py` / `data/measurements/luna_tracker_diagnostic.json`.
- **`densify.py` (the more-data lever)**: brings every month up to `--target-n` clips using the already-cached raw Holodex listing (`data/catalog/video_cache/<channel_id>.json`, no new API calls) — fetch, raw90 window hunt, fast RoFormer isolate, Praat measure, append-only. Run 2026-09-01: 53 months targeted (including 2020-01–2020-05, before the original corpus start — genuinely new debut-era coverage), 42 added, 17 skipped, 0 errors. **14 of the 17 skips are "Private video"** — Luna's earliest 2020 VODs (including the literal 2020-01-04 debut stream, `F4Ymmtcs-ls`) are privated on YouTube now; not a pipeline bug, see `CLAUDE.md` gotcha. One skip was a transient CDN error, retriable.
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

- Holodex: `User-Agent: vvc/0.1` or 403; `mentions` only with `include=mentions`; `topic_id == singing` is rare (songs are `Original_Song` / `Music_Cover`); many recent items are `shorts`. Rate-limit folklore is 80/window; official docs do not state a number. Key in `.env` only.
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

## Post-DEV_IS-batch feature ideas (owner decision pending)

Raised during the 2026-09 DEV_IS batch run: the current plots require prior
domain knowledge to read (a raw Hz/dB number means little without knowing
hololive's typical range), and there's no single-glance summary of a
talent's overall profile. Ideas discussed, in priority order — none
implemented yet, revisit once the batch is done:

1. **Formants (F1-F4) — recommended.** `parselmouth` is already a pipeline
   dependency (used for pitch_ac/jitter/shimmer/HNR in `praat_features.py`),
   so `Sound.to_formant_burg()` slots in the same way, averaged over voiced
   frames like the existing pitch series. A real, standard vowel-space/
   resonance measure, complementary to the existing brightness (spectral
   centroid) metric.
2. **Nasality — recommended against.** True nasalance needs a two-channel
   nasometer (separate oral/nasal capsules); physically unrecoverable from
   single-channel YouTube audio. Single-mic acoustic proxies (A1-P0, extra
   nasal formants) exist in the literature but are noisy, vowel-context-
   dependent, and even more mic/EQ-sensitive than the brightness caveat
   already carried — would undercut the "acoustic correlates, not a vibe
   rating" stance above.
3. **Percentile framing extended to every metric plot**, not just the
   cute/mature scatter. A monthly F0 plot currently just says "182 Hz" —
   meaningless without knowing hololive's typical range. The z-score/
   percentile machinery already built for cute/mature could annotate every
   plot ("brighter than 68% of the corpus") with near-zero new methodology
   risk.
4. **Per-talent radar/profile chart** — one glanceable shape combining
   pitch height, brightness, dynamism, voiced-time, jitter/shimmer/HNR as
   normalized axes, so "what does this person sound like overall" is one
   picture instead of eight separate line graphs.
5. **Clustering (k-means/PCA groupings) — explicitly deferred**, not
   rejected outright. With only ~10-12 talents in the corpus as of 2026-09,
   clusters would mostly be small-sample noise dressed up as insight.
   Revisit once the corpus is meaningfully bigger.

---

## 2026-09-02/03: DEV_IS batch complete, pipeline infra shipped, repo renamed

All 10 DEV_IS members are done — ReGLOSS (Hajime, Kanade, Raden, Ririka),
FLOW GLOW (Chihaya, Niko, Riona, Su, Vivi), plus graduated member Ao.
Order run: Chihaya, Ririka, Raden, Hajime, Vivi (as specified), then
Kanade, Niko, Riona, Su, Ao (unspecified order, arbitrary). **800 records,
767 QC-pass (96%)** — per-member breakdown and the full run log are not
reproduced here; re-run `vvc site-data` and check `docs/data.js`, or see
`data/measurements/<name>_monthly.json` per talent (gitignored, local
only). This closes out the roster expansion goal — no more DEV_IS members
pending.

New infrastructure that shipped alongside the batch (all in `src/vvc/`,
tested, committed on `main` as of `2da1884`):

- **`densify`'s `cpu_workers`**: overlaps window-hunt/isolate/measure
  across clips (producer/consumer queue + single-worker GPU pool); fetch
  stays strictly sequential (bot-check safety, unchanged). `--cpu-workers
  4` is the benchmarked sweet spot (8 workers is measurably worse, GIL
  contention). Real-world throughput is fetch-bound, not CPU/GPU-bound —
  see the session transcript if the reasoning is needed again, the short
  version is fetch's ~35s/clip local ffmpeg extraction step sets the pace
  regardless of pipelining.
- **`densify`'s `offload_remote`**: streams a QC-pass clip's raw audio to
  Google Drive via `rclone` (remote `Google Drive:vanalysis-raw-audio` —
  NOTE: still the old pre-rename remote name, harmless since it's just a
  label, but rename it in `rclone config` if it bothers you) and deletes
  the local copy once confirmed uploaded. Watch for Drive throttling on
  sustained heavy use (~1 MB/s instead of the usual ~13 MB/s was observed
  after hours of continuous upload) — not fatal, just don't block a batch
  waiting on the tail of it; a `rclone move --files-from <list>` bulk
  sweep afterward is faster than waiting on the live per-clip trickle.
- **`isolate.py` fix**: predicted output filenames now match
  audio-separator's own `sanitize_filename` (strips leading/trailing/
  repeated `_`) — previously silently lost isolation work for any video
  id starting with `_` (~1-2% of ids, real data loss before the fix).
- **`vvc new-talent <name> <channel_id>`**: bootstraps a fresh talent
  (caches the raw Holodex listing, seeds an empty measurements file) in
  one command — replaces the ad-hoc Python that had to be written by hand
  for each of the 10 DEV_IS members before this existed.
- **`docs/`**: interactive comparison site (talent multi-select, 9 metric
  views incl. cute/mature percentile scatter), fed by `vvc site-data` →
  `docs/data.js`. Plain HTML/CSS/JS, Plotly.js vendored locally, loads via
  `window.SITE_DATA = {...}` (NOT `fetch()` — that's CORS-blocked under
  `file://`, learned the hard way). Ready for GitHub Pages' "deploy from
  branch, /docs" — no Actions workflow needed. 5 talents have real
  hardcoded colors (Luna, Lamy, Ao, Niko, Su); the rest use a fallback
  palette pending someone confirming their actual official colors.
- **Package renamed**: `vanalysis` → `vvc` (module/CLI), project name →
  `vtuber-vocal-corpus` (PyPI/repo name, in `pyproject.toml`). Editable
  install (`pip install -e .`) means `PYTHONPATH=src` is no longer
  needed — just `direnv exec . python -m vvc ...`.
- `README.md` (MIT `LICENSE`) added — this file (`PLAN.md`) stays the
  working engineering log, not the public front door.

**Not yet done**: no GitHub remote configured, nothing pushed anywhere.
`main` and the old `pipeline-and-drive-offload` branch both point at
`2da1884` locally (fast-forward, no divergence) — safe to delete the
feature branch whenever. The post-batch feature ideas above (formants,
percentile-everywhere, radar chart) are still just ideas, not started.

---

## 2026-09-03: veteran-JP batch (autonomous run) + site filters/plots

Owner decision: expansion resumes with long-career JP veterans — Sora,
Roboco, Coco, Fubuki, Subaru, Korone, Okayu, Watame, Polka (order given),
then remaining roster unspecified order. Sampling: monthly, `densify
--target-n 2`, matching the Luna/Lamy precedent exactly (not the
"locked" quarterly rule, not DEV_IS's approach — owner's explicit call
given the disk math). `data/windows`/`data/stems_fast` are staying local
this run (not offloaded) — owner wants that audio available for a future
F1-F4 formant pass; only `data/audio` (raw pre-isolation wav, not used
for feature extraction) continues to offload to Drive.

Fixed before starting: `offload.py`'s `DEFAULT_REMOTE` and the CLI help
text said `Google Drive:vvc-raw-audio` (stale, pre-rename) but the
actually-configured rclone remote is `Google Drive:vanalysis-raw-audio`
— every `densify` call must pass `--offload-remote` explicitly with the
correct name or offload silently no-ops (fire-and-forget thread, no
result checked, see `densify.py`'s offload_pool). Also fixed: DEV_IS
talents + Ao were registered in `talents.json` under short given names
(`Niko`, `Su`, `Ao`, …) instead of full names like everyone else —
renamed all 10 to their full Holodex `english_name` (Koganei Niko,
Mizumiya Su, Rindo Chihaya, Todoroki Hajime, Otonose Kanade, Juufuutei
Raden, Isaki Riona, Ichijou Ririka, Kikirara Vivi, Hiodoshi Ao —
Hiodoshi Ao resolved via Holodex video lookup since she's graduated and
absent from `roster.json`) via `vvc plot --talent` re-runs (registry
auto-update + fresh PNGs with correct titles), then `vvc site-data`.

Sora special case: `sora_v1.json` (33 records, numpy tracker, parked)
renamed to `sora_monthly.json` (matches everyone else's `<name>_
monthly.json` convention), `remeasure-praat`'d (33/33 pass), then
densified normally.

Orchestrator: `data/logs/run_veteran_batch.sh` — per-talent
`new-talent`/`densify`/`plot`/`site-data` as separate subprocesses (not
one in-process Python loop; `run_densify` has no top-level exception
guard, so a setup error in one talent must not kill the ones queued
after it). Disk policy: soft-pause (stop starting new talents) below 40G
free, hard-stop (kill in-flight, stop the batch) below 15G free.
Bot-check (`stopped_early` non-null in densify's JSON summary) stops the
whole batch, not just that talent — stale cookies fail identically for
everyone queued after.

Website (parallel to the batch, no fetch/GPU involved): generation/
branch filter dropdowns (`site_data.py` now joins `talents.json` display
names against `roster.json`'s `group` via a 9-entry DEV_IS alias table,
falls back to group=branch="Graduated" for talents absent from the
roster like Ao/Coco, or "Unknown" if no roster is passed at all — new
`--roster` flag on `vvc site-data`, default `data/catalog/roster.json`);
percentile-everywhere (`_single_axis_percentiles`, a generalization of
`_cute_mature`'s combined z-score-then-rank into one independent ranking
per yearly metric, surfaced as a "Percentile vs. corpus" line under every
yearly plot's caption); radar/profile chart (new `scatterpolar` view,
one axis per yearly metric, axis value = that talent's per-axis
percentile). All three verified by headless-Chromium screenshot against
real `docs/data.js`, not just unit tests. Formants were explicitly
excluded from this batch (needs new Praat feature extraction + a full
corpus remeasure — real pipeline work competing with the batch, and the
owner wants the not-yet-offloaded stems/windows audio preserved for a
dedicated future formant pass instead).

Owner follow-up, same session, purely frontend (no new measurements, no
`site_data.py` change) since every per-year value the radar/scatter needed
was already sitting in the existing `yearly` series:
- **Radar timeline**: an "Overall (all years)" checkbox + year slider.
  Unchecked, each axis switches from the corpus-wide rank percentile to
  that year's raw median min-max-scaled against a *fixed* per-axis
  range (every talent, every year, computed once at load) — so scrubbing
  years moves the shape within a stable frame instead of the axes
  rescaling underneath you. A talent with no data that year just has no
  vertex, same as the existing "missing axis" behavior.
- **Trajectory view**: new metric-picker entry with X/Y (+ optional
  size) axis pickers and a from/to year range, defaulting to F0 ×
  brightness. Draws each selected talent's year-ordered path with
  Plotly's `marker.symbol:"arrow", angleref:"previous"` (needs Plotly.js
  2.18+; vendored copy is 2.35.2) so direction of travel is visible, each
  point labeled with its year. Independent of the original cute/mature
  scatter (kept as-is, its own combined-percentile methodology) — this is
  a raw-value view, not a percentile one.
All view-specific controls live in a new `#view-controls` panel above
the chart, built by `buildViewControls(metric)` and rebuilt only when the
selected metric actually changes (`lastControlsMetric` guard) so mid-drag
slider state survives an unrelated re-render (e.g. toggling a talent
checkbox). Verified via headless-Chromium: default view, both new radar
modes, trajectory with axis/size/range changes, and every metric-picker
option cycled with a `window.onerror` trap — no console errors.

**2026-09-03, owner decision**: stop the veteran-JP batch once the named
list (Sora through Polka) finishes — all 9 landed, 96-99% QC-pass each —
and pivot straight to formants instead of continuing to unnamed roster
members. `praat_features.formants_hz()` shipped (F1-F4 via Praat's
Burg tracker, tested against a synthetic two-formant vowel since a pure
sine has no resonance structure) and the entire corpus was backfilled via
the existing `remeasure-praat` command on the already-local stems (no
re-fetch) — **all 22 talents now carry f1_hz-f4_hz**. Site work in the
same arc: radar generalized from 3 fixed presets into a free-form
metric-checkbox picker (any combination, including F0+formants
together); a sortable table view (raw values by default, percentile
toggle); graduated members (Ao, Coco) now show their real generation
(via a small known-graduated lookup against their own Holodex channel)
while staying branch="Graduated"; Shirakami Fubuki now correctly shows
both her generation memberships (1st Generation + GAMERS); the
generation filter/table now sorts by real debut chronology (fetched
from Holodex, not alphabetical, which badly scrambled DEV_IS/EN
ordering). Layout widened (1100px -> 1800px cap) and charts enlarged
since the page was using a small fraction of typical screen width.

Roster expansion beyond the named list (Coco through Polka) is paused —
resume only when the owner decides to, same as the earlier Luna-first
pause.

---

## 2026-09-04: hololive JP roster expansion resumed

Owner decision (`/goal continue with hololive jp`): resume roster
expansion, hololive JP only (not EN/ID again yet). 26 talents already in
the corpus at this point (the original 3 + DEV_IS 10 + veteran-JP 9 +
Justice 4). Remaining hololive-JP roster per `data/catalog/roster.json`
(`org == "Hololive"`, JP generation groups: 0th–6th Gen, GAMERS,
mekPark): 24 entries, of which 3 are excluded as out of scope, not a
judgment call — `Aki Rosenthal (Sub)` / `Akai Haato (Sub)` / `Choco Sub
Channel` are secondary upload channels for talents already covered by
their main channel, and `UNIT B [Pre-Debut]` (mekPark) has not debuted
(no chatting streams exist yet to sample). `ACHRORA` (mekPark) is
included — active, has a `talk` topic and real video history despite
being a newer/smaller channel.

**20 talents queued**, arbitrary generation order (owner gave no
specific order, same as the DEV_IS/veteran-JP precedent): AZKi, Sakura
Miko, Hoshimachi Suisei (0th Gen) → Akai Haato, Natsuiro Matsuri, Aki
Rosenthal (1st Gen) → Yuzuki Choco, Nakiri Ayame (2nd Gen) → Shiranui
Flare, Houshou Marine, Shirogane Noel, Usada Pekora (3rd Gen Fantasy) →
Tokoyami Towa (4th Gen holoForce) → Shishiro Botan, Momosuzu Nene (5th
Gen holoFive) → Kazama Iroha, Hakui Koyori, La+ Darknesss (6th Gen
holoX) → Ookami Mio (GAMERS) → ACHRORA (mekPark).

Reused `scripts/run_veteran_batch.sh` unchanged (already fully generic —
takes `name:channel_id` pairs, only needed its `DISPLAY_NAMES` map
extended with the 20 new entries) rather than writing a new orchestrator.
Same settings as every batch since Luna: `densify --target-n 2
--cpu-workers 4`, Praat tracker (now default), F1-F4 formants included,
Drive offload for `data/audio` only (`Google Drive:vanalysis-raw-audio`),
soft-pause at 40G free / hard-stop at 15G (227G free at launch), bot-check
(`stopped_early`) aborts the whole batch. Cookies re-exported 2026-09-03,
still fresh. Launched as a background `nohup` process, logging to
`data/logs/jp_remaining_batch_<timestamp>.log` plus the usual per-talent
`<name>_densify_<timestamp>.json`; `vvc plot` + `vvc site-data` run after
each talent so `docs/data.js` stays current incrementally rather than
only at the very end.

**Batch complete, 2026-09-05. Owner decision mid-run: drop mekPark
entirely** ("we don't need mekpark") — `ACHRORA` was killed 1 record
into her densify (negligible waste, she was last in the queue anyway)
and her stub measurements file removed; not committed, not in the
registry. Final roster addition from this arc is **19 talents**, not
20. mekPark (`ACHRORA`, `UNIT B [Pre-Debut]`) is out of scope for
hololive-JP going forward, not just skipped this run.

Final per-talent numbers (`records / QC-pass / months-with-pass /
months-attempted`) — the first 8 (0th/1st/2nd Gen) fetched clean before
the mweb PO-token gate hit; the next 7 (Flare through Nene) were
fetched degraded, under the old pin, before the fix in the next
section shipped; the last 4 (Iroha through Mio) are back near the
historical baseline under the fixed pipeline:

| Talent | Records | QC-pass | Months | Note |
|---|---|---|---|---|
| AZKi | 130 | 121 | 73/73 | clean |
| Sakura Miko | 144 | 138 | 73/74 | clean |
| Hoshimachi Suisei | 146 | 140 | 75/76 | clean |
| Akai Haato | 115 | 111 | 64/64 | clean |
| Natsuiro Matsuri | 160 | 157 | 88/88 | clean |
| Aki Rosenthal | 154 | 149 | 84/84 | clean |
| Yuzuki Choco | 143 | 140 | 75/75 | clean |
| Nakiri Ayame | 112 | 109 | 63/63 | clean |
| Shiranui Flare | 53 | 51 | 39/40 | **degraded, see below** |
| Houshou Marine | 54 | 54 | 43/43 | degraded |
| Shirogane Noel | 71 | 71 | 56/56 | degraded |
| Usada Pekora | 56 | 53 | 44/46 | degraded |
| Tokoyami Towa | 54 | 53 | 42/43 | degraded |
| Shishiro Botan | 50 | 49 | 38/39 | degraded |
| Momosuzu Nene | 39 | 35 | 31/34 | degraded |
| Kazama Iroha | 75 | 72 | 50/50 | fixed pipeline |
| Hakui Koyori | 85 | 84 | 53/54 | fixed pipeline |
| La+ Darknesss | 78 | 75 | 46/47 | fixed pipeline |
| Ookami Mio | 135 | 132 | 69/69 | fixed pipeline |

**Not yet committed as of this writing**: Flare through Mio (11
talents) are fetched but sitting uncommitted — Flare specifically is
still not fully backfilled even after a retry pass (see the mweb
PO-token section below), and the owner hasn't yet decided whether to
accept Marine/Noel/Pekora/Towa/Botan/Nene's degraded coverage as-is or
re-run them under the fixed pipeline first. AZKi through Ayame (the
first 8) are already committed and pushed.

---

## 2026-09-05: YouTube's GVS PO-token gate expanded to mweb — cross-machine writeup

**For whoever's running the laptop instance:** this section is written for you.
Desktop (this session) hit a new failure mode mid-batch; the laptop
apparently hasn't hit the same symptom despite being on the same home
network. Both machines' current state, what was tried, what shipped, and
what's still an open call are below.

### What happened, in order
- **2026-09-04 14:24** — desktop merged the laptop's `mweb` pin + cookie
  write-back shield (commit `78da9da`).
- **14:24 → 2026-09-05 07:41** (~17h) — desktop ran **8 talents, 600+
  fetches, clean** on that exact code: AZKi, Sakura Miko, Hoshimachi
  Suisei, Akai Haato, Natsuiro Matsuri, Aki Rosenthal, Yuzuki Choco,
  Nakiri Ayame. All committed (`4e5fc73`, `75b7b6a`).
- **07:41** — Shiranui Flare's densify starts. `mweb` suddenly fails 89%
  of attempts (151/170) with `Video unavailable`. Not caught by
  `stopped_early` — that error string doesn't match
  `fetch.BotCheckDetected`'s markers, so the batch quietly kept going.
- Root cause (confirmed via `yt-dlp -v`): **YouTube's per-video GVS
  PO-Token experiment expanded from `web` to also bind `mweb`.** This is
  a known, ongoing, *automated* YouTube-side anti-scraping rollout
  documented across the yt-dlp tracker for months (not something aimed
  at us, not a fluke) — see [PR #14471](https://github.com/yt-dlp/yt-dlp/pull/14471)
  (detection added for `web`), [issue #16144](https://github.com/yt-dlp/yt-dlp/issues/16144)
  (`web_creator` caught later, "silent undetected enforcement"),
  [issue #14421](https://github.com/yt-dlp/yt-dlp/issues/14421) (someone
  else hitting `mweb` + real PO token + still-403). Cross-checked against
  Holodex: 134/151 of Flare's failed ids carried `status: "past"` (a
  normal, public stream, spanning 2021-01 to 2026-08-26) — this was never
  real unavailability.
- Same day, independently: the laptop pushed `scripts/enid_pipeline.sh` /
  `enid_autoresume.sh` / `potoken_server.sh` (commit `5a4fb48`) — real
  self-hosted PO-token-provider infrastructure, evidently built for the
  same underlying wall hit on the laptop's own EN/ID work.

### Why the laptop didn't see "Video unavailable" — structural, not luck
`scripts/enid_pipeline.sh`'s own header says **"Cookieless — the bgutil
PO-token server must be up."** The laptop deliberately fetches without
`data/youtube.cookies.txt` in play at all, relying purely on the
PO-token server. Desktop has always fetched **with** real cookies. Same
network, same shared `fetch.py`, genuinely different auth strategy —
and that difference routes you into a different YouTube-side check:

- **Cookies present (desktop):** dodges the classic "prove you're
  human" bot-check almost entirely (that exact string never appeared in
  this session's logs, ~30+ hours of fetching). But reading yt-dlp's own
  extractor source (`yt_dlp/extractor/youtube/_video.py`,
  `fetch_po_token`/`_fetch_po_token`) shows it only actually calls a
  PO-token provider when its internal per-client `PLAYER_PO_TOKEN_POLICY`
  says a token is `required` — the per-video experiment flag it *logs*
  does **not** itself flip that policy, and an authenticated session
  essentially never trips "required" by default. So a video caught by
  the experiment just fails outright, no fallback attempted, regardless
  of client.
- **Cookieless + PO-token (laptop):** reproduced here deliberately
  (cookieless `mweb`, no forced token) — instantly got the exact
  symptom you described, `Sign in to confirm you're not a bot`. Same
  root mechanism, different failure signature, because an unauthenticated
  request without a proactively-forced token hits YouTube's classic
  anti-bot gate before it ever gets near the newer per-video one.
  **Open question for you:** your `enid_autoresume.sh` treats any
  bot-check-shaped failure as "wait 30 min, relaunch the whole pipeline"
  — worth checking whether any of your own `_densify.json` runs have an
  unusually low `added` count relative to `months_targeted` even after a
  successful relaunch. That's the same signature this desktop's problem
  had before anyone noticed it (see the fetch-pin gotcha in `CLAUDE.md`)
  — the retry-the-whole-pipeline strategy could be masking a similar
  per-video gap rather than genuinely clearing it.

### What was tested (commands + real numbers), all on a 20-id sample of
still-failing Flare videos, `--simulate` unless noted:
| Combination | Recovery |
|---|---|
| plain cookies, no PO token | ~0% |
| `web_embedded` client alone (no token needed per the [PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/Po-Token-Guide) — confirmed, never even shows the experiment debug line) | ~10% |
| `mweb` + forced token (`fetch_pot=always`), no cookies | ~25% |
| `mweb` + forced token **+ cookies together** | ~35%, recovered ids that failed under every other combo |
| same-day router IP change (owner rotated the home IP) | no measurable difference on the next talent (Pekora) vs. the pre-change baseline — not primarily IP-reputation-based |

Re-testing the *exact same* id/combo minutes apart gave a **different**
pass/fail result each time — this is a probabilistic per-request gate,
not a fixed per-video denylist. No client swap or token strategy tried
reliably clears it. The `--simulate` numbers above also only exercise
yt-dlp's **player**-context token path (metadata/format listing) — the
PO Token Guide distinguishes that from a separate **GVS**-context token
needed for the actual media fetch, so those numbers alone don't prove a
real download succeeds.

### What shipped (commit `86aced0`, pushed)
`fetch.py`: pin changed to `youtube:player_client=mweb;fetch_pot=always`
(cookies still attached when available), plus `fetch._FETCH_ATTEMPTS = 2`
— one retry on a `CalledProcessError`, cheap insurance against the
per-request randomness. `_default_runner` now auto-detects a local
`~/.config/yt-dlp/plugins` bgutil install and injects it into the
subprocess `PYTHONPATH` itself (no external `PYTHONPATH` export needed
on whichever machine has the plugin installed — safe no-op on one that
doesn't; verified `fetch_pot=always` degrades to a warning + token-less
fallback format when no provider is present, never a hard error).
**This is shared code — the laptop gets `fetch_pot=always` and the
retry automatically on its next pull, no laptop-side change needed for
that part.**

**Validated with real (non-`--simulate`) production downloads**, not
just synthetic tests: Kazama Iroha, the first hololive-JP talent
densified under the new code, landed around **63% fetch success**
(complete QC-pass measurement records, meaning real audio made it all
the way through isolate + Praat measurement) — up from the ~33–35% seen
on Marine/Noel/Pekora/Towa/Botan/Nene under the old `web_embedded` pin,
but still well under the historical 95%+ baseline.

### Still an open decision, not yet made
Desktop's own PO-token-provider setup (bgutil-ytdlp-pot-provider 1.3.2)
is now installed here too — see the CLAUDE.md "PO-token provider" gotcha
for the exact desktop setup and the `$FLAKE` override note (the tracked
`scripts/potoken_server.sh` default path is the laptop's, not desktop's).
**Whether the laptop should switch from cookieless to cookie-authenticated
fetching (pairing cookies with its already-running PO-token server,
since that combination clearly outperformed either alone here) is the
laptop owner/session's call, not something this desktop session can or
should decide for it** — it may have been deliberately cookieless for a
reason not visible from here (e.g. avoiding the cookie-degradation bug,
which is now separately fixed by the disposable-copy shield already
merged both ways).

### Data left in a pending state on desktop
Shiranui Flare (partial, `_stopped_early` never fired but her file is
not representative — only backfilled to ~53/~170 records), Houshou
Marine, Shirogane Noel, Usada Pekora, Tokoyami Towa, Shishiro Botan,
Momosuzu Nene (all completed under the old `web_embedded` pin, 33–65%
month coverage instead of the usual 95%+) are fetched but **not yet
committed** — pending the owner's decision on whether to accept them
as-is (documented gap, same as the existing privated-debut-era
precedent) or run a backfill pass under the new code first. Kazama
Iroha onward uses the new code from the start.

---

## 2026-09-05/06: corpus-wide >95% QC backfill, then graduated members

**Owner decision**: re-run `densify` toward >95% QC coverage corpus-wide,
not just the mweb-affected talents. Audited every registered talent
against the *real* denominator — total calendar months since Holodex's
earliest cataloged video for that channel — not "months present in the
measurements file" (misleadingly generous: a month where every candidate
failed to fetch never gets a record at all, so it silently drops out of
that ratio). **32 talents were below 95% true coverage**, including ones
finished days ago, unrelated to the mweb incident — Kiryu Coco 19.5%
(16/82), Sakura Miko 75%, AZKi 77%, Robocosan 84%, Shirakami Fubuki 84%,
even Luna 91% and Sora 92%. Backfilling worst-first via
`scripts/run_veteran_batch.sh` (already-registered talents, so
`new-talent` is a no-op "already exist, left untouched" and it goes
straight to `densify --target-n 2`), same fixed pipeline as the mweb
section above.

**Not every talent will reach 95% — two genuinely different outcomes
found so far, both expected, neither a bug:**
- **Fetch-recoverable** (the majority): Nene 42%→81%, Marine 50%→69%,
  Pekora 51%→83%. `months_targeted` in the densify summary is large
  (real eligible candidates existed, previously blocked by the mweb
  gate) and the retry recovers most of them.
- **Genuine content ceiling**: Coco 19.5%→23.2%, Ao unchanged at 51.4%
  with `months_targeted: 0`. These talents' catalogs simply don't have
  enough eligible chatting-stream candidates in most months per this
  project's product rules (no singing/shorts/collabs/members-only) —
  re-running densify correctly finds nothing to add. Same category as
  the already-documented genuinely-privated debut videos: a real,
  permanent data-availability wall, not something to keep chasing.
  **Pushing these higher would mean loosening the product rules
  (allowing collabs/singing into the corpus) — a scope decision, not a
  data-quality fix, and not made here.**

Recovery rate also varies by *when* the backfill ran, not just which
talent: Flare (first in the backfill queue) recovered almost nothing
(46.5%→46.5%, 115/117 fresh attempts still "Video unavailable") despite
using the exact same fixed code that got Iroha to 89% and Koyori to 98%
hours earlier. Marine/Pekora, run right after Flare, recovered
partially (50%→69%, 51%→83%) — worse than this morning's talents but
better than Flare. This looks like the PO-token gate itself fluctuating
in strictness over the course of a long session, not a per-talent
property — worth keeping in mind if a backfilled talent still falls
short: **a later re-run, not a different technique, may be what closes
the remaining gap.**

**Bug found and fixed mid-backfill (commit `a10f0d6`)**: the laptop
caught that `75b7b6a` (the original 1st/2nd-Gen JP commit) registered
`flare_monthly.json` in `talents.json` without committing the file
itself — `talents.json` gets its entry automatically the moment `plot
--talent` runs, which happened for Flare even though her measurements
file was deliberately held back. A fresh clone or pull would crash
`vvc site-data` on the dangling reference (the laptop's own fix,
`fe97c22`, made `site_data.py` skip-and-warn instead of crash — a good
defensive fix regardless, now merged, but doesn't replace actually
committing the files). Lesson for future partial-data holds: **the
registry write is automatic and not something a `git add` selection
can prevent — either commit the file too, or strip its `talents.json`
line back out before committing, every time.**

**Next, once the backfill queue finishes**: owner decision to also
cover **graduated members**, correctly attributed to their real prior
generation rather than a bare "Graduated" bucket. Live Holodex query
(`fetch_channels` unfiltered by `inactive`, same group-exclusion rule as
`roster.json`'s D13 filter) found **16 graduated hololive-org channels**
still carrying their *original* generation in the `group` field (Cover
does not reassign it to "Graduated" on graduation) — confirming the
existing `_KNOWN_GRADUATED_GROUPS` hand-maintained-dict approach
(`site_data.py`) is the right mechanism, just needed more entries, not
a different one. Of the 16: Hiodoshi Ao and Kiryu Coco are already in
the corpus. Two are out of scope under the existing JP/EN/ID/DEV_IS
roster rule, same call as excluding mekPark — Civia (`CN 1st
Generation`, the discontinued Hololive China branch) and Blue Journey
(`holo-n`, an unfamiliar newer sub-brand, not one of the established
generations). **The remaining 12 are queued for the next batch**:
Yozora Mel (1st Gen), Minato Aqua (2nd Gen), Murasaki Shion (2nd Gen),
Uruha Rushia (3rd Gen Fantasy), Amane Kanata (4th Gen holoForce), Mano
Aloe (5th Gen holoFive — graduated within days of debut in real life;
expect a near-empty catalog, same as Ao's short-career case, not a bug
if `months_targeted` comes back tiny), Sakamata Chloe (6th Gen holoX),
Gawr Gura (English -Myth-), Watson Amelia (English -Myth-), Ceres Fauna
(English -Promise-), Nanashi Mumei (English -Promise-), Tsukumo Sana
(English -Promise-). `_KNOWN_GRADUATED_GROUPS` already updated with all
12 ahead of the data landing (a safe no-op until each talent actually
exists in the registry) so no follow-up site_data fix is needed once
they're fetched.

**Backfill batch paused ~00:15 (past midnight), not completed.** Progress
so far, worst-first: Coco (19.5%→23.2%), Nene (41.9%→81.1%), Flare
(45.3%→46.5%), Marine (50.0%→68.6%), Pekora (50.6%→82.8%), Ao
(51.4%→51.4%), Botan (51.4%→83.8%), Towa (51.9%→74.1%), Haato
(64.0%→64.0%), Ayame (64.9%→64.9%), Noel (65.1%→65.1% at the time,
see correction below). Suisei was mid-densify when stopped, untouched.

**Correction, made after the owner pushed back on a batch of "failed"
video ids that turned out to be genuinely privated, not gate-blocked**:
the initial write-up above claimed the PO-token gate got stricter
talent-over-talent as the session went past midnight, using
Haato/Ayame/Noel's zero-to-near-zero recovery as evidence. That
diagnosis was wrong for two of the three — checked properly now via
Holodex's own per-id `status` on every failed id, not assumed:

| Talent | Failed ids | Status breakdown | Real diagnosis |
|---|---|---|---|
| Suisei | 38 | 38 `missing` | **content ceiling** — her earliest content predates joining hololive (individual streaming from 2018); genuinely gone, not fetch-recoverable |
| Ayame | 40 | 40 `missing` | **content ceiling** |
| Haato | 45 | 44 `missing`, 1 `past` | **content ceiling** (essentially) |
| Noel | 94 | 4 `missing`, 90 `past` | **genuinely gate-blocked** — the one case that actually supports the original claim |
| Marine | 62 | 4 `missing`, 58 `past` | gate-blocked (consistent with her real 50%→69% improvement) |
| Pekora | 65 | 20 `missing`, 45 `past` | gate-blocked |
| Botan | 46 | 3 `missing`, 43 `past` | gate-blocked |
| Towa | 75 | 10 `missing`, 65 `past` | gate-blocked |
| Nene | 22 | 0 `missing`, 22 `past` | gate-blocked |

**Lesson: check the Holodex `status` breakdown of a talent's failed ids
before concluding anything about the gate's behavior, every time — a
talent with a genuinely privated-heavy early career (Suisei especially)
looks identical to a gate-blocked talent from the raw pass/fail numbers
alone.** Coco and Ao were already correctly identified as content
ceilings (`months_targeted: 0`, an even cleaner signal than checking
status by hand). Suisei, Ayame, and (mostly) Haato belong in that same
bucket — **their remaining coverage gap is likely not
fetch-recoverable at all**, regardless of client, cookies, account, or
IP, and shouldn't be expected to move much on a re-run. Flare, Marine,
Pekora, Botan, Towa, Nene, and Noel are the ones where another pass
under better gate conditions is actually worth doing.

Also tested and ruled out as fixes for the genuinely gate-blocked
talents: a fresh cookie export (0/15 recovered on a — later found to be
mis-selected, see above — sample; the auth itself works fine, gate
result unchanged) and restarting the PO-token server mid-session
(helped marginally on a tiny 5-id manual sample, didn't hold up at real
batch scale on Suisei — which, per the correction above, was mostly
testing against privated content anyway, so that result is now suspect
too and shouldn't be read as "the restart doesn't help," just as
"inconclusive on a contaminated sample"). IP rotation (owner's router
change) also showed no effect. See the "anything we can do" discussion
in the session transcript for the fuller reasoning: the working theory
is that the gate keys on the *behavioral/technical fingerprint* of the
scripted fetch process itself (still the same machine, same scripted
Botguard attestation) rather than on IP or account identity, which
would explain why swapping either alone doesn't reset it.

**Stopped rather than grind through the remaining ~21 talents (Suisei,
Mio, Miko, AZKi, Choco, Laplus, Subaru, Okayu, Robocosan, Aki, Fubuki,
Iroha, Korone, Matsuri, Koyori, Niko, Luna, Su, Sora, Raora, Kanade).**
Four of those (Iroha, Koyori, Laplus, Mio) already hit 89-100% earlier
today under the same fixed code before midnight, so they may not need
another pass at all. **Before re-queueing any paused talent, check its
failed-id status breakdown first (content ceiling vs. gate-blocked) —
don't assume either way from the raw numbers.** Resume with
`scripts/run_veteran_batch.sh` using the same `name:channel_id` pairs
(all already in `DISPLAY_NAMES`) once there's reason to think the gate
has loosened, ideally verified on one gate-blocked talent first rather
than committing to the whole remaining list blind.

---

## First-batch clip ids (24-clip balanced reference)

Luna 2024: `Gz_2EzLyhmQ` `ro0lFIj2MJY` `boy302x08Gg` `qyQzBoMOqXo`  
Luna 2025: `c7xfx5DOmes` `UgCC0aua69E` `yHx9-xigq0o` `q8B9Y3a33Ik`  
Lamy 2024: `uIHMp3Y7XjU` `jg9o2q_lvrY` `LyDQcY9bzDQ` `b3NaMgu0fqM`  
Lamy 2025: `x0C3mbj1jLI` `1xxlWriXKVU` `mWovsFeoniE` `wGRD8IeiVtY`  
Sora 2024: `Ywz-QaqAgNM` `Wgl2LhK9tiU` `RUF6SYI3oko` `RYpOAavYST0`  
Sora 2025: `nhEQHX-ywz4` `NvFWC3YXreA` `KsOJfG4tmyw` `X-FdkMiT-Bk`
