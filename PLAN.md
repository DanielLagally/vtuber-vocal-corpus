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
5. Measure **the whole 90 s stem** (do not re-apply `windows.json` offsets onto the stem — those times are for the 15 min wav). Persist `window` as metadata.
6. Plots from measurements JSON.

**Intended next on the same audio (approved, not implemented):** if the stem fails QC, hunt a **2nd-best** 90 s on the same raw 15 min that **does not overlap** the first window, isolate as `_raw90b` with the **same** RoFormer. QC plot: **keep 2nd only if it passes**; else gap. All-clips may still show the fail. No new YouTube.

Do not isolate 15 min `vocal_balanced` for new talents unless a new comparison says we must.

---

## What is already done

### Code (beyond the 24-clip era)
- Catalog: `pick_monthly`, `catalog --channel` + `--monthly` (Holodex `channel_id`, `include=mentions`).
- Fetch: `player_client=android`, `fetch_audio_many` skip-and-continue, skip existing wavs.
- Isolate: `model_filename` single-model path; `--windowed` reads `data/windows/<id>_raw90.wav`.
- Window CLI: `best_speech_window` + `slice_wav`.
- Measure/plot CLI exist; **measure still looks up `<id>_(vocals)_…` and used to re-slice the stem with 15 min offsets** — that crashed on the 90 s files. Luna stem plots were produced by a one-off that measured `*_raw90_(vocals)_*` as whole files. **Fix measure lookup + no second slice** as part of the 2nd-window work. `second_speech_window` / `retry` CLI: not implemented yet.
- Product tests: catalog monthly pick, isolate single-model, slice_wav, series/QC, measure shape, fetch android + skip-continue.

### Data — 24-clip batch (unchanged)
- 24 wavs, 24 full `vocal_balanced` stems, 90 s slices, fast 90 s stems, first plot set in `data/plots/`.
- Ids at the bottom of this file.

### Data — Luna monthly
- Catalog: `data/catalog/luna_monthly.json` — **75 months**, 2020-06 through 2026-08, contiguous, 1 pick each. All 8 first-batch Luna ids are in this set (remeasured on RoFormer 90 s, not the old 15 min balanced numbers).
- 75 × 15 min wavs in `data/audio/` (91 wavs total including Lamy/Sora).
- 75 × `_raw90.wav` + 75 × RoFormer 90 s vocals in `data/stems_fast/`.
- Measurements: `data/measurements/luna_monthly.json` (stems), `luna_monthly_raw.json` (unisolated 90 s).
- Plots: `data/plots/luna_monthly/` (monthly F0 + IQR), `luna_monthly_raw/`, `luna_quarterly/` (`f0_quarterly_all.png`, `f0_quarterly_qc.png`) — quarterly is the readable view.

### Luna monthly numbers (stem 90 s, first window only)
- 75 clips; **34 fail** IQR ≥ 200 or nan F0; **41 pass**.
- QC-pass F0: mean **~345 Hz**, sd **~36 Hz**, range ~250–440. Year means after 2020 sit in **~330–360 Hz**. 2020 is higher (~393) but n=3. 2024 QC-pass n=2.
- 6 stems sit on the **~615 Hz** tracker cap (all already IQR-fail).
- |stem − raw| median F0: **~24 Hz**; 38/74 differ by >20 Hz. Isolation moves some clips hundreds of Hz (octave / leftover BGM), so stem is the series to trust **when it passes QC**.
- Quarterly QC: 26 quarters (2020-Q2–2026-Q3); **11 quarters n=1**; **3 empty** (2024-Q3, 2024-Q4, 2025-Q3). All-clips bands inflated by 615 Hz junk.
- **Reading:** not a clear pitch-over-career slope. Consistent with a stable ~350 Hz character voice plus tracker/window noise. Monthly plots were too noisy; quarterly mean + min–max is the graph that is worth looking at.

### 24-clip comparison / plots (still on disk)
Old QC on those PNGs marked IQR ≥ 200 **or** F0 ≥ 500. Illustrative 2025−2024 balanced 90 s: Luna −24 / −59 after QC (almost no 2024 left); Lamy −25 / −34; Sora +5 / +27. **QC flips Sora’s sign** — n=4 with this much junk is not a voice-change claim.

---

## Why the Luna series disappointed (and what we are not doing yet)

Window hunt on **raw** (high voiced + low IQR) can lock onto periodic BGM; RoFormer then leaves melody (cap) or jumps octave. 45% fail IQR, so 1 stream/month cannot fill quarterly QC. Numpy ACF is a **failure detector**, not a career-F0 meter. Praat later; it will not fix a window on a jingle.

**Levers discussed; only the first is approved next:**
1. **Cap + 2nd non-overlapping raw window on existing 15 min wavs** (no new downloads). ← next
2. More streams per month (new fetches) — later if 2024 stays empty.
3. Hunt 90 s on the stem (more GPU) — only if retries still pick music.
4. Stop chasing a slope (~350 Hz + no trend is the result) — product call after (1).

---

## Operator notes

- Holodex: `User-Agent: vanalysis/0.1` or 403; `mentions` only with `include=mentions`; `topic_id == singing` is rare (songs are `Original_Song` / `Music_Cover`); many recent items are `shorts`. Rate-limit folklore is 80/window; official docs do not state a number. Key in `.env` only.
- Isolation: always the same processing per talent so year-over-year is not “Demucs vs raw.”
- `vocal_balanced` is lunalearn’s Anki preset. For median F0 we need BGM down, not prettier stems.
- First 15:00 of a stream is usually intro — never sample 0:00–15:00.
- Long GPU / yt-dlp: `nohup` / CLI, don’t block chat on a Task. Cancelled subagents can leave `audio-separator` running.
- nixpkgs `cudaSupport = true` + torch/onnxruntime in `withPackages` is a disk bomb. Pip GPU wheels ~6 GB. `unset PYTHONPATH` before venv pip (yt-dlp leaks 3.14 wheels).

---

## Next steps (in order)

1. **2nd-window retry (approved, not built):** `second_speech_window` (same 90 s / 15 s grid, no overlap with first). Slice `_raw90b`, isolate same RoFormer. Retry ids where IQR ≥ 200 or F0 nan or F0 ≥ 600 (~34). QC plot: replace month only if 2nd **passes**; else gap. All-clips may show the fail. Fix measure: find `*_raw90_(vocals)_*` (and test name `<id>_(vocals)_*`), measure whole stem, persist `windows.json` as metadata only. Snapshot `luna_monthly.json` before replace. Replot quarterly mean + min–max.
2. Look at post-retry quarterly QC. If 2024 still empty / still no trend → decide lever 2 (more streams/month), 3 (hunt on stem), or 4 (stop).
3. **Aggregate helper in-repo** so plots are not throwaway scripts; then **static site** (F0-over-years, profile, F0×brightness + “correlates not vibes”). Generated from aggregates only.
4. **Other talents only after Luna data is actually good.** Fast path (90 s RoFormer), one person at a time, until every in-scope member is high-quality and complete.
5. **Optional later:** Praat F0 (or whatever tracker best serves the data); audio-only yt-dlp when a real `m4a`/`webm` exists.

Do not: kill/replace the 24 `balanced` 15 min stems; mix presets inside one talent; commit `data/`; download Cover audio into git.

---

## First-batch clip ids (24-clip balanced reference)

Luna 2024: `Gz_2EzLyhmQ` `ro0lFIj2MJY` `boy302x08Gg` `qyQzBoMOqXo`  
Luna 2025: `c7xfx5DOmes` `UgCC0aua69E` `yHx9-xigq0o` `q8B9Y3a33Ik`  
Lamy 2024: `uIHMp3Y7XjU` `jg9o2q_lvrY` `LyDQcY9bzDQ` `b3NaMgu0fqM`  
Lamy 2025: `x0C3mbj1jLI` `1xxlWriXKVU` `mWovsFeoniE` `wGRD8IeiVtY`  
Sora 2024: `Ywz-QaqAgNM` `Wgl2LhK9tiU` `RUF6SYI3oko` `RYpOAavYST0`  
Sora 2025: `nhEQHX-ywz4` `NvFWC3YXreA` `KsOJfG4tmyw` `X-FdkMiT-Bk`
