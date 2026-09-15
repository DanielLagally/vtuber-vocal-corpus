# v1 methodology

Exactly what the v1 pipeline did, as it is frozen. This document is
self-contained: it describes v1's own rules and parameter values, not the
project's current ones.

v1 is frozen but not abandoned — it remains the published site, and it stays
regenerable and fixable. Its known defects are documented separately in
[`LIMITATIONS.md`](LIMITATIONS.md), which you should read alongside this.

The current project methodology lives in [`../../reference/`](../../reference/README.md)
and differs from v1 in several places. Where they disagree about v1, this
document is correct.

## Corpus

Active hololive **talent** channels — JP, EN, ID, DEV_IS — plus graduated
members whose archives remain reachable. Holodex `Official` and `Misc` groups
and anything with HOLOSTARS in the group name are excluded.

One JSON file per talent at `data/measurements/<talent>_monthly.json`; the
registry of display names, branches, and generations is `talents.json`. The site
export `data.js` is generated from these by `vvc site-data`.

## Stream eligibility

A video was rejected outright when any of these held:

- `type` was not `stream`
- `topic_id` was one of `singing`, `shorts`, `Original_Song`, `Music_Cover`,
  `Teaser`, `membersonly`
- `mentions` was non-empty — Holodex's collab marker, which is only populated
  when the request passes `include=mentions`
- duration was under 900 s
- the title contained `karaoke`, `歌枠`, `ゲスト`, or `guest`

## Stream selection

Score = topic term + duration term:

| Topic | Points | | Duration | Points |
|---|---|---|---|---|
| `talk` / `chatting` / `morning` | 40 | | 30–180 min | +30 |
| unlabelled (`topic_id` absent) | 20 | | over 180 min | +20 |
| any other topic | 10 | | under 30 min | +10 |
| `watchalong` | 0 | | | |

Highest score wins; ties break to the newest `available_at`. **Topic was a
preference, not a gate** — game streams and watchalongs were eligible and were
selected when a month offered nothing better. `catalog.reliability_band`
classified these as lower reliability but was never called by anything, and
`topic_id` was not stored on the record, so the stream type of a v1 clip cannot
be recovered from the corpus.

Sampling density was monthly, targeting two clips per month (`densify
--target-n 2`). An earlier rule of four streams per year, one per quarter, was
superseded before the corpus was built.

## Audio preparation

1. **Fetch** — the 15:00–30:00 section of the stream, skipping the waiting-room
   intro. Downloaded in full via range requests, then cut locally with `ffmpeg`.
2. **Window** — a 90 s window chosen from a 15 s grid over the fetched audio,
   scored on `(voiced_fraction, -f0_iqr)` compared lexicographically. Scoring
   used the numpy autocorrelation tracker on the **raw, unseparated** audio,
   resampled to 16 kHz. Because `voiced_fraction` is a continuous float, exact
   ties essentially never occurred, so **the IQR term never influenced the
   choice** — selection maximised voiced fraction alone.
3. **Isolate** — the 90 s window was passed through the RoFormer separator
   `bs_roformer_vocals_resurrection_unwa.ckpt`, unconditionally, for every clip.
   Uniformity was the goal: the same transform everywhere, so year-over-year
   comparison would not be "separated versus raw".

Measurement then ran on the whole separated stem. The `window` offsets stored on
a record describe where the clip came from in the fetched file; they were not
re-applied to the stem.

## Features

All pitch and voice-quality features came from Praat via `parselmouth`
(`praat_features.py`); spectral and amplitude features from numpy
(`features.py`). Praat pitch parameters throughout: **autocorrelation
(`To Pitch (ac)`), floor 75 Hz, ceiling 600 Hz, time step 0.02 s.**

| Feature | Definition |
|---|---|
| `median_f0` | Median F0 of voiced frames. No voiced frames → `nan` |
| `f0_iqr` | Q3 − Q1 of voiced-frame F0, in **Hz** |
| `voiced_fraction` | Share of all frames the tracker called voiced |
| `dynamism_semitones` | Mean absolute semitone change between **consecutive voiced frames**; a pause contributes nothing rather than a fabricated jump |
| `jitter_local` | Praat local jitter via `To PointProcess (cc)`, period bounds 0.0001–0.02 s, max period factor 1.3 |
| `shimmer_local` | Praat local shimmer, same bounds plus max amplitude factor 1.6 |
| `hnr_db` | Mean of cross-correlation harmonicity, time step 0.01 s, minimum pitch 75 Hz, whole file |
| `brightness_hz` | Spectral centroid: mono at 16 kHz, 50 ms Hann frames, 25 ms hop, magnitude-weighted mean frequency per frame, unweighted mean over frames with non-zero energy |
| `loudness_dynamics_db` | Standard deviation of per-frame RMS in dB, frames above 1e-6 |
| `f1_hz` … `f4_hz` | **Mean** F1–F4 from Praat's Burg method: time step 0.02 s, 5 formants, ceiling 5500 Hz, 25 ms window, 50 Hz pre-emphasis, averaged over every frame returning a finite value |

The formant row is the one to read carefully: **v1 applied no voiced-frame
mask.** See [`LIMITATIONS.md`](LIMITATIONS.md).

## Quality control

`qc.qc_verdict` evaluated four gates in order, first failure deciding:

| Order | Condition | Reason code |
|---|---|---|
| 1 | `median_f0` missing or non-finite | `f0_missing` |
| 2 | `voiced_fraction` < 0.15 | `voiced_fraction` |
| 3 | `f0_iqr` ≥ 200 Hz | `f0_iqr` |
| 4 | `median_f0` ≥ 600 Hz | `f0_high` |

The voiced-fraction floor exists because a clip with almost no voiced content
can post a deceptively *tight* IQR purely from having too few points to spread
across. The threshold sits in a natural gap in the observed distribution.

The IQR ceiling is the main junk flag — background music, a followed melody, a
second speaker, or octave errors all widen within-clip spread beyond what one
person talking produces.

Failing clips are **gaps, never zeros**, everywhere in the pipeline.

`qc.requalify` recomputes stored verdicts from stored features without re-reading
audio. Plots called `qc_verdict` live; the site export read the **stored**
verdict instead, so the two could drift until a requalify pass ran.

## Aggregation

- **Monthly**: mean of that month's QC-passing clip medians.
- **Quarterly**: mean of the quarter's values.
- **Yearly**: `statistics.median_low` over **clip-level** values pooled across
  the year, plus raw min and max and a clip count.

The yearly level therefore differs from the others in two ways: a lower median
rather than a mean, and clip-level rather than month-level pooling.

## Site

`docs/v1/` is plain HTML/CSS/JS with Plotly vendored locally and no build step.
Data loads as `window.SITE_DATA = {...}` rather than `fetch()`, because `fetch()`
is CORS-blocked under `file://`.

Views: time series, cute × mature scatter, percentile radar, trajectory, sortable
table, group comparison. The radar carries 11 yearly metrics, each shown as that
talent's rank percentile against the whole registered corpus, computed
independently per axis — it is not a combined score.

The **cute/mature percentile** is computed in `site_data.py` as: per talent, the
unweighted mean of all QC-passing clip values for `median_f0`, `brightness_hz`,
and `dynamism_semitones`; then a population z-score per axis across talents;
then an equal-weight (1/3 each) sum; then a rank percentile,
`100 × rank / (n − 1)`. A talent missing any one axis is dropped from the
scatter entirely. The site captions it "acoustic correlates, not a vibe rating".
