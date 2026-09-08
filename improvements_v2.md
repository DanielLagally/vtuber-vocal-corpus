• ## Bottom line

  This is a strong engineering prototype and a useful exploratory corpus. The pitch
  data supports real descriptive findings. It is not yet a scientifically validated
  voice-change dataset, and several secondary metrics—especially the formants—
  currently deserve less confidence than the site gives them.

  My honest assessment:

  - Engineering and traceability: good.
  - Aggregate F0 comparisons: usable with caveats.
  - Individual monthly F0 points: noisy.
  - Long-term F0 trends: credible as trends in the sampled stream voice, not proven
    physiological change.

  - Brightness: exploratory.
  - Jitter, shimmer, and HNR: weak as personal voice traits in this material.
  - Current F1–F4 results: not usable for their advertised interpretation until
    corrected.

  - “Cute/mature” percentile: an arbitrary exploratory index, not a validated
    measurement.

  ## What the project is doing

  The intended pipeline is sensible:

  1. Select eligible streams deterministically.
  2. Extract a fixed 15–30-minute portion.
  3. Find a 90-second voice-heavy window.
  4. Apply the same RoFormer vocal separator.
  5. Measure F0 and other acoustic properties with Praat.
  6. Apply centralized QC.
  7. Aggregate by month, quarter, year, talent, and group.

  That uniformity is a genuine strength. The current corpus has:

  - 55 registered talent/channel series
  - 5,881 measured clips
  - 5,688 QC passes, or 96.7%
  - No duplicate video IDs
  - No disagreement between stored and recomputed QC
  - One current tracker and one isolation model throughout
  - 288 passing tests

  The 96.7% figure is measurement QC, not career coverage: completely unfetched or
  ineligible months never enter that denominator. Within each file’s observed first-
  to-last month span, median coverage is about 97.5%, but it is only 52% for Flare
  and about 69% for Marine and Noel.

  The implementation is unusually careful about gaps, provenance, snapshots, retry
  behavior, and not silently zero-filling failures. The centralized QC in src/vvc/
  qc.py:40 and deterministic selection in src/vvc/catalog.py:46 are both good
  choices.

  ## Do the results look realistic?

  Mostly, at the aggregate level.

  Across talent-level, equal-month averages:

  - Typical F0 ranges from about 210 to 391 Hz.
  - Brightness ranges from about 1,812 to 2,628 Hz.
  - Voiced fraction centers around 48%.
  - The central ranges for HNR, jitter, shimmer, and the formants are numerically
    plausible.

  But plausible values are not the same as validated measurements.

  There are clear residual F0 errors among QC passes:

  - 14 passing clips are below 150 Hz.
  - Three are approximately 88, 100, and 102 Hz.
  - Most of these have a same-talent, same-month counterpart around 200–300 Hz.
  - Luna’s documented 152 Hz “legitimate low outlier” shares its month with a 368 Hz
    clip. Its frame track ranges from roughly 75 to 599 Hz and is strongly
    multimodal. Calling that legitimate from its aggregate IQR and voiced fraction is
    not defensible; it should be marked for review.

  This is consistent with octave errors. The code uses Praat’s raw autocorrelation
  path in src/vvc/praat_features.py:16. Praat now recommends filtered autocorrelation
  for F0 and intonation, specifically because it reduces octave rises and drops; its
  documented gold-standard comparison found gross errors more than three times lower
  than raw AC. Praat pitch-method guidance
  (https://praat.org/manual/how_to_choose_a_pitch_analysis_method.html)

  So the F0 corpus is good enough for averages and sustained patterns, but the QC is
  not sufficient to trust every passing clip.

  ## Notable trends

  The strongest corpus-level result is widespread pitch lowering over career time.

  Among 48 talents with at least 24 covered months and two years of span:

  - 35 have negative F0 slopes.
  - The median slope is −5.15 Hz/year.
  - The median change between the first and last 12 measured months is −20.2 Hz.
  - A robust Theil–Sen analysis gives a median slope of −4.51 Hz/year.
  - First- and second-ranked stream slopes correlate at 0.73 and agree in direction
    for 39 of 45 testable talents.

  That replication makes the broad trend difficult to dismiss as a single-stream
  sampling accident. It still cannot distinguish character-performance change from
  microphone, content, vocal-isolation, or streaming-format changes.

  Standouts include:

   Talent                    F0 slope      R²    First vs last 12 months
  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━
   Oozora Subaru        −15.8 Hz/year    0.77                   −75.8 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   Hakui Koyori                 −14.0    0.40                   −49.7 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   Hoshimachi Suisei            −12.7    0.39                   −81.2 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   Nakiri Ayame                 −11.7    0.40                   −42.1 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   Himemori Luna                −7.71    0.28                   −42.0 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   AZKi                         +10.1    0.24                   +97.1 Hz
  ───────────────────  ───────────────  ──────  ─────────────────────────
   FUWAMOCO                     +28.0    0.53                   +59.9 Hz

  Luna’s annual monthly-median sequence is approximately 367, 342, 350, 336, 328,
  324, and 309 Hz from 2020 through partial 2026. The downward pattern is real within
  this pipeline. The claim in PLAN.md:123 should nevertheless say “a robust downward
  trend in sampled stream F0,” rather than implying confirmed underlying voice
  change.

  FUWAMOCO must not be interpreted as one speaker: it is a two-person channel, so its
  apparent rise may reflect changes in who spoke more.

  There are also broad increases in jitter and decreases in F4 across many talents.
  That kind of corpus-wide synchronization is suspicious for a recording/source-era
  effect, not dozens of independent biological changes.

  ## Correlation structure

  The data naturally falls into several correlated families:

   Relationship              Between talents     Within a talent    Interpretation
                                                     over months
  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   F2 ↔ F3                             +0.77               +0.81    One resonance/
                                                                    spectral family
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   F1 ↔ F2                             +0.65               +0.78    Same; four
                                                                    separate formant
                                                                    columns are
                                                                    redundant
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   Shimmer ↔ HNR                       −0.61               −0.72    Shared
                                                                    periodicity/
                                                                    noisiness factor
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   Jitter ↔ HNR                        −0.58               −0.49    Same factor
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   F0 ↔ F0 IQR                         +0.69               +0.58    Hz IQR is
                                                                    inherently
                                                                    pitch-scale
                                                                    dependent
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   F0 IQR ↔ dynamism                   +0.59               +0.44    Dynamism partly
                                                                    repeats pitch-
                                                                    track
                                                                    variability
  ────────────────────────  ─────────────────  ──────────────────  ──────────────────
   Voiced fraction ↔                   −0.58               −0.47    Likely context/
   loudness dynamics                                                window/capture
                                                                    behavior

  This means the 11-axis radar is presenting fewer than 11 independent things. It
  visually double- or triple-counts some phenomena.

  The F0–IQR relationship also exposes a QC problem: a fixed 200 Hz IQR threshold
  treats the same musical variability differently for high- and low-pitched voices.
  Store F0 quartiles and use semitone IQR, or at least IQR / median, for cross-talent
  QC.

  ## The most serious implementation issue

  The current formants do not implement the stated voiced-speech measurement.

  formants_hz() (src/vvc/praat_features.py:129) computes Burg formants over every
  frame where Praat returns a finite number. Praat commonly returns formants for
  unvoiced/noisy frames too; the code never applies a voiced-frame mask.

  I rechecked 12 real stems:

  - Unvoiced frames made up roughly 12–66% of the frames contributing formant values.
  - Restricting to voiced frames shifted median F1 by −32 Hz, F2 by −71, F3 by −62,
    and F4 by −33.

  - In the worst sampled clip, F1 changed by −433 Hz.

  Therefore the current F1–F4 numbers should be removed or visibly marked
  experimental until remeasured. Even after voiced gating, averaging formants across
  arbitrary vowels does not isolate vocal-tract length: vowel composition and the
  speaker-specific formant ceiling matter. Praat explicitly says the ceiling must
  suit the speaker and vowel. Praat Burg-formant documentation
  (https://praat.org/manual/Sound__To_Formant__burg____.html) Historical recording
  quality can also materially alter LPC formants. Recording-quality/formant study
  (https://kar.kent.ac.uk/58978/)

  Jitter and shimmer have a similar presentation problem. The code itself
  acknowledges that they are designed for sustained vowels, but the site captions
  omit that warning. Praat states that jitter is typically measured on long sustained
  vowels. Praat jitter documentation (https://praat.org/manual/Voice_2__Jitter.html)

  ## Methodological weaknesses to fix

  - Window selection is biased toward continuous, easily tracked speech. In src/vvc/
    windows.py:58, (voiced_fraction, -IQR) is lexicographic: voiced fraction
    dominates and IQR is effectively only a tie-breaker. Voiced fraction and
    “dynamism” therefore should not be presented as natural behavioral properties.

  - Selection is deterministic but not random. Choosing the highest-scored and then
    newest stream can correlate content and career time.

  - Metadata filters cannot catch unannounced guests, game voices, alerts, or
    residual music. There is no speaker-identity verification.

  - QC validates F0 plausibility, then implicitly blesses every spectral and voice-
    quality feature. A clean pitch track does not prove clean formants or brightness.

  - The yearly lower median in src/vvc/series.py:233 introduces a small needless
    downward bias. Use the normal median.

  - Min–max is not uncertainty and expands with sample size. Show robust spread and
    confidence separately.

  - The “cute/mature” score in src/vvc/site_data.py:240 equally combines three
    unvalidated, partly correlated axes. The caveat is honest, but the title still
    promises more than the calculation establishes.

  - The protocol documentation has drifted: README says chatting only, code permits
    game streams and fallback watchalongs; PLAN says four streams/year in one place
    while current production uses roughly two/month; PLAN says Android fetching while
    src/vvc/fetch.py:137 uses mweb;fetch_pot=always.

  - The README says audio is not stored, although substantial audio is stored
    locally/offloaded. It should say “not published or committed.”

  - The generated site is currently five Noel records behind the measurement file: 71
    versus 76. That is understandable during processing, but snapshots need an                  
    explicit release identifier.                                                                
                                                                                                
  ## How I would flatten and present it                                                         
                                                                                                
  Keep four non-destructive tables:                                                             
                                                                                                
  1. clips.csv: one row per measurement, including video ID, topic, stream date,                
     selection score, window, tracker, model, every feature, QC result, and review              
     flags.                                                                                     
                                                                                                
  2. talent_month.csv: one row per talent/month with attempted, fetched, passed, n,             
     month-equal metric aggregates, replicate difference, and missingness reason.               
                                                                                                
  3. talent_summary.csv: one row per talent with eligible-month coverage, typical               
     value, robust spread, percentile, Theil–Sen trend, first/last-12 change,                   
     replicate reliability, and warning count.                                                  
                                                                                                
  4. correlations.csv: one row per metric pair with both between-talent and within-             
     talent correlations. Never publish only a pooled correlation.                              
                                                                                                
  For humans, default to five groups rather than eleven raw axes:                               
                                                                                                
  - Pitch level: F0, corpus percentile, change over time.                                       
  - Pitch movement: semitone spread and smoothed contour movement.                              
  - Periodicity/noisiness: jitter, shimmer, HNR, explicitly experimental.                       
  - Spectrum/resonance: brightness and corrected formant summary.                               
  - Context/measurement quality: voiced fraction, loudness dynamics, coverage,                  
    replicate agreement.                                                                        
                                                                                                
  The default talent card should answer:                                                        
                                                                                                
  > Typical sampled pitch: 335 Hz, 78th corpus percentile.                                      
  > Trend: −7.7 Hz/year; first-to-last 12-month change −42 Hz.                                  
  > Coverage: 74/75 months; 136 passing clips.                                                  
  > Typical same-month clip disagreement: 26 Hz.                                                
  > Confidence: useful long-run trend, noisy individual months.                                 
                                                                                                
  That gives a person a conclusion without hiding the measurements that produced it.            
                                                                                                
  I would prioritize: temporarily demote formants, test filtered autocorrelation on a           
  stratified real-audio sample, add review flags for octave-discordant same-month               
  clips, then produce the four flattened exports and a correlation heatmap. I made no           
  repository changes during this evaluation.
