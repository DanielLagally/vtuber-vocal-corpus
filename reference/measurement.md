# Measurement

Every acoustic feature: what it is, how it is computed, and what it is valid
for. Implementation: `praat_features.py` (pitch and voice quality, via Praat)
and `features.py` (spectral and amplitude, pure numpy/FFT).

Praat is the standard phonetics analysis program; the project drives it from
Python. Its pitch algorithms are named after the signal-processing method they
use — autocorrelation, cross-correlation, and (in newer releases) a filtered
autocorrelation variant intended to reduce octave errors.

## The validity question, stated once

Every number here is measured on audio that has been through a lossy chain:
YouTube's encoder, a fixed-length window chosen by a biased rule, and — for v1,
unconditionally — a neural vocal separator. **A feature is only as meaningful as
its invariance to that chain.** Features differ enormously in this respect, and
that difference matters more than any of their absolute values:

| Feature | Robust to separation | Notes |
|---|---|---|
| Median F0 | Yes — shifts by ~2 Hz median | The one feature that survives the chain cleanly |
| F0 IQR, dynamism | Mostly, but inherits any octave error | Scale-dependent; see below |
| Voiced fraction | **No** | Separation changes it materially, and selection ran on unseparated audio |
| HNR | **No** — systematically inflated | The separator removes noise, so harmonicity rises by construction |
| Jitter, shimmer | **No** | Also calibrated for sustained vowels, not conversational speech |
| Brightness | **No** | Separation reshapes the spectrum directly |
| Formants F1–F4 | **No** — shifts are large and clip-dependent | |

Everything below inherits this table. It is the reason the site presents pitch
as a result and the rest as exploratory.

## Pitch

**Median F0 (Hz)** — the median of voiced frames' fundamental frequency across
the measurement window. The typical speaking pitch of the clip. A window with no
voiced frames yields `nan`, never an invented value.

**F0 IQR** — the interquartile range of voiced-frame F0 within the clip: the
static spread of pitch. Reported in Hz historically; Hz IQR is inherently
pitch-scale dependent, so a high-pitched voice and a low-pitched voice with the
same *musical* variability produce different Hz values. A semitone-based spread
is the scale-free form and is what a cross-talent comparison requires.

**Voiced fraction** — the share of frames the tracker considers periodic. Read
[`sampling.md`](sampling.md) before using it: the window is chosen to maximise
this quantity, so it describes the selection rule at least as much as the
speaker.

**Dynamism (semitones)** — the mean absolute semitone change between
*consecutive voiced frames*. A frame pair contributes only when both frames are
voiced, so a pause never fabricates a jump across the gap. There is no smoothing
and no octave-jump rejection: a single tracker octave error contributes a full
twelve semitones, which makes this partly a measure of tracker noise.

### Pitch search range

The floor and ceiling bound the tracker's search. They are not cosmetic: they
are the dominant control on octave errors in this material.

- A floor far below the population's real range invites **octave halving** —
  the tracker locks onto a subharmonic and reports half the true pitch. This is
  the mechanism behind essentially every implausibly low reading in the corpus.
- A floor set too high clips the genuine bottom of the range and produces
  **upward** errors instead, which are harder to notice because they look like
  expressive speech.
- Raising the ceiling has the same failure mode in the other direction. A
  textbook two-pass adaptive range (wide first pass, then bounds derived from
  that pass's quantiles) is **not** appropriate here: the first pass is already
  contaminated on exactly the clips that need fixing, so the floor never rises,
  while the widened ceiling introduces fresh upward jumps on clips that were
  fine.

The rule: a **fixed** range, chosen for this population, validated by confirming
it moves healthy clips by less than the measurement noise floor while correcting
implausible ones.

### Choice of tracker

The tracker is selected by evidence, against four criteria in order: absolute
accuracy on synthetic signals with known F0, including signals with a competing
periodic sound mixed under the target voice; deviation from cross-tracker
consensus on real audio; invariance to encode quality; and cost.

**The production tracker is CREPE**, a neural tracker. The decisive criterion
is the second half of the first one. On a clean or merely noisy tone every
Praat variant is exact and CREPE carries a bias of roughly a tenth of a
semitone. Add a competing periodic sound under the voice, and every Praat
variant — autocorrelation, cross-correlation, and the newer filtered
autocorrelation alike — locks onto the competitor and lands about nineteen
semitones out, while CREPE is unmoved. Competing sound is precisely what
corrupts this material, so that one column decides it. CREPE's clean-signal
bias is far below the corpus's own measurement noise floor, and its cost is
roughly twenty times Praat's per clip, which is affordable.

Two findings worth keeping, because both contradict a plausible expectation:

- **Filtered autocorrelation is not the fix for octave errors here.** It is
  Praat's current recommendation for F0 and it fails this material exactly as
  the older methods do, because the problem is a competing source rather than
  the correlation method. It also sits marginally further from consensus on
  real audio than plain autocorrelation.
- **Not every Praat method is reachable from every binding.** The Python
  binding and the standalone Praat program ship different Praat versions, and
  the newer methods exist only in the latter, so evaluating them means driving
  Praat as a subprocess. Probe for a method before designing around it.

The search range is fixed rather than adaptive, and the tracker's identity and
full range are recorded on every measurement, because a quality threshold
derived from a tracker parameter has to be re-derivable.

## Voice quality

**Jitter (local)** and **shimmer (local)** — cycle-to-cycle variation in period
timing and in amplitude respectively, as fractions. **HNR (dB)** —
harmonics-to-noise ratio; high is clear and tonal, low is breathy or noisy.

All three carry the same two caveats, and both belong in any caption that
displays them:

1. Praat's algorithms for these are calibrated for a **sustained vowel**, not
   for conversational speech. The absolute values are not comparable to clinical
   reference ranges.
2. They are strongly affected by the separator. HNR in particular is inflated by
   it in a systematic direction — a denoising transform raises harmonicity
   whether or not the voice changed — so a trend in HNR across the corpus is a
   candidate artifact before it is a finding.

## Spectrum

**Brightness (spectral centroid, Hz)** — the magnitude-weighted mean frequency
of each frame, summarised across the clip by the **median** of voiced frames.
Interpreted as perceived brightness. The median rather than the mean because a
handful of frames can produce wild centroids, and one of them should not move
the clip's value.

Three things constrain it:

- It is computed after a **downsample**, which must be anti-aliased. Resampling
  by plain interpolation folds energy above the new Nyquist frequency back into
  the band, and for a statistic that *is* a weighted mean frequency that
  contaminant lands directly in the published number.
- It must be masked to frames that actually contain voice. Averaging silence and
  separator artifact into a spectral centroid makes it partly a measure of how
  much silence the window contained.
- Microphone and EQ differences between talents, and between eras for the same
  talent, are large relative to the between-talent spread. Relative shape within
  one talent is more trustworthy than a cross-talent ranking.

**Loudness dynamics (dB)** — the standard deviation of per-frame RMS in dB, over
frames above an energy floor. Being a standard deviation it is gain-invariant,
so it survives differences in upload level, but not differences in compression
or limiting.

## Formants

**F1–F4 (Hz)** — vocal-tract resonance frequencies from Praat's Burg method.
Formant spacing is the most literature-grounded acoustic correlate of perceived
voice maturity, because it tracks vocal-tract length.

Four constraints, all load-bearing:

1. **Voiced-frame gating is mandatory.** Burg returns finite values for
   unvoiced and noisy frames too. Averaging over every frame with a finite value
   mixes consonants, breath, silence, and separator artifact into the estimate.
   The unvoiced share of contributing frames is substantial — it can exceed half
   — and gating shifts the result by tens to hundreds of Hz. v1 did not gate;
   this is its most serious measurement defect. Formants are sampled at the
   pitch track's own frame times, so the gate is exact by construction rather
   than correct to within a frame of alignment, and the kept and total frame
   counts are recorded so the gating can be audited. The per-clip summary is
   the mean of the kept frames.
2. **The formant ceiling must suit the speaker.** A single fixed ceiling applied
   to every voice is a compromise, not a neutral default. Until a per-speaker
   ceiling is validated, the fixed value is recorded on every record as a
   parameter so the compromise is visible.
3. **Averaging across arbitrary vowels does not isolate vocal-tract length.**
   Vowel composition varies by what someone happened to say. Two clips differing
   in F1 may differ in vowel inventory, not anatomy.
4. **Recording quality materially alters LPC formant estimates**, so the encode
   era and the separator both act on these numbers directly.

F1–F4 are correlated with each other strongly enough that four separate
published axes overstate the dimensionality. They are presented as one resonance
summary, labelled experimental.

## Background-energy index

The vocal separator emits both a vocals stem and a residual stem. The energy
ratio between them is a direct, per-clip measure of **how much non-target sound
the source contained** — music, game audio, video being watched, crowd.

This is the check that window selection cannot perform, and it costs nothing
beyond what separation already produced. It serves three purposes:

- deciding whether a clip needs separation at all (see below)
- detecting the contaminated clips that produce implausible pitch readings
- testing claims about stream types empirically rather than by assumption

## Separation as a conditional step

Separation is a lossy transform. On audio that contains no competing sound it
can only add error, and the table at the top of this file shows that the error
it adds is large for most features. Applying it unconditionally to every clip —
v1's rule, adopted so that the transform would at least be *uniform* — trades a
known distortion for a uniform one.

The better rule is conditional: separate when the background-energy index says
there is something to remove, and measure directly otherwise, with the decision
recorded per clip so it can be controlled for. Uniformity is then achieved by
recording the treatment rather than by applying the same damage everywhere.

Which separator, and the threshold for applying it, are settled by comparison
against an *identity-preservation* criterion: on clips that need no separation,
the transform that perturbs F0, formants, and HNR least is the better one.
