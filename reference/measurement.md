# Measurement

Every acoustic feature: what it is, how it is computed, and what it is valid
for. Implementation: `acoustics.py` (the v2 extractor), `voice_source.py`
(source, timbre and tempo), and `praat_features.py` / `features.py` (v1's
extractor, frozen).

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
| H1*–H2*, harmonic tilt, alpha ratio, Hammarberg index | Largely — the shift is a small fraction of the noise floor | Measured by re-running directly-measured clips on their separated stem (`vvc voice-validate`) |
| CPP | **Partly** — the shift approaches the noise floor, and is one-directional | Separation raises it, as it raises HNR: denoising makes any voice measure clearer. Compare like with like |
| Formant dispersion | Largely, unlike the individual formants | The spacing survives even where each formant's absolute value moves |
| Speaking rate | Partly — a sizeable fraction of the noise floor | Depends on the voiced mask, which separation moves |

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

## Voice source and timbre

Pitch and formants place a voice — how high, roughly how long the vocal tract —
but cannot tell a breathy, soft voice from a pressed, full one at the same
pitch. That quality lives in the voice *source*: how completely the vocal folds
close, which sets how fast energy falls away above the fundamental and how much
aspiration noise rides on the harmonics. Implementation: `voice_source.py`.
Every measure is taken per voiced frame on the pitch track's own grid and
summarised by the median.

**H1*–H2* (dB)** — the first harmonic's level over the second's. A breathy
voice has a dominant fundamental and weak overtones; a pressed voice the
reverse. Harmonic levels are read from a zero-padded spectrum within a tenth of
F0 of each harmonic. The raw difference is also recorded, but the published
value is **formant-corrected** (Iseli and Alwan): the level each harmonic gains
from F1 and F2, given their measured frequencies and bandwidths, is removed.
The correction is not optional in this corpus. At the pitches these voices use,
the second harmonic very often sits on F1, which boosts it by many decibels, so
the raw value largely measures the vowel. `h2_near_f1_fraction` records how
often that happened (the second harmonic closer to F1 than half an F0 step),
because the correction is only as good as the formant estimate beneath it.

**Harmonic tilt (dB per octave)** — the slope of harmonic levels against
log-frequency, up to 5 kHz. Steeply negative is a soft voice with little
energy up high; shallow is a bright, full one.

**Alpha ratio and Hammarberg index (dB)** — the same question asked of bands:
energy in 1–5 kHz over 50 Hz–1 kHz, and the strongest peak below 2 kHz over the
strongest in 2–5 kHz. Cheaper and less dependent on harmonic resolution than
tilt; both are standard in the eGeMAPS feature set.

**CPP (dB)** — cepstral peak prominence: how far the voice's periodicity peak
in the cepstrum stands above a regression line through the cepstral baseline.
Aspiration noise fills in the spectrum between harmonics and lowers it. It is
the best-validated breathiness measure for *connected* speech, which is why it
is preferred here over harmonicity, jitter and shimmer. The peak is searched
over the tracker's own F0 range.

All of these are spectral, so everything said below about microphones and EQ
applies to them in full. Separation, by contrast, moves the spectral-shape
measures little; CPP is the exception, raised by it in a consistent direction
(see the table at the top). They describe a voice as recorded and
processed; the invariance column at the top of this file says how much of that
processing they survive.

## Tempo

**Speaking rate (syllables per voiced second)** — syllable nuclei counted as
intensity peaks that stand at least a couple of decibels above the dips either
side, lie within a fixed range of the clip's loudest frame, and fall inside
voiced speech (after de Jong and Wempe). Dividing by voiced time rather than
the whole window makes it an articulation rate, not a measure of how much of
the window was talk — which the window selection rule controls, not the
speaker.

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
summary, labelled experimental: **formant dispersion**, the mean spacing
(F4 − F1) / 3 over voiced frames, the usual single-number correlate of
vocal-tract length.

## Speaker embeddings

A speaker-recognition network maps a clip to a vector — an embedding — arranged
so that clips of the same voice land close together. Its geometry is learned
from thousands of speakers, so the distance between two talents' embeddings is
the strongest available measure of how alike their voices sound: much stronger
than a handful of acoustic features, and blind to *why*. The measured features
answer why; the embedding answers whether. Implementation: `embed.py`.

- The encoder is a **general** speaker model (ECAPA-TDNN trained on VoxCeleb),
  deliberately not one trained on this corpus. A model trained to tell these
  talents apart learns to push similar voices apart, which is the opposite of
  what a similarity ranking needs.
- Each clip is embedded from the same audio its features were measured on. A
  talent's voice is the re-normalised mean of their usable clips' embeddings.
- **Embeddings are never published or committed.** They are exactly the input a
  zero-shot voice-cloning system conditions on. Only similarity scores derived
  from them leave the machine.
- They inherit every recording confound the spectral features have —
  microphone, EQ, separation — and add one: the network was trained on
  interview speech in many languages, and a language switch moves an embedding
  even for the same voice. Early-versus-late splits in the validation report
  are how the recording-era part is measured.

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
