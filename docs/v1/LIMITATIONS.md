# v1 known defects and limitations

v1's measurement pipeline, as published. Read with
[`METHODOLOGY.md`](METHODOLOGY.md).

These are stated plainly because the site is live and the numbers are quotable.
Nothing here is hypothetical — each item was verified against v1's code and, where
noted, against v1's own audio.

## Defects in the measurement, by severity

### Formants F1–F4 are not what the site says they are

`formants_hz` averaged Praat's Burg output over **every frame returning a finite
value**, with no voiced-frame mask. Praat returns formants for unvoiced and
noisy frames too, so consonants, breath, silence, and separator artifact were
averaged into the estimate. Measured on real v1 stems, unvoiced frames made up
roughly a third to three-quarters of the contributing frames, and restricting to
voiced frames shifts median F1 and F2 by tens to hundreds of Hz.

The function's own docstring claims undefined frames are excluded. Only `NaN`
frames were.

Separately, the formant ceiling was a single fixed 5500 Hz for every speaker,
and averaging across whatever vowels someone happened to say does not isolate
vocal-tract length regardless of gating.

**Treat v1's F1–F4 as unusable for their advertised interpretation.**

### Some passing clips are not the talent's voice

A number of QC-passing clips sit implausibly low, several around 90–130 Hz, and
most have a same-talent, same-month counterpart two to three times higher.

The intuitive explanation is octave halving — a tracker locking onto a
subharmonic and reporting half the true pitch. **Tested on v1's own audio, that
is not what happened.** Three trackers from different algorithm families (Praat
autocorrelation, Praat filtered autocorrelation, and the neural tracker CREPE)
independently agree on these readings, and the clips carry a large voiced mass
far below the talent's range — often more than half of all voiced frames —
where their same-month counterparts have almost none. The audio genuinely
contains a lower voice: a guest, game or video dialogue, or music the separator
did not remove.

This matters for anyone tempted to "fix" v1 by raising the pitch floor. Doing so
does move these clips to plausible-looking values, but it would be forcing an
answer out of audio that is not the talent — manufacturing a result rather than
correcting an error. The clips should be treated as gaps.

v1's gate cannot detect this: it checks one clip against a plausibility band,
and a clip carrying someone else's voice is perfectly plausible on its own.

### QC has no lower bound, and its upper bound cannot fire

The gate rejected `median_f0 ≥ 600 Hz`, but the tracker's own ceiling was also
600 Hz, so it could never report a value that high: that gate is unreachable
code. There was no floor gate at all, so a 76 Hz median passed cleanly. The
failure mode most characteristic of this material was the one QC did not check.

### The IQR gate is scale-dependent

A fixed 200 Hz IQR threshold is a far weaker constraint on a high-pitched voice
than a low-pitched one — proportionally, 200 Hz is a much larger share of a
200 Hz median than of a 400 Hz one. For a corpus specifically about high-pitched
voices, the gate was most permissive exactly where it needed to be strictest. A
semitone-based threshold is the scale-free form.

### Brightness is computed on an aliased signal

`features._load_mono` downsampled to 16 kHz by linear interpolation with no
anti-aliasing filter, so energy above 8 kHz folded back into the band. For a
spectral centroid — a magnitude-weighted *mean frequency* — that contaminant
lands directly in the published number. The centroid was also averaged over all
frames with non-zero energy, including silence and separator artifact, with no
voicing mask.

### The separator is an uncontrolled transform

Every clip was separated, unconditionally. Measuring v1 clips both ways — raw
90 s window versus its own vocal stem — shows the transform is not neutral:

| Feature | Typical shift | Worst observed |
|---|---|---|
| Median F0 | ~2 Hz | ~34 Hz |
| Voiced fraction | ~0.03 | ~0.22 |
| HNR | ~1 dB, systematically upward | ~4 dB |
| F1 | ~40 Hz | several hundred Hz |

Median F0 survives well — that is why v1's pitch results remain usable. HNR does
not: a denoiser raises harmonicity by construction, so an HNR trend is a
candidate artifact before it is a finding. Voiced fraction and formants are
materially distorted.

Note also that **window selection ran on raw audio while the published
`voiced_fraction` came from the stem**, so the two never described the same
signal.

### Three different definitions of "voiced"

Window selection used a numpy autocorrelation tracker on raw 16 kHz audio;
measurement used Praat on the separated stem; and the QC thresholds were
originally derived for the numpy tracker and never re-derived for Praat.

### Aggregation is inconsistent between views

Monthly and quarterly values are means; yearly values are `statistics.median_low`
over clip-level data. That means the yearly view differs from the others both in
estimator and in pooling level — a densely sampled month outweighs a sparse one
in a yearly figure — and `median_low` carries a downward bias whose sign depends
on the parity of the clip count. The plotted yearly band is raw min–max, which
expands with sample size and is not an uncertainty interval.

### Stored QC verdicts could drift from the rule

Plots recomputed the verdict live; the site export read the stored verdict.
Changing a threshold updated one and not the other until `requalify` ran.

## Interpretive limits

**Window selection biases toward continuously voiced speech.** The score
maximises voiced fraction, so `voiced_fraction` and `dynamism` describe the
selection rule as much as the speaker and must not be read as natural
behavioural traits.

**Stream type is unrecoverable.** Game streams and watchalongs were eligible and
selected, but `topic_id` was never stored, so a v1 record cannot be stratified or
filtered by stream type after the fact.

**Speaker identity is never verified.** Unannounced guests, game and video
dialogue, and multi-person channels all pass every gate. A channel operated by
more than one person cannot be treated as a single speaker, and an apparent trend
there may reflect which person spoke more.

**Career time and recording era cannot be separated.** Career months = calendar
time − debut date, so a corpus-wide change in codec, microphone, or streaming
setup produces the same pattern as a corpus-wide change in voices — and the two
are not merely hard to tell apart, they are **not separately identifiable**.
This is the age-period-cohort problem. Staggered debut years do not rescue it,
because debut is the third leg of the same identity, so no analysis of v1's
data can settle it.

A trend in v1's data is therefore **a trend in the sampled stream voice**, not a
demonstrated change in anyone's voice, and a *synchronised* corpus-wide move in
any metric should be read as a suspected era effect first. What *could* still
be established — a shared shock at one calendar date, and a direct measurement
of the encode component — v1 never attempted.

**Composite indices are exploratory.** The cute/mature percentile combines three
correlated axes at equal weight, collapses the entire time axis so a talent whose
voice drifted is one point that never existed, weights a talent with hundreds of
clips equal to one with a handful, and assigns rank percentiles that pin the
extremes at exactly 0 and 100 regardless of how close the underlying scores are.

**QC pass rate is not coverage.** Unfetched and ineligible months never enter its
denominator. Per-talent coverage varies substantially, and early-era streams are
frequently privated — a real data-availability wall, not a pipeline failure.

**Talents with one or zero measurements are exported.** They appear in
comparison views where a single clip carries no meaningful spread.

## Snapshot identity

The generation timestamp is recorded inside `data.js` as `generated_at` but is
not rendered in the UI, so a visitor cannot tell how fresh the numbers are. There
is no release tag.
