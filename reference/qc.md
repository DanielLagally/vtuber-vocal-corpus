# Quality control

## The clip gate

Every clip is checked in this order; the first failure is recorded as the
reason (`quality.py`).

| Check | Fails when | Reason |
|---|---|---|
| Pitch measured | No voiced frames | `f0_missing` |
| Enough speech | Voiced fraction below 0.15 | `voiced_fraction` |
| Plausible spread | Pitch interquartile range of 14 semitones or more | `f0_spread` |
| No lower voice | A quarter or more of voiced frames below 160 Hz | `contaminated` |
| Plausible pitch | Median pitch outside 100–700 Hz | `f0_out_of_range` |

Spread is judged in semitones because a fixed Hz threshold is much looser for
high voices. The 160 Hz check catches the most common contamination: a lower
voice from a guest, a game, or a video. The pitch range is a wide backstop; the
contamination check does most of the work.

A failed clip is a gap. It is never replaced with a default or a neighbouring
value. The verdict is recomputed from the stored features wherever it is used.

Passing the gate means the pitch track is sound. It says nothing about formants,
brightness, or voice quality.

## Plausibility against the talent's own baseline

A clip can pass the gate and still not be the talent, for example a guest with
a normal-sounding voice. Once a talent has enough months, each clip is compared
with that talent's own pitch trend:

- The threshold is in semitones, not in units of the talent's spread.
  Contaminating voices sit a similar absolute distance below every talent.
- The baseline follows the talent's trend, so genuine long-term drift is not
  mistaken for contamination.
- The band is tight below and loose above. Contamination is almost always a
  lower voice, while character voices and wide natural ranges extend upward.

Talents with too few months get no band. The thresholds were set against human
listening labels and are deliberately coarse.

Three cases are known to pass: character performances within a talent's range,
instruments tracked as a voice, and pitch-shifted game voices. Sampling several
clips per month and taking the median limits the damage from all three.

## Review flags

A clip is flagged for review when it differs from a same-month clip of the same
talent by more than a musically implausible interval. Flags mark clips for a
person to check; they do not remove data.
