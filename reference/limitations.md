# Limitations

What this corpus cannot support. Read this before quoting a number from it.

## What it measures

**Sampled stream voice**, through a specific processing chain. Not "someone's
voice". Every published number describes what this pipeline extracted from a
short window of a public stream, and the gap between that and a person's voice is
where every limitation below lives.

## The confounds that are not resolved

**Character performance versus voice.** Talents perform a character. A change in
the measured series is consistent with the performance changing, the person's
voice changing, or the performer's choices about the character changing. Nothing
here distinguishes them.

**Recording era — and this one cannot be fully solved.** Codec, microphone,
streaming software, and audio processing all changed over the span the corpus
covers, and they changed for everyone at roughly the same time.

Career months = calendar time − debut date, so an era effect and a career
effect are not merely hard to separate: they are **not separately
identifiable**. This is the age-period-cohort problem, and staggered debut
years do not rescue it, because debut is the third leg of the same identity. No
additional data collected under this design fixes it.

What can be established is narrower, and is set out in
[`statistics.md`](statistics.md):

- a *shock* — talents at different career stages all moving at the same
  calendar date — is detectable, and its absence is informative;
- the encode-quality component can be measured directly, from outside the
  panel, by re-encoding the same audio;
- a *smooth* era drift remains indistinguishable from a smooth career trend,
  permanently, within this corpus.

Therefore **a trend here is a trend in the measured series.** Promoting it to a
claim about anyone's voice needs evidence this design cannot supply on its own.
A corpus-wide synchronised move in any metric is a suspected era effect first.

**Content and context.** What someone is doing while talking — reading chat,
reacting to a game, narrating a film — plausibly changes how they speak. Stream
type is recorded so it can be controlled for, but it is a coarse proxy and the
underlying question is open.

The corpus is also more heterogeneous than "chatting streams" suggests: talk
and morning streams are under half of it, the rest being games, ASMR, superchat
readings, anniversaries, cooking, and a long tail of one-off formats. ASMR is
the worst case for measurement by a wide margin — deliberate non-vocal sound is
the entire format, and often carries more energy than the voice — and it was
selected freely by v1's scoring rule. Watchalongs, which that rule ranked
lowest, measure cleaner than talk streams.

**Selection.** Windows are chosen to maximise continuously voiced speech, so the
sample is not a random sample of a stream, and `voiced_fraction` in particular
describes the selection rule as much as the speaker.

## Speaker identity is not verified

Nothing in the pipeline confirms that the measured voice is the intended
talent's. The consequences:

- An unannounced guest, or a collab the metadata did not mark, contributes to a
  talent's series.
- Game dialogue, video being watched, alerts, and audience audio can survive
  into the measured signal.
- A channel operated by more than one person cannot be treated as one speaker at
  all, and any trend on such a channel may reflect which person spoke more.

Cross-clip review flags catch the loudest cases. They are a partial proxy, not a
solution.

## Per-feature confidence

| Feature | Supports |
|---|---|
| Median F0 | Descriptive comparison and long-run trends within the sampled stream voice. Robust to the processing chain. Individual months are noisy relative to the trend |
| F0 spread, dynamism | Descriptive, in scale-free units. Inherits any tracker octave error |
| Brightness | Exploratory. Microphone and EQ differences are large relative to between-talent spread |
| Jitter, shimmer, HNR | Exploratory at best. Calibrated for sustained vowels, not conversation, and HNR is systematically inflated by separation |
| Formants F1–F4 | Experimental. Voiced gating is necessary but not sufficient — vowel composition and a fixed speaker-independent ceiling both remain uncontrolled |
| Composite indices | Exploratory by construction: built from correlated axes, so they measure fewer independent things than their axis count suggests |

## Coverage is not uniform

Coverage varies substantially between talents, for reasons that are not random:
early-era streams are frequently privated, and access gating affects some eras
and archives more than others. A talent with sparse coverage is not comparable
to a densely covered one without accounting for it, and the aggregate QC pass
rate is not a coverage figure — unfetched and ineligible months never enter its
denominator.

## Audio is retained locally

The repository and the published site contain no audio, video, or transcripts —
only derived numbers. Substantial audio is retained **locally and in a private
archive** to make re-measurement possible. "Not published or committed" is the
accurate claim; "not stored" is not.

## Not attempted

No voice synthesis, no cloneable embeddings, no speaker models, and no
identification of anyone behind a character. None of these is published, and
none is a planned capability.
