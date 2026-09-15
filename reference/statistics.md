# Statistics

How clip measurements become series, summaries, and claims.

## The sampling unit

A **clip** is one measurement. A **month** is the reporting unit. Months are
aggregated into quarters and years.

Aggregation is always **month-first**: a month's value is computed from its
clips, and a quarter or year is computed from its months' values. Pooling
clip-level values directly into a year lets a densely sampled month outweigh a
sparse one, turning uneven sampling effort into apparent signal.

One estimator is used at every level. Using a mean at one granularity and a
median at another makes two views of the same data disagree for reasons that
have nothing to do with the data. Where a median is used on an even count, it is
the ordinary median — the lower median introduces a bias whose sign depends on
the parity of the sample count, which is an artifact with no defensible reading.

Any operation that caps or subsets clips must have a **deterministic tie-break**,
so the same inputs always produce the same outputs regardless of file order.

## The noise floor

Most months carry more than one clip. **The difference between two clips of the
same talent in the same month is the instrument's noise floor** — same person,
same period, same pipeline, so what remains is measurement error plus
day-to-day variation.

This is computed per metric, per talent, and corpus-wide, and it is the number
that makes every other number interpretable. A trend is only a finding if it is
large relative to the noise floor over the span measured; a difference between
two talents is only a difference if it exceeds it.

It decomposes further when several windows are cut from one stream: variation
*within* one stream is pure instrument noise, while variation *between* streams
in a month adds genuine day-to-day variation. Separating the two distinguishes
"she sounded different that day" from "our measurement is noisy".

Publishing a metric without its noise floor is publishing a number without its
units.

## Spread and uncertainty

**Min–max is not uncertainty.** It expands with sample size, so a
well-sampled period looks less certain than a poorly sampled one. It is a range,
it is labelled as a range, and it is never presented as a confidence interval.

Uncertainty on an aggregate is a bootstrap over the month values that produced
it. Robust spread and uncertainty are shown as separate things because they
answer different questions: how variable is this person, versus how well do we
know their typical value.

## Trends

Trends are reported with a robust slope estimator alongside the ordinary
least-squares fit, and with the first-to-last change over comparable spans. A
slope without a goodness-of-fit figure and without the noise floor beside it
invites a reader to see a trend that the scatter does not support.

Two independent clips per month also give a **replication check**: fit the trend
separately on each month's first and second clip and compare. Agreement in sign
and magnitude across independent samples is strong evidence the trend is not a
sampling accident. It is *not* evidence about cause.

## Correlation

Metrics in this corpus fall into a few correlated families — the formants move
together, the voice-quality measures move together, and pitch spread is
partly determined by pitch level. A display with many axes therefore shows fewer
independent things than it appears to, and double-counts whatever the correlated
family measures.

Correlations are reported **twice and never pooled**: between talents (do people
with higher pitch also have higher brightness?) and within a talent over time
(when this person's pitch rises, does their brightness rise?). These are
different questions with different answers, and a single pooled correlation is a
mixture of the two that answers neither.

Composite indices built from correlated axes are exploratory by construction.
Where one is published, it is titled and captioned as an index, not as a
measurement of the thing its name evokes.

### Presenting fewer things than there are columns

The measured correlation structure — not an assumed one — groups the metrics
into five families. The formants move together strongly enough that four axes
are close to one. The voice-quality measures move together too, with
harmonicity opposing jitter and shimmer, because all three are reading the same
periodic-versus-noisy quality. Pitch spread is partly determined by pitch level
whenever spread is expressed in Hz.

So a display is organised as:

1. **Pitch level** — typical F0, its corpus percentile, its trend.
2. **Pitch movement** — scale-free spread and frame-to-frame movement.
3. **Periodicity and noise** — jitter, shimmer, harmonicity, together and
   labelled experimental.
4. **Spectrum and resonance** — brightness and a single formant summary.
5. **Context and measurement quality** — voiced fraction, loudness dynamics,
   coverage, background level, and same-month agreement.

Showing every column as an independent axis implies more dimensions than the
data has, and lets one underlying phenomenon vote several times in anything
built on top.

## Confounding

### Career time versus calendar time

The corpus's most striking pattern is a change in pitch over career time. A
corpus-wide change in *recording conditions* — codec, microphone, streaming
software, or the separator's behaviour on older encodes — produces the same
pattern as a corpus-wide change in *voices*.

**The obvious test is impossible, and this is a fact about the data rather than
a missing feature.** Career months = calendar time − debut date. Once a
talent-specific intercept is in the model, the career and calendar slopes are
exactly collinear and the design matrix is rank-deficient. This is the
age-period-cohort identification problem, and staggered debut years do **not**
rescue it: debut is the cohort term, and it is the third leg of the same
identity. A two-term fit returns coefficients chosen by the solver's
pseudo-inverse, not by the data. Reporting them would be worse than reporting
nothing, because they look like an answer.

Three things can be done instead, and together they bound the problem:

1. **Non-linear period effects are identifiable.** Remove each talent's own
   linear career trend, then look for calendar-time structure shared across
   talents who are at different career stages. A shock — everyone moving at the
   same calendar date — survives this and is strong evidence of an era effect.
   Talents are weighted equally, not pooled by clip, so a densely sampled
   talent cannot *become* the common effect.
2. **A smooth era drift is absorbed by step 1 and stays invisible.** That limit
   is real and must be stated wherever a trend is published. It is the reason
   the corpus cannot, by itself, promote "the measured series fell" to "voices
   changed".
3. **External measurement breaks the deadlock for one mechanism.** Re-encoding
   the same audio at different qualities measures the codec component directly,
   from outside the panel. It is the only way to put a number on part of what
   step 1 cannot see, which is why the encode-sensitivity experiment is not
   optional.

A corpus-wide *synchronised* move in any metric is treated as a suspected era
effect until step 1 says otherwise. Dozens of independent people do not change
the same way at the same time for biological reasons.

### Encode era

The era confound is also measurable directly: re-encode the same audio through
the bitrate and loudness profiles characteristic of different eras, run the
whole pipeline, and report how far each feature moves. This gives a per-feature
sensitivity in the feature's own units, and it simultaneously tests whether the
separator behaves differently on degraded input.

A feature whose era sensitivity is comparable to the trend it shows cannot
support a claim about voices.

### Selection

Window selection maximises voiced fraction, which biases toward continuous,
confidently-tracked speech (see [`sampling.md`](sampling.md)). Stream type
varies across a career as a talent's content changes. Both are recorded per clip
so they can enter an analysis as covariates.

## What gets published

A talent summary answers, in order: the typical value and where it sits in the
corpus; the trend with its fit quality; coverage, as passing clips over eligible
months; the same-month disagreement, which is that talent's noise floor; and a
plain statement of what the series does and does not support.

The last line is not decoration. A reader who sees a number without it will
assume more than the measurement establishes.
