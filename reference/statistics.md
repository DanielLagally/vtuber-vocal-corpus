# Statistics

## Aggregation

A clip is one measurement; a month is the reporting unit. A month's value is the
median of its clips. Quarters and years are the median of their months' values,
so a heavily sampled month does not outweigh a sparse one. The same estimator is
used at every level.

A talent's **typical value** is the median of their monthly values. Its 95%
confidence interval is a bootstrap over those months. Min–max ranges are shown
as ranges, not as uncertainty.

## Noise floor

Two clips of the same talent in the same month should measure alike; the
difference between them is the measurement's noise floor. It is reported per
metric, per talent, and across the corpus. A trend or a difference between
talents means little unless it is larger than the noise floor.

## Trends

Trends are fitted over each talent's career time, measured from their first
month in the corpus. They are reported with their fit quality (r²) next to the
noise floor.

## Metric families

Related metrics move together, so the site groups them:

1. **Pitch level** — median pitch
2. **Pitch movement** — pitch spread and dynamism
3. **Voice source and timbre** — H1*–H2*, CPP, harmonic tilt, alpha ratio,
   Hammarberg index
4. **Periodicity and noise** — jitter, shimmer, HNR
5. **Spectrum and resonance** — brightness, formant dispersion, F1–F4
6. **Tempo** — speaking rate
7. **Recording context** — voiced fraction, loudness dynamics, background ratio

Pitch level and movement are marked robust; the others are experimental, and
recording context describes the recordings rather than the voice.

Percentiles rank each talent's typical value against all other talents. One-off
profiles are placed within that ranking without changing anyone else's
percentile. They appear in closest-voice lists, tagged, but do not affect other
talents' scores.

## Voice similarity

"Closest voices" is computed two ways (`similarity.py`).

**Sounds like** uses speaker embeddings. Each talent's average embedding is
compared by cosine similarity. Embeddings respond to the language spoken as well
as to the voice, so the average embedding of each language group (JP, EN, ID) is
subtracted first. A one-off in another language can declare its language.

**Measured** compares talents' typical values on the voice metrics. Each
difference is scaled by how much that metric varies between one talent's own
clips, and correlated metrics are accounted for together, so noisy or redundant
metrics count for less (a Mahalanobis distance under the pooled within-talent
covariance). Frequencies are compared on a log scale.

Both are reported as a **voice match** from 0% to 100%. 0% is the median
similarity between two unrelated talents; 100% is the median similarity between
two halves of one talent's own clips, i.e. as alike as a talent is to
themselves across streams. The rank among all talent pairs is shown on hover.
Each match lists the metrics on which the pair is unusually close compared with
how much talents typically differ.

The radar chart shows each talent's distance from the median talent on each
voice metric, in units of within-talent variation.

## Validation

`vvc voice-validate` checks the metrics and embeddings against the data itself:

- **Speaker identification** — each clip is assigned to the talent whose average
  it is nearest, with that clip left out. Higher accuracy means the metrics
  capture more of who is speaking.
- **Between-talent share** — the fraction of each metric's variation explained by
  which talent is speaking.
- **Agreement** — rank correlation between the two similarity methods.
- **Stability** — whether a talent's closest voices stay the same when computed
  from two random halves of their clips, and from earlier versus later clips.
- **Separation effect** — how much each metric changes when the same clip is
  measured with and without separation.

## Career and recording era

Career time is calendar time minus debut, so a slow change in recording
conditions across the whole corpus cannot be separated from a slow change in
voices. `vvc analyse` removes each talent's own trend and looks for changes
shared across talents at different career stages, which points to recording
effects. `vvc bakeoff` re-encodes the same audio at different qualities to
measure how much encoding alone moves each feature.
