# Sampling

What enters the corpus, and how a single clip is chosen to represent a month.

## Roster

Active hololive **talent** channels: JP, EN, ID, DEV_IS, plus graduated members
whose archives remain reachable. Holodex `Official` and `Misc` groups are
excluded, as is anything with HOLOSTARS in the group name — the corpus is about
individual talents' own channels, not org-produced content.

Implementation: `roster.py`.

## Eligibility

A video is rejected outright (`catalog.reject_reason`) when any of these holds:

- it is not a `stream` (clips, premieres, uploads)
- its `topic_id` is one of: `singing`, `shorts`, `Original_Song`, `Music_Cover`,
  `Teaser`, `membersonly`
- it has `mentions` — Holodex's collab marker
- it is shorter than the minimum duration (a stream too short to contain a
  usable speech window after skipping the intro)
- its title contains `karaoke`, `歌枠`, `ゲスト`, or `guest`

Two filters exist and both are in use: `filter_videos` is the looser pre-pass
applied when caching a channel listing; `reject_reason` is the strict gate
applied at selection.

**`mentions` is only populated when the Holodex request passes
`include=mentions`.** Without it, collabs are indistinguishable from solo
streams. This is the single most consequential API detail in the sampling path.

## Selection

`catalog.score_video` assigns a score, and `pick_monthly_n` takes the
highest-scored videos per month, breaking ties by newest `available_at`.

Score is the sum of a topic term and a duration term. The topic term prefers
talk/chatting/morning streams, then unlabelled streams, then everything else,
with watchalongs lowest. The duration term prefers streams long enough to
contain a representative window without being so long that the sampled window is
unrepresentative of the stream.

**Topic is a preference, not a gate.** A month with no chatting stream will be
represented by whatever scored highest among what exists — a game stream, or a
watchalong. This is deliberate: a gap in the series is worse than a clip whose
stream type is recorded and can be controlled for.

Two consequences follow, and both are larger than they sound:

- **A talent's early era may contain no talk streams at all.** Some months
  offer twenty eligible candidates and every one of them is a game stream.
  There is no selection rule that produces a chatting clip from a month
  containing none, so "chatting-stream corpus" describes an intent rather than
  the contents.
- **The topic field is frequently absent**, and absent does not mean talk. An
  unlabelled stream scores as unlabelled whether it is a chat or a game, so the
  score cannot distinguish them. Any analysis that treats the stored topic as a
  reliable content label will be wrong in the same direction repeatedly.

## Stream type is a covariate, not a quality tier

The corpus does not assume a ranking of stream types by audio quality. Two
separate questions are often conflated:

1. **Does the audio contain non-target sound?** Music, game audio, a second
   speaker, video being watched. This is measurable directly — see the
   background-energy index in [`measurement.md`](measurement.md) — and measuring
   it shows that it **does not follow stream type the way the v1 scoring rule
   assumed**.

   Watchalongs, which v1 ranked lowest, measure slightly *cleaner* than talk
   streams: the talent's commentary is the audio, and the thing being watched
   frequently is not in the mix at all. Meanwhile ASMR — which v1 treated as an
   ordinary stream and selected freely — is by far the worst category, often
   carrying more non-voice energy than voice, because deliberate non-vocal
   sound is the entire point of the format. Game streams sit between the two
   and vary by game rather than by being games.

   The lesson generalises: rank clips by what their audio measures, never by
   what their label suggests.

2. **Does the talent speak differently in this context?** Reading chat, reacting
   to a game, and narrating a film plausibly differ in pitch, dynamism, and
   pause structure. This is a genuine open problem and it is **not** specific to
   watchalongs — it applies across every stream type the corpus samples.

Consequently: `topic_id` and `catalog.reliability_band` are **stored on every
record** so stream type can be entered into an analysis as a covariate and its
effect estimated, rather than being silently assumed. Nothing in the measurement
or QC path treats a stream type as disqualifying.

## Density

Each month a talent was active is a sampling unit. Months with no eligible
stream are gaps, and a gap is never filled with an interpolated or zero value
anywhere in the pipeline.

**Three clips per month is the threshold that matters**, and the reason is
worth spelling out because two looks almost as good and is not. With two clips
the monthly median *is* the mean, so a single bad clip moves the month by half
its error — and contamination errors here are around an octave, so a month
moves by half an octave. With three clips and a median, a single bad clip
cannot move the month at all: more than half the clips have to be bad. At the
observed contamination rate that is a roughly seventy-fold reduction in
corrupted months, which is far more than any detection gate delivers, and it
requires detecting nothing.

Clips per month therefore buys robustness more cheaply than better filtering
does. It also buys a better noise-floor estimate, since within-month
disagreement is the only direct measure of it (see
[`statistics.md`](statistics.md)).

## Choosing among candidates

A month often offers far more eligible streams than the target number of clips.
Where it does, candidates may be measured and the cleanest kept — but only on
signals that are **properties of the recording, not of the voice**.

The background-energy ratio is such a signal: it describes how much competing
sound the source contained, independently of what the talent's pitch did.

Selecting on anything derived from the measured quantity is forbidden, and the
failure mode is severe rather than subtle. Preferring clips whose pitch sits
close to the talent's baseline would suppress within-talent variance and flatten
long-run trends — manufacturing the stability the corpus exists to test for.
The same objection rules out selecting on the plausibility distance, on
within-month agreement, or on anything else computed from the pitch track.

Selecting on recording quality still biases the sample, because cleaner audio
correlates with content type. That bias is acceptable only because the
selection signal is recorded per clip and can be controlled for.

## Window within a stream

The fetch window skips the opening of the stream, which is typically a waiting
room or intro rather than speech.

Within the fetched audio, a fixed-length measurement window is chosen by
scanning on a regular grid and scoring each candidate. The historical score is
lexicographic on `(voiced_fraction, -f0_iqr)`, which in practice means voiced
fraction alone decides — `voiced_fraction` is a continuous float over thousands
of frames, so the IQR term only breaks exact ties, which essentially never
occur.

This biases selection toward continuous, easily tracked speech. Two consequences
follow and both are stated wherever the affected numbers are published:

- `voiced_fraction` is a property of the *selection rule* as much as of the
  speaker, and must not be read as a natural measure of how much someone talks.
- A window dense with confidently-tracked frames can be one dense with periodic
  background music, which tracks as voiced. The background-energy index is the
  check for this, not the window score.
