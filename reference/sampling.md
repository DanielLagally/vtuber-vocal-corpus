# Sampling

## Talents

Active hololive talents in JP, EN, ID, and DEV_IS, plus graduated members whose
archives are still available. Only each talent's own channel is used.

## Eligible streams

A video is excluded if any of these apply:

- it is not a live stream (uploads, premieres, clips)
- its topic is singing, a Short, an original song, a cover, a teaser, or
  members-only
- it is a collab (Holodex `mentions`)
- it is too short to contain a speech window after the intro
- its title mentions karaoke (`歌枠`) or a guest (`ゲスト`)

Collab detection depends on requesting `include=mentions` from Holodex; without
it, collabs look like solo streams.

## Choosing streams

Each eligible stream gets a score from its topic and length. Talk and chatting
streams score highest, then unlabelled streams, then everything else. Streams
long enough to contain representative speech score higher than very short or
very long ones. The highest-scoring streams in each month are measured, with
ties broken by the most recent.

Topic is a preference, not a requirement. A month with no talk stream is
represented by its best alternative, so game streams and watchalongs are part of
the corpus. The topic label is also often missing. Each record stores its topic
and a reliability band so stream type can be accounted for in analysis.

Stream type is a poor guide to audio quality. Watchalongs measure slightly
cleaner than talk streams, ASMR is by far the noisiest category, and game
streams vary by game. Clips are judged by their measured background level, not
their label.

## Clips per month

The target is three clips per month. With a median of three, one bad clip cannot
move the month's value; with two, it moves it by half its error. Extra clips per
month protect against contamination more effectively than any filter, and they
provide the within-month comparisons the noise floor is computed from.

Months with no eligible stream are gaps. Nothing fills them in.

When a month has more candidates than needed, the cleanest are kept, judged only
by properties of the recording such as background level. Selecting on anything
derived from the voice itself would bias the results toward stability.

## Window within a stream

The opening of each stream is skipped. Within the remaining audio, a 90-second
window is chosen to maximise continuously voiced speech. This favours clear,
continuous talking, so voiced fraction describes the selection as much as the
speaker.

## One-off clips

A one-off covers a chosen stretch of a single stream instead of a monthly
sample, for example a talent speaking another language. Every consecutive
90-second window of the stretch is measured. One-offs appear as their own
profile but are not ranked against other talents.
