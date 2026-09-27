# Limitations

Every number describes a short window of a public stream after encoding and
processing. It is a measurement of the stream voice, which is not the same
thing as a person's voice.

## Character, context, and recording

- **Character performance.** Talents perform characters. A change in the data
  can reflect the character, the performer's choices, or the voice itself.
- **Context.** People speak differently when reading chat, playing a game, or
  watching a video. Stream type is recorded but is only a rough proxy.
- **Recording era.** Microphones, codecs, and streaming software changed over
  time, often for everyone at once. A slow change in recording conditions
  cannot be told apart from a slow change in voices, so a trend in the data is a
  trend in the measurements, not proof that a voice changed.
- **Window selection.** Windows are chosen for continuous speech, so they are
  not a random sample of a stream.

## Speaker identity

Nothing confirms that every clip contains only the intended talent. An
unmarked guest, game dialogue, or a video being watched can end up in a clip.
Quality checks and review flags catch the obvious cases, not all of them.

## Metrics

| Metric | Suitable for |
|---|---|
| Median pitch | Comparing talents and following long-term trends |
| Pitch spread, dynamism | Descriptive comparison |
| H1*–H2*, tilt, alpha ratio, Hammarberg, formant dispersion | Exploratory comparison; affected by microphones and EQ |
| CPP | Exploratory; higher on separated clips |
| Speaking rate | Exploratory |
| Brightness, F1–F4 | Exploratory; strongly affected by recording setup |
| Jitter, shimmer, HNR | Exploratory at best |

## Voice similarity

Similarity rankings reflect recordings as well as voices. Talents with similar
microphones or recording eras can appear closer than they sound. The embedding
ranking removes the average effect of each language group, but not every
language effect. One-off profiles rest on a single stream.

## Coverage

Coverage differs between talents. Early streams are often private or
unavailable, and some months have no eligible stream at all. A sparsely covered
talent's summary is less certain than a densely covered one.
