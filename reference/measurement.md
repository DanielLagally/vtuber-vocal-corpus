# Measurement

Features are measured on 90-second clips (`acoustics.py`, `voice_source.py`).
One pitch track is computed per clip, and every frame-based feature uses that
track's frames, so all features share the same definition of voiced speech. Each
clip value is the median over voiced frames unless stated otherwise.

## Robustness

Clips pass through YouTube's encoder and, when they contain competing sound, a
vocal separator. Features differ in how much this changes them:

| Feature | Effect of separation |
|---|---|
| Median pitch | Negligible |
| Pitch spread, pitch movement | Small; affected by occasional octave errors |
| H1*–H2*, harmonic tilt, alpha ratio, Hammarberg index | Small relative to the noise floor |
| Formant dispersion | Small |
| Speaking rate | Moderate |
| CPP, HNR | Raised consistently |
| Jitter, shimmer | Large |
| Brightness, individual formants | Large |
| Voiced fraction | Large |

Separation is applied only when a clip contains enough competing sound, and each
record notes whether it was separated.

## Pitch

**Median F0 (Hz)** — typical speaking pitch.

**F0 spread (semitones)** — interquartile range of pitch within the clip.

**Pitch dynamism (semitones)** — mean absolute change in pitch between
consecutive voiced frames. Pauses are skipped.

Pitch is tracked with CREPE, a neural pitch tracker, over a fixed 60–800 Hz
range. It was chosen over Praat's methods because it stays accurate when
another periodic sound, such as music or a second voice, is mixed under the
speaker, which is the main source of error in stream audio.

## Voice source and timbre

These describe how a voice sounds at a given pitch: airy or pressed, soft or
bright.

**H1*–H2* (dB)** — level of the first harmonic relative to the second. Higher
values mean an airier, breathier voice. Harmonic levels are corrected for the
boost they receive from the first two formants (Iseli and Alwan). At the pitches
in this corpus, the second harmonic often falls near the first formant, and
without the correction the value mostly reflects the vowel.

**CPP (dB)** — cepstral peak prominence: how clearly the voice's periodicity
stands out from noise. Lower values mean a breathier voice. Separation raises
it, so separated and unseparated clips are not directly comparable.

**Harmonic tilt (dB per octave)** — slope of harmonic levels up to 5 kHz. Steeper
(more negative) means a softer voice with less high-frequency energy.

**Alpha ratio (dB)** — energy from 1–5 kHz relative to 50 Hz–1 kHz.

**Hammarberg index (dB)** — strongest peak below 2 kHz relative to the strongest
from 2–5 kHz.

**Brightness (Hz)** — spectral centroid of voiced frames. Microphone and EQ
differences are large relative to differences between talents.

## Formants

**F1–F4 (Hz)** — vocal-tract resonances, measured with Praat's Burg method on
voiced frames only.

**Formant dispersion (Hz)** — mean spacing, (F4 − F1) / 3. A standard correlate
of vocal-tract length, and more stable than the individual formants, which vary
with the vowels spoken and use a fixed analysis ceiling for every speaker.

## Tempo

**Speaking rate (syllables per second)** — syllable nuclei per second of voiced
speech, counted as intensity peaks within voiced regions (after de Jong and
Wempe).

## Periodicity

**Jitter, shimmer** — cycle-to-cycle variation in period and amplitude.

**HNR (dB)** — harmonics-to-noise ratio.

These are designed for sustained vowels rather than conversation, and are
strongly affected by separation.

## Recording context

**Voiced fraction** — share of frames with detected pitch. Largely determined by
how the window is chosen.

**Loudness dynamics (dB)** — standard deviation of frame loudness.

**Background ratio (dB)** — energy of the separator's residual relative to the
voice: how much music, game audio, or other sound the clip contained. It decides
whether a clip is separated.

## Speaker embeddings

A speaker embedding is a vector produced by a speaker-recognition network
(ECAPA-TDNN trained on VoxCeleb) such that recordings of the same voice land
close together. Each clip is embedded, and a talent's voice is the average of
their clips' embeddings. Embeddings capture much more about identity than the
measured features, but they do not explain what makes two voices similar. They
are also affected by the language spoken; see
[`statistics.md`](statistics.md).
