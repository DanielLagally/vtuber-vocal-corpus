# Quality control

One gate, one implementation: `qc.qc_verdict(features) -> (pass, reason)`.

## Principles

**A failing clip is a gap, never a zero.** Nothing in the pipeline substitutes a
default, an interpolation, or a neighbouring month's value for a clip that did
not pass. A gap in a plot means missing or failed data and the site says so.

**The gate is recomputed, never trusted from storage.** A verdict is persisted
on each record for convenience, but every consumer — plots, exports, the site —
calls `qc_verdict` on the stored features rather than reading the stored
verdict. Otherwise changing a threshold silently updates some views and not
others. `qc.requalify` recomputes persisted verdicts from stored features
without re-reading audio, for when the persisted copy needs refreshing.

**The gate validates pitch, and only pitch.** A clean pitch track does not
establish clean formants, clean brightness, or clean voice quality. Passing QC
must not be read as blessing every feature on the record; the per-feature
validity constraints in [`measurement.md`](measurement.md) still apply.

## Gates

Evaluated in order; the first failure determines the reason.

1. **Median F0 is missing or non-finite** → `f0_missing`. Nothing was
   measurable.
2. **Voiced fraction below the floor** → `voiced_fraction`. A clip with almost
   no voiced content can post a deceptively *tight* F0 IQR purely from having
   too few points to spread across — low sample size masquerading as a clean
   signal. The floor sits in the natural gap in the observed distribution,
   below the first real cluster.
3. **F0 spread above the ceiling** → `f0_spread`. Background music, a melody
   the tracker followed, a second speaker, or an unstable track all widen the
   within-clip spread far beyond what one person talking produces. The
   threshold is expressed in **semitones**, not Hz: an absolute Hz threshold is
   proportionally far more permissive for a high-pitched voice, which is
   backwards for a corpus whose subject is high-pitched voices.
4. **Too much voiced energy below the talent's range** → `contaminated`. A
   per-clip signal that a lower voice is present in the audio.
5. **Median F0 outside the plausible range** → `f0_out_of_range`. A deliberately
   generous backstop, wider than any voice in the corpus, for readings no human
   speaking voice would produce. Both bounds are required and neither may
   duplicate a tracker parameter: v1's ceiling equalled the tracker's own search
   ceiling, so the tracker could never report a value that would trip it.

A threshold that duplicates a tracker parameter is not a quality gate. When the
search range changes, the range gate must be re-derived, not left pointing at
the old value.

A record that carries no contamination measure at all was produced by some
other pipeline, and is failed rather than waved through a check it was never
measured against.

## The plausibility pass

The per-clip gate above cannot answer the question that matters most here: *is
this clip even the talent?* A clip carrying a guest's voice is entirely
plausible on its own — ordinary spread, ordinary voicing, a pitch a human could
produce — and only looks wrong beside the rest of that talent's career. So a
second pass runs once a talent's series exists, comparing each clip to that
talent's own baseline.

Three properties of it were each forced by measuring against human listening
labels, and each contradicts a reasonable guess:

- **The threshold is in semitones, not standard deviations.** Normalising by a
  talent's own spread destroys the separation it is meant to sharpen: a
  wide-range talent's contamination divides into invisibility, while a
  narrow-range talent's legitimate extreme divides into looking like
  contamination. The two populations overlap completely in spread units and
  separate cleanly in absolute ones. The reason is physical — a male voice sits
  about an octave below these talents however variable the talent is — so the
  natural scale is absolute.
- **The baseline follows the talent's own trend**, rather than being one
  career-wide number. Some talents genuinely drift most of an octave across a
  career. A fixed centre puts both ends of that near the band edge, so the pass
  would trim the largest real trends in the corpus — the worst failure
  available, since those trends are the finding.
- **The band is asymmetric: tight below, loose above.** Every observed
  contamination is a lower voice, and below the baseline there is a clean gap
  between the worst legitimate clip and the mildest contaminated one. Above it
  there is no gap at all: a deliberate character performance sits *closer* to
  the baseline than one talent's entirely ordinary wide range. A tight upper
  bound therefore destroys real data and catches nothing, so the upper bound
  exists only as a backstop for physically implausible readings.

Where the baseline cannot be estimated — too few months — no band is applied.
Rejecting against a guess would be inventing a standard rather than measuring
against one.

Three failures are accepted rather than papered over. A **character
performance** cannot be separated from a wide natural range by pitch, so it is
kept. An **instrument** tracked as a voice can sit dead centre of a talent's
normal range, so nothing pitch-based sees it. And a **processed game voice** —
a character voice from the game itself, pitch-shifted rather than obviously
male — is neither clearly another speaker nor clearly off-pitch, which makes it
the hardest case of all.

### The thresholds are coarse, and should stay that way

They were fitted against a small set of human listening labels, on which
several nearby thresholds score identically. That means the data cannot
distinguish them, so the values should be read as "about this much", never
tuned finer, and never narrowed toward the data.

Revisiting them properly means more labels, not more arithmetic: flag a fresh
batch, listen, extend the label set, and re-score. Until then the honest claim
is that the pass removes the clearest contamination without removing anything a
listener called fine — not that it is optimal.

### Detection is the weaker lever

A gate can only remove what it recognises. Sampling more clips per month and
aggregating with a median makes contamination *harmless* rather than merely
detectable, and it does so for the cases no detector sees — including all three
accepted failures above. Where the two compete for effort, density wins; see
[`sampling.md`](sampling.md).

## What the gate deliberately does not do

- It does not reject a clip for implausible jitter, shimmer, HNR, brightness, or
  formants. Those features have no defensible per-clip plausibility band in this
  material.
- It does not verify speaker identity. An unannounced guest, a second voice on a
  shared channel, or game and video dialogue can pass every gate.
- It does not detect a plausible-but-wrong pitch reading. A clip whose true
  pitch is misread within the plausible range looks identical to a correct one.

The last two are covered, partially, by cross-clip checks rather than by the
gate — see [`statistics.md`](statistics.md).

## Review flags

Separate from pass/fail, and non-destructive: a clip is **flagged for review**
when it disagrees with its own same-month, same-talent counterpart by more than
a musically implausible interval. Two clips of one person weeks apart do not
differ by an octave; when they appear to, one of them is wrong even though both
may pass every gate.

Flags do not remove data. They mark it for a human, and they are exported
alongside the measurements so a reader can see how much of a series carries one.
