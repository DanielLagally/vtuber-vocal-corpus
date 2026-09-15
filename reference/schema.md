# Measurement record schema

One JSON file per talent, one record per measured clip, under
`data/measurements/`. These contain video ids, numbers, and verdicts — no
audio — which is why they are the only part of `data/` that git tracks at all.

The v1 files are tracked, via explicit negations in `.gitignore`. The v2
directory is **not** tracked yet: whether the v2 corpus is published alongside
v1, replaces it, or waits until its methodology settles is a decision that has
not been made, and adding an ignore-rule negation would quietly make it.

## v1 (frozen)

`data/measurements/<talent>_monthly.json`, a flat list of records:

```json
{
  "id": "…",
  "month": "YYYY-MM",
  "score": 40.0,
  "window": {"start_s": 15.0, "end_s": 105.0},
  "features": {
    "median_f0": …, "f0_iqr": …, "voiced_fraction": …,
    "brightness_hz": …, "dynamism_semitones": …,
    "jitter_local": …, "shimmer_local": …, "hnr_db": …,
    "loudness_dynamics_db": …,
    "f1_hz": …, "f2_hz": …, "f3_hz": …, "f4_hz": …
  },
  "qc": {"pass": true, "reason": null},
  "model": "…",
  "tracker": "…"
}
```

`data/measurements/talents.json` is the registry mapping talent keys to display
names, branch, and generation.

**These files are not rewritten by v2.** They remain the input to the v1 site
export, and they are the baseline every v2 comparison is made against.

## v2

`data/measurements/v2/<talent>.json`. Same one-record-per-clip shape, extended
so that the analyses in [`statistics.md`](statistics.md) are actually possible.
What v1 lacks and v2 adds, and why each is needed:

| Field group | Why |
|---|---|
| Exact publish date, not just month | Regression on real time; auditing sampling; ordering within a month |
| `topic_id`, `reliability_band`, duration | Stream type as a covariate rather than an assumption ([`sampling.md`](sampling.md)) |
| F0 quartiles, voiced/total frame counts | Scale-free spread, and the ability to recompute a gate without re-reading audio |
| Background-energy index | Which clips actually contained competing sound ([`measurement.md`](measurement.md)) |
| Separation applied, and with which model | Separation is conditional, so the treatment must be recorded per clip |
| Formant voiced/total frame counts, formant ceiling | Makes voiced-gating auditable and the fixed-ceiling compromise visible |
| Tracker identity and full search range | A threshold derived from a tracker parameter must be re-derivable |
| Lever provenance | Which path produced this clip — first window, second window, stem hunt, second stream |
| Measured-at timestamp, corpus release id | Snapshot identity, so a published number can be traced to the data that produced it |

## Rules that hold for both versions

**A record is one clip.** There is no record type for a month, a summary, or an
aggregate; those are computed, never stored as if measured.

**Absent is absent.** A feature that could not be measured is `nan` or missing.
No record carries a substituted, defaulted, or interpolated value, and nothing
downstream creates one.

**Verdicts are recomputed, not trusted.** The stored `qc` block is a convenience
copy; consumers call the gate on the stored features. See [`qc.md`](qc.md).

**Provenance is recorded, not inferred.** Anything that varied between clips —
tracker, separator, window, treatment — is on the record. A corpus where the
treatment differs between clips is usable if the treatment is known and unusable
if it is not.

**A record whose audio is no longer reachable and whose features came from a
superseded configuration is marked as such**, and is excluded from corrected
analyses rather than silently mixed with re-measured records.
