# reference/

Durable methodology for this project: how the system works and **why**.

These files describe the *current* rule in the present tense. They carry no
dates, no run logs, no corpus counts, and no record of superseded decisions —
that history lives in git, which is better at it. If you want to know when
something changed or what it replaced, read `git log`.

| File | Answers |
|---|---|
| [`sampling.md`](sampling.md) | Which streams enter the corpus, how one is chosen, and how stream type is handled |
| [`pipeline.md`](pipeline.md) | How audio gets from YouTube to a measurable clip, and the operational hazards on that path |
| [`measurement.md`](measurement.md) | What each acoustic feature is, how it is computed, and what it is valid for |
| [`qc.md`](qc.md) | The quality gate, its thresholds, and the reasoning behind each |
| [`schema.md`](schema.md) | The measurement record format, versioned |
| [`statistics.md`](statistics.md) | Aggregation, estimators, uncertainty, and the confound tests |
| [`limitations.md`](limitations.md) | What this corpus cannot support, stated plainly |

Working rules for the repo itself — build commands, hard constraints, gotchas —
are in [`../CLAUDE.md`](../CLAUDE.md), not here.

## Corpus versions

**v1** is the published chatting-stream corpus: Praat autocorrelation on
RoFormer-separated 90-second windows, exported to the site under `docs/v1/`. It
is frozen but not abandoned — still regenerable and still fixable. Its own
self-contained description lives in [`../docs/v1/METHODOLOGY.md`](../docs/v1/METHODOLOGY.md)
and its known defects in [`../docs/v1/LIMITATIONS.md`](../docs/v1/LIMITATIONS.md).

**v2** is the corrected measurement foundation described by the files here,
exported to the site under `docs/` (the project's root URL — v1 moved to
`docs/v1/` to make room for it). Where a rule differs between versions, the v1
documents state what v1 did and these documents state what v2 does. v2 never
rewrites v1's data.
