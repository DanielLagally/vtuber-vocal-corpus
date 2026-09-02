# vtuber-vocal-corpus

Acoustic measurement of VTuber chatting-stream speech: pitch (F0), spectral
brightness, and voice-quality correlates, tracked per talent over time and
published as aggregate plots.

Not affiliated with Cover Corporation or any talent agency covered by this
project. Source material is public YouTube broadcasts. No raw audio, video,
or transcripts are stored or redistributed — only derived numerical
aggregates.

## Scope

- Chatting/talk streams only. Singing, collabs, and official-channel content
  are excluded.
- One acoustic tracker (Praat autocorrelation) applied uniformly across
  talents and time, so results are comparable within the corpus.
- Every measurement passes a documented QC gate (voiced-fraction floor,
  pitch-tracker sanity bounds). Failing clips are gaps, never zero-filled.

## Output

- `docs/` — interactive comparison site (talent selection, multiple
  metrics, cute/mature percentile scatter). No build step; open
  `docs/index.html` directly, or serve via GitHub Pages.
- `data/plots/runs/` — static per-run PNG plots (gitignored, generated
  locally).

## Running

Requires the Nix flake environment (`direnv exec .` or `nix develop`):

```
direnv exec . python -m pytest -q          # test suite
direnv exec . python -m vvc --help         # CLI
```

Environment setup, tooling details, and known gotchas are documented in
`CLAUDE.md`.

## Methodology and status

`PLAN.md` is the working engineering log: locked-in product rules, what has
been measured so far, and open questions. It is the authoritative record of
project state, not this file.
