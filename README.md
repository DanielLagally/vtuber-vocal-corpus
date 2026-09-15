# vtuber-vocal-corpus

Acoustic measurement of VTuber stream speech: pitch (F0), spectral brightness,
and voice-quality correlates, tracked per talent over time and published as
aggregate plots.

Not affiliated with Cover Corporation or any talent agency covered by this
project. Source material is public YouTube broadcasts.

## What is and isn't published

Only derived numbers. No audio, video, transcripts, embeddings, or speaker
models are published or committed, and no voice synthesis is attempted.

Audio *is* retained locally and in a private archive, because re-measuring the
corpus requires it. "Not published or redistributed" is the accurate claim.

## Scope

- Streams on a talent's own channel. Singing, covers, collabs, and
  official-org-channel content are excluded.
- Chatting and talk streams are preferred by the selection score, but **game
  streams and watchalongs are eligible** and are picked when a month has nothing
  better. Stream type is recorded on every measurement so it can be controlled
  for rather than assumed.
- One acoustic tracker applied uniformly across talents and time, so results are
  comparable within the corpus.
- Every measurement passes a documented QC gate. Failing clips are gaps, never
  zero-filled.

## What the numbers support

Median pitch supports descriptive comparison and long-run trends **in the
sampled stream voice**. It does not, on its own, establish that anyone's voice
changed: career time and recording era are hard to separate in this material,
and the tests that separate them are part of the v2 work. The other metrics —
brightness, voice quality, formants — are exploratory or experimental.

[`reference/limitations.md`](reference/limitations.md) states this in full, per
feature. Read it before quoting a number.

## Output

- **Live site: https://daniellagally.github.io/vtuber-vocal-corpus/** — the v2
  interactive comparison site: a talent directory, a per-talent profile page,
  and a compare view (time series, table, scatter) organised around the five
  correlated metric families in [`reference/statistics.md`](reference/statistics.md),
  with a robust/experimental badge on each. No build step; open
  `docs/index.html` directly, or serve via GitHub Pages.
- `docs/v1/` — the original interactive comparison site (talent selection,
  multiple metrics, cute/mature percentile scatter), frozen but still live at
  [`/vtuber-vocal-corpus/v1/`](https://daniellagally.github.io/vtuber-vocal-corpus/v1/)
  and still maintained. No build step; open `docs/v1/index.html` directly, or
  serve via GitHub Pages. Its own methodology and known defects are documented
  alongside it in [`docs/v1/METHODOLOGY.md`](docs/v1/METHODOLOGY.md) and
  [`docs/v1/LIMITATIONS.md`](docs/v1/LIMITATIONS.md).
- `data/measurements/` — the corpus: one JSON file per talent, tracked in git.
- `data/plots/runs/` — static per-run PNG plots (gitignored, generated locally).

## Running

Requires the Nix flake environment (`direnv exec .` or `nix develop`). Supported
on `x86_64-linux` (CUDA) and `aarch64-darwin` / Apple Silicon (Metal/CoreML).
The first shell entry builds a project-local `.venv` from pip (~6 GB: torch +
onnxruntime + audio-separator).

```
direnv exec . python -m pytest -q          # test suite
direnv exec . python -m vvc --help         # CLI
```

Read-only commands worth knowing about, none of which modify the corpus:

```
python -m vvc reliability   # the measurement's own noise floor + review flags
python -m vvc bakeoff       # compare pitch trackers; measure encode sensitivity
python -m vvc analyse       # career trends and the era-confound test
python -m vvc export        # four flat CSV tables of the whole corpus
```

## Documentation

- [`reference/`](reference/README.md) — methodology: sampling, pipeline,
  measurement, QC, schema, statistics, limitations.
- [`CLAUDE.md`](CLAUDE.md) — the working contract: build, hard rules, gotchas,
  and where each kind of writing belongs.
- `git log` — project history. It is not duplicated in prose.

## License

MIT. See [`LICENSE`](LICENSE).
