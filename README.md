# vtuber-vocal-corpus

Acoustic measurements of hololive stream speech, tracked per talent over time.

Each month of a talent's streams is sampled, and short windows of speech are
measured for pitch, voice quality, and timbre. The results are published as an
interactive site where talents can be browsed, compared, and ranked by how
similar their voices are.

**Site: https://daniellagally.github.io/vtuber-vocal-corpus/**

Not affiliated with Cover Corporation. Source material is public YouTube
streams; the repository contains measurements only.

## What is measured

- **Pitch** — typical speaking pitch, its range, and how much it moves
- **Voice source and timbre** — airiness (H1*–H2*), clarity (CPP), spectral
  tilt, brightness, and formant spacing
- **Tempo** — speaking rate
- **Voice similarity** — how alike two talents sound, from speaker embeddings
  and from the measured metrics

Pitch is the most reliable measure. The others are affected by microphones,
recording era, and audio processing, and are marked experimental on the site.
See [`reference/limitations.md`](reference/limitations.md) before drawing
conclusions from any number.

## Data

- `docs/data.js` — the site's data: per-talent summaries, monthly, quarterly,
  and yearly series, and closest-voice rankings.
- `data/measurements/` — per-clip measurements behind the original (v1) site.

The earlier version of the site is at
[`/v1/`](https://daniellagally.github.io/vtuber-vocal-corpus/v1/).

## Running

Requires Nix with flakes. Supported on `x86_64-linux` (CUDA) and Apple Silicon.
The first shell entry installs a Python environment of roughly 6 GB.

```
direnv exec . python -m pytest -q      # tests
direnv exec . python -m vvc --help     # all commands
```

Commands that read the corpus without changing it:

```
python -m vvc reliability      # measurement noise floor
python -m vvc voice-validate   # how well the metrics identify speakers
python -m vvc analyse          # career trends
python -m vvc export           # flat CSV tables
```

The site is static: open `docs/index.html` directly.

## Documentation

[`reference/`](reference/README.md) describes the method: sampling, pipeline,
measurement, quality control, record format, statistics, and limitations.

## License

MIT. See [`LICENSE`](LICENSE).
