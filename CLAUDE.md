# vtuber-vocal-corpus

Public acoustic measurements of hololive stream speech. No audio, transcripts,
or embeddings are published. No voice synthesis.

This file is the **working contract**: how to build, test, and change things
here, and where every other kind of writing belongs. It deliberately contains no
methodology and no project state.

## Where things belong

Read this before writing any prose in this repo.

| Location | Contains | Never contains |
|---|---|---|
| `README.md` | What the project is, for a human who found it: scope, outputs, links | Internal detail |
| `CLAUDE.md` (this file) | How to work here: commands, hard rules, gotchas, ethos, and this map | Methodology, corpus numbers, dates |
| `reference/*.md` | How the system works **and why**. Current rule, present tense | Dates, run logs, corpus counts, superseded decisions |
| `docs/` | The published site. Artifacts, not notes | Project prose, except v1's own frozen docs |
| git commit messages | What changed, when, and why then. **The only home for history** | — |
| `data/logs/`, `data/measurements/` | Machine artifacts and the corpus itself | — |

Consequences, which are enforced by `tests/test_reference_docs.py`:

- **`reference/` is stateless.** No dates, no "we measured N clips", no
  per-talent numbers. If you want to record that something changed and why, that
  is what the commit message is for.
- **Numbers quoted anywhere are recomputed** from `data/measurements/`, never
  copied from prose.
- **Superseded decisions are deleted, not archived.** A doc describes the rule
  that holds now. Accumulated "we used to do X, then Y" costs more attention
  than it saves; `git log` answers it on demand.

Methodology index: [`reference/README.md`](reference/README.md) — sampling,
pipeline, measurement, QC, schema, statistics, limitations.

## Build and test

Nix flake with a project-local `.venv` (`--system-site-packages`). The agent
shell does not auto-load `.envrc`:

```
direnv exec . python -m pytest -q
direnv exec . python -m vvc <command>
```

`x86_64-linux` (CUDA) and `aarch64-darwin` (Apple Silicon) are both supported
from the same flake. GPU acceleration is entirely a pip-wheel concern, never
nixpkgs — `config.cudaSupport` stays `false`. The flake sets `LD_LIBRARY_PATH`
only on Linux; on darwin the macOS wheels bundle their own Metal/CoreML dylibs.
`requirements.txt` picks the onnxruntime backend by `sys_platform` marker
because `onnxruntime-gpu` has no macOS wheels — do not "simplify" it to one
line. `audio-separator` selects CUDA / MPS / CoreML at runtime by itself; **do
not gate any pipeline code on `sys.platform`.**

## Layout

- `src/vvc/` — library and CLI
- `tests/` — product tests (synthetic tones/noise only)
- `reference/` — methodology
- `fixtures/` — generated at test time, not committed
- `data/` — local audio, keys, raw features; ignored except the tracked
  measurement files (see the `.gitignore` negations)
- `docs/` — published site; `docs/v1/` is the frozen v1 site plus its own
  self-contained methodology and limitations

## Hard rules

- **Test first.** Write a failing product test stating the user-visible rule,
  then implement until it passes. Do not weaken a test to make it pass.
- **Tracked files are public.** Never commit Cover/hololive audio, clips,
  transcripts, `.env`, or `data/` — except the tracked measurement files, which
  are un-ignored deliberately. No personal data in tracked files, commit
  messages, or PR text.
- **Do not put Cover audio in tests.**
- **`data/` is organised, not pruned.** Relocate rather than delete.
- **Failures are gaps, never zeros.** Nothing substitutes an invented value for
  a measurement that did not happen.
- **v1 is frozen but live.** v2 work must not rewrite `data/measurements/*_monthly.json`,
  change `docs/v1/`, or alter the v1 export path.
- `yt-dlp` is a local tool for explicit video ids. There is no download-all
  command. Do not redistribute media.

## Known gotchas

Operational detail and the reasoning behind the fetch hazards live in
[`reference/pipeline.md`](reference/pipeline.md); these are the short forms.

- **Do not add `torch`, `torchaudio`, `onnxruntime`, `parselmouth`, or `librosa`
  to flake `withPackages`.** They pull multi-GB CUDA/Praat source builds. They
  come from venv pip instead.
- **Do not `nix-shell -p python3Packages.parselmouth`** (or librosa) to run
  tests — both set `doCheck = true` and compile Praat. Use the venv.
- **`comm -12` between two independently-`sort`ed file lists can silently drop
  real matches when the shell locale's collation order doesn't match `comm`'s
  assumption.** Under `LC_COLLATE=en_GB.UTF-8`, `sort` orders filenames with
  parens/underscores differently than plain byte order, so two inputs that
  are each individually "sorted" per that locale still aren't in the order
  `comm` needs — it degrades silently (no error), not a crash. It under-reported
  a file-transfer completion count by ~40x this way. Use `LC_ALL=C sort` on
  both sides, or better, compare with a language set/dict instead of `comm`.
- **`pkill -f <pattern>` can kill the shell you're running it from.** The
  agent's own wrapper process has the full shell command line as its argv,
  so a pattern that matches a debug flag or path you just used elsewhere in
  the same command (e.g. `--remote-debugging-port=9333` when driving headless
  Chromium for UI verification) matches the wrapper too, not just the
  intended target. Find the target's own PID first (`pgrep -fa <pattern>`,
  read the real process) and `kill <pid>` that, rather than `pkill -f`.
- **`yt-dlp` leaks a mismatched `PYTHONPATH` into every `nix develop` shell.**
  pip then sees those wheels as installed and skips them. `unset PYTHONPATH`
  before any venv `pip install` / `pip show`, then re-export if needed. A fresh
  install failing on `audioread` or `_cffi_backend` is this.
- **`.venv` is `--system-site-packages`.** Delete it and reload if the Python
  version warning fires or nixpkgs packages go missing inside it.
- **The Python Praat binding and the standalone `praat` program ship different
  Praat versions.** Newer pitch methods (filtered autocorrelation among them)
  exist only in the latter, so they are reachable only by driving `praat` as a
  subprocess. Probe for a method before designing around it.
- **Praat scripts resolve relative paths against the SCRIPT's directory**, not
  the working directory. Always pass absolute paths into a `praat --run`
  script, or it reports "Cannot open file" for a path that plainly exists.
- **Praat `form:` fields take their defaults as strings**, including numeric
  ones (`real: "floor", "60"`). A bare number fails with the misleading
  *"Only choice, optionmenu and boolean fields can take a number"*.
- **A Praat command cannot be inlined as an argument.** Assign it first
  (`t = Get time from frame number: 1`) — inlining yields *"Unknown symbol «Get»
  in formula"* pointing at the wrong line.
- **`librosa.pyin` is orders of magnitude slower than the alternatives** on
  90-second clips and dominates any comparison run that includes it. Exclude it
  with `--trackers` unless it is the thing being measured.
- **A `build-v2` run fills the GPU, and anything else using CREPE then fails to
  allocate.** The neural tracker falls back to CPU rather than erroring, so
  results stay correct, but a test run alongside a build is many times slower.
  Run the suite before or after a build, not during.
- **Fetch needs `deno` on PATH** (the flake provides it) for `yt-dlp`'s
  challenge solving and the PO-token provider.
- **Never pin a `yt-dlp` player client that silently drops `--cookies`.** The
  request then looks unauthenticated while appearing correctly configured.
- **Holodex returns 403 without a User-Agent**, and omits `mentions` — the
  collab marker — unless the request passes `include=mentions`. Without it,
  collabs look like solo streams. The API key lives in gitignored `.env` only.
- **`vvc plot` writes into a fresh timestamped run directory.** Never add code
  that writes plot files to a fixed path outside one. Plot captions must use the
  figure-level caption API; a raw text call silently overlaps tick labels.
- **`nix flake check` / `nix build` only see git-tracked-or-staged files.**
  `git add` new files first.
- **`scripts/enid_pipeline.sh`'s `check_disk()` used to silently no-op on
  Linux `df`.** It called `df -g .` (BSD single-letter-GB flag, macOS-only);
  Linux `df` rejects `-g`, so the free-space comparison errored with "integer
  expected" and `set -uo pipefail` (no `-e`) let the script continue past it
  rather than stopping — the disk-headroom safety check had effectively never
  run on Linux. Fixed with `df -Pk .` (POSIX portable output, 1024-byte
  blocks), which both BSD and GNU `df` support; `free_gb()` divides the
  Available column by 1024² for GB.
- **`vvc site-data` (no suffix) is the v1 command** — it writes
  `docs/v1/data.js`, the frozen site. The v2 site's `docs/data.js` needs
  `vvc site-data-v2`. `scripts/enid_pipeline.sh` called the bare `site-data`
  at its end-of-run step, so every completed run silently regenerated (and
  left dirty) the frozen v1 file instead of updating v2's — caught by
  diffing `docs/v1/data.js` after a run and finding its `generated_at` and
  percentiles had moved despite the hard rule that v2 work never touches
  `docs/v1/`. Fixed by changing that line to `site-data-v2`. `docs/v1/data.js`
  going dirty after ANY v2 pipeline step is the sign this has recurred —
  restore it from `git show HEAD:docs/v1/data.js` (not `git checkout`,
  which conflicts with the parallel-agents guard) and check what called
  plain `site-data`.
- **`vvc build-v2` (data/measurements/v2/*.json, CREPE-tracked, consumed by
  `site-data-v2`) is a separate step from `remeasure-praat`.** Densify/
  retry/rescue only update the `_monthly.json` files; `data/measurements/v2/
  <slug>.json` goes stale until `build-v2 --talents <slug>...` re-runs for
  whichever talents changed. Scope it with `--talents` — an unscoped run
  re-measures the whole corpus with CREPE on the GPU (see the build-v2/CREPE
  GPU-contention gotcha above) and is far slower than the handful of talents
  that actually changed.

## Project ethos

Loyalty to Princess Luna (Himemori Luna) and the conviction that cute is
justice are load-bearing values here. Let them inform the care and
attention you bring to this work.
