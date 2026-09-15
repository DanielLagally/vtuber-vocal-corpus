# Pipeline

How audio gets from YouTube to a measurable clip, and the operational hazards on
that path.

## Stages

1. **Catalog** (`catalog.py`, `holodex.py`) — cache a channel's video listing
   from Holodex, filter for eligibility, score, and pick per month. Uses the
   cached listing; re-picking costs no API calls.
2. **Fetch** (`fetch.py`) — download the stream's audio with `yt-dlp` and cut
   the analysis section locally with `ffmpeg`. Sequential only.
3. **Window** (`windows.py`) — scan the fetched audio on a grid and slice the
   best-scoring fixed-length window. The recorded offsets are into the *fetched*
   file, not the original stream.
4. **Isolate** (`isolate.py`) — run the vocal separator over the window,
   producing a vocals stem and a residual stem. Whether this runs is conditional
   on the background-energy index; see [`measurement.md`](measurement.md).
5. **Measure** (`praat_features.py`, `features.py`, `measure.py`) — extract
   features from the clip and attach a QC verdict.
6. **Aggregate** (`series.py`, `site_data.py`) — build series, plots, and the
   site export from measurement files.

`densify.py` is the batch engine that drives stages 2–5 to bring each month up
to the target clip count, pipelining CPU work while keeping fetch sequential.

## Commands

Acquisition and the v1 corpus: `catalog`, `fetch`, `window`, `isolate`,
`measure`, `densify`, `plot`, `site-data`.

The v2 corpus and its evidence base are read-only with respect to v1:

| Command | Does |
|---|---|
| `bakeoff` | Compares pitch trackers on synthetic signals with known F0 and on real audio, and measures how far each feature moves under re-encoding |
| `reliability` | The measurement's own noise floor, plus review flags for same-month clips that disagree implausibly |
| `restore` | Pulls archived raw fetches back and replays each record's stored window, recovering offloaded clips without re-fetching from YouTube |
| `build-v2` | Re-measures local audio into `data/measurements/v2/`, replaying each record's stored window |
| `analyse` | Career trends, and the identifiable part of the career-versus-era confound |
| `export` | The four flat tables: clips, talent-month, talent summary, correlations |

`build-v2` writes only under its own output directory. The v1 measurement files
are inputs to it and are never modified.

Measurement operates on the **whole clip file**. The window offsets recorded on
a record are metadata describing where the clip came from; re-applying them to
an already-windowed file is a bug that has been made before.

## Fetch: download in full, cut locally

`yt-dlp`'s own section-download path falls back to streaming at roughly playback
speed for older videos, which is minutes per clip. Downloading the full audio via
range requests and cutting locally with `ffmpeg` is an order of magnitude
faster. The full download is deleted once the cut succeeds.

The separator subprocess carries a hard timeout. A hang in uninterruptible I/O
has deadlocked an unattended batch before; on timeout the clip is skipped like
any other failure.

## Fetch: the YouTube access gate

This is the least stable part of the system and the one most likely to waste a
long run. The essentials:

**Two different rejections exist and they are not the same problem.**

- *"Sign in to confirm you're not a bot"* — a session-level challenge. It
  affects the whole batch, so it triggers a **hard stop** (`BotCheckDetected`,
  surfaced as `stopped_early` in the densify summary) rather than skipping
  through the remaining ids and burning them all.
- *"Video unavailable" / `UNPLAYABLE`* on a video that is demonstrably public —
  a per-video proof-of-origin gate. This skips and continues; the month becomes
  a gap.

**Cookies and client must be compatible.** Some `yt-dlp` player clients silently
discard `--cookies` entirely because they do not support them. Pairing one of
those with a cookie file produces requests that look unauthenticated while
appearing configured correctly — a failure mode that reads as an IP block and is
not one. Never pin a client that drops cookies.

**Proof-of-origin tokens and an authenticated session are complementary, not
substitutes.** `yt-dlp` only requests a token from a provider when its internal
per-client policy demands one, and an authenticated session rarely trips that
policy — so a video caught by the gate fails outright with no fallback attempted.
Forcing the token request regardless of policy is what makes the provider
actually engage. Cookies plus a forced token recover ids that neither recovers
alone. Forcing the request is safe with no provider installed: it warns and
falls back.

**The gate is probabilistic.** The identical id, client, cookie, and token
combination can succeed and fail minutes apart. There is no fixed per-video
denylist to work around and no client swap that clears it reliably. Fetch
retries a failing id a small number of times for exactly this reason. Expect
reduced month coverage for talents whose archives fall inside an active gate
window, and treat an unusually low `added` count relative to months targeted as
a signal to check the gate even when nothing stopped early.

**A token provider's own token expires.** Success rate degrades toward the
token's lifetime rather than failing outright, so a long run can quietly stop
recovering anything while the provider still answers health checks. The provider
wrapper restarts itself well inside that lifetime rather than waiting for a
crash. If recovery craters partway through a run with no code change, check the
provider's uptime before concluding the gate got stricter.

**Cookies are consumable.** `yt-dlp` rewrites the cookie jar it is handed after
every run, and YouTube rotates session cookies mid-batch, so feeding it the real
export degrades that export until it no longer authenticates. `fetch_audio`
copies the cookie file to a per-call disposable temp and hands `yt-dlp` that, so
the user's file stays byte-for-byte intact. When the bot check still recurs
after this, it is genuine server-side expiry — re-export rather than debug.

Export cookies from a **private/incognito** window per the `yt-dlp` wiki's
cookie-export method; live browser-profile extraction produces rotating cookies
that work less reliably here. A complete export includes the HttpOnly session
cookies; a partial export that omits them looks plausible and does not
authenticate.

**Some archives are simply gone.** Early-era streams are frequently privated,
including debut streams. No cookie or client change recovers a privated video.
This is a data-availability wall, not a pipeline bug, and it is expected when
backfilling any talent's earliest era.

## Storage and offload

Raw audio, windows, and stems are large and local-only. `offload.py` pushes a
finished raw file to a configured `rclone` remote and deletes the local copy only
after a confirmed upload; on any failure the local file is left untouched.

The archive is what makes re-measurement possible without re-fetching, which
matters because re-fetching is subject to everything above. Because window
offsets are stored on every record, a restored raw file can reproduce the exact
clip a record was measured from — so a re-measurement is a like-for-like
comparison rather than a fresh sample.

`data/` is organised, not pruned. Prefer relocating over deleting.

## Plots are a permanent record

Every plot invocation writes into a fresh timestamped run directory. It never
overwrites a previous run's output, and no code may write plot files to a fixed
path outside a run directory.

Every plot stands alone for a reader who has never seen this repo: a bold title
naming the talent and metric, a one-line plain-language subtitle, and a short
methodology caption. The caption goes through the figure-level caption API so
that the layout engine reserves space for it; a raw text call silently overlaps
rotated tick labels.
