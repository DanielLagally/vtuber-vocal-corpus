# Pipeline

## Stages

1. **Catalog** — cache each channel's video listing from Holodex, filter for
   eligibility, and pick streams per month (`catalog.py`).
2. **Fetch** — download the stream audio with `yt-dlp` and cut the analysis
   section locally with `ffmpeg` (`fetch.py`).
3. **Window** — choose the 90-second speech window (`windows.py`).
4. **Separate** — when the clip contains competing sound, isolate the voice
   with a neural vocal separator (`isolate.py`).
5. **Measure** — extract features and apply quality checks (`acoustics.py`,
   `voice_source.py`, `quality.py`).
6. **Aggregate** — build series, similarity rankings, and the site data
   (`series_v2.py`, `similarity.py`, `site_data_v2.py`).

`densify.py` runs stages 2–5 in bulk to bring each month up to its target
number of clips.

## Commands

| Command | Does |
|---|---|
| `densify` | Fetch, window, separate, and measure new clips |
| `segment` | Measure a chosen stretch of one stream as a one-off |
| `build-v2` | Re-measure the corpus from local audio (`--features-only` reuses each clip's recorded audio choice) |
| `embed` | Compute speaker embeddings for voice similarity |
| `site-data-v2` | Write the site data (`docs/data.js`) |
| `reliability` | Report the measurement noise floor and flag disagreeing clips |
| `voice-validate` | Check how well the metrics and embeddings identify speakers |
| `bakeoff` | Compare pitch trackers and measure sensitivity to re-encoding |
| `analyse` | Career trends and era tests |
| `export` | Flat CSV tables of the corpus |

## Fetching from YouTube

Fetching is the least reliable stage.

- A "confirm you're not a bot" response affects the whole session, so a batch
  stops rather than skipping through the remaining videos.
- "Video unavailable" on a public video is a per-video check. The video is
  skipped and the month becomes a gap. The check is intermittent; the same
  request can fail and then succeed minutes later.
- Some `yt-dlp` player clients ignore `--cookies`. Using one makes requests
  unauthenticated while appearing configured.
- Cookies and a proof-of-origin token work together. The token request is
  forced, since `yt-dlp` otherwise rarely asks for one on an authenticated
  session.
- `yt-dlp` rewrites the cookie file it is given, so each call uses a temporary
  copy. Export cookies from a private browser window.
- Many early streams have been made private and cannot be recovered.

## Plots

Each plot run writes to a new timestamped directory and never overwrites an
earlier run.
