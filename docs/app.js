"use strict";

// Caption text mirrors series.py's _WHAT_IS_F0 / _QC_FOOTER wording (no
// Python import path available from a static page, so the strings are
// copied here — keep them in sync if series.py's wording changes).
const WHAT_IS_F0 =
  "Median F0 = typical pitch of voiced speech in a ~90 s clip of a public " +
  "chatting stream. Higher = higher-pitched voice.";
const QC_FOOTER =
  "QC excludes clips with too little voice, unstable pitch (BGM/tracker " +
  "error), or an implausible reading. Gaps are missing/failed data, never 0 Hz.";
const CUTE_MATURE_CAPTION =
  "Acoustic correlates, not a vibe rating. Percentile = equal-weight average " +
  "of each talent's z-scored mean F0, brightness, and pitch dynamism vs. " +
  "every other talent shown here with QC-pass data.";
const RADAR_CAPTION =
  "One shape per talent: each axis is that talent's percentile (0=lowest, " +
  "100=highest) vs. the whole registered corpus on that metric, computed " +
  "independently per axis (not a combined cute/mature score). A talent " +
  "missing an axis (no QC-pass data on that metric, or fewer than 2 " +
  "talents corpus-wide have it) simply has no vertex there.";

// Each talent's official/fandom-recognized image color, where confidently
// known. A talent NOT in this map falls back to Plotly's default
// categorical palette (assignTalentColor below) rather than a guessed
// hex value — colors are identity, worth getting right or not at all.
// Values below (Himemori Luna through Omaru Polka) are the official
// hololive schedule color, provided directly (2026-09-03) — the same
// source also has an official "subtitle" color per talent, not used
// here yet, see [[future: subtitle color]] if a lighter/alt variant is
// ever wanted. DEV_IS (ReGLOSS/FLOW GLOW), Hiodoshi Ao (graduated), and
// Takane Lui aren't in that source at all — STILL NEEDS VERIFICATION:
// Takane Lui, Rindo Chihaya, Isaki Riona, Kikirara Vivi, Todoroki
// Hajime, Otonose Kanade, Juufuutei Raden, Ichijou Ririka, Hiodoshi Ao,
// Koganei Niko, Mizumiya Su (fill in real values here once confirmed).
const TALENT_COLORS = {
  "Himemori Luna": "#F7ABD5",
  "Yukihana Lamy": "#6ABADF",
  "Tokino Sora": "#266AFF",
  "Robocosan": "#D192FE",
  "Shirakami Fubuki": "#43BFEF",
  "Oozora Subaru": "#E5FB67",
  "Nekomata Okayu": "#B190FC",
  "Inugami Korone": "#FEE039",
  "Kiryu Coco": "#F38514",
  "Tsunomaki Watame": "#F9AFB2",
  "Omaru Polka": "#B92731",
  "Hiodoshi Ao": "#3A7CA5",
  "Koganei Niko": "#E8973D",
  "Mizumiya Su": "#3AA6B9",
};

// Plotly's default categorical palette (d3.schemeCategory10-derived),
// used for any talent not in TALENT_COLORS — cycled independently of
// the hardcoded talents so no two visible traces share a color.
const FALLBACK_PALETTE = [
  "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
  "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
];
const _fallbackAssigned = new Map();
function assignFallbackColors(allNames) {
  // Deterministic regardless of selection/render order: walk every
  // talent alphabetically once at load, skipping anyone with a
  // hardcoded color, so the same un-mapped talent always gets the same
  // fallback color across a session.
  let i = 0;
  for (const name of [...allNames].sort()) {
    if (TALENT_COLORS[name]) continue;
    _fallbackAssigned.set(name, FALLBACK_PALETTE[i % FALLBACK_PALETTE.length]);
    i += 1;
  }
}
function talentColor(name) {
  return TALENT_COLORS[name] || _fallbackAssigned.get(name);
}

// Yearly metrics: keyed by the data.json "yearly" feature key. label/unit
// mirror the existing per-talent PNGs (__main__.py's _EXTRA_FEATURE_PLOTS).
const YEARLY_METRICS = {
  median_f0: { label: "F0 (Pitch) — Yearly", unit: "Median F0 (Hz)", caption: WHAT_IS_F0 },
  brightness_hz: {
    label: "Brightness — Yearly",
    unit: "Brightness (Hz)",
    caption:
      "Spectral centroid - a brighter/more forward vs. darker/warmer voice. " +
      "Mic/EQ and leftover BGM affect this too; trust the relative shape.",
  },
  dynamism_semitones: {
    label: "Pitch Dynamism — Yearly",
    unit: "Dynamism (semitones)",
    caption:
      "Mean semitone change between consecutive voiced frames - how much " +
      "the pitch actually moves, not just its static spread.",
  },
  jitter_local: {
    label: "Jitter — Yearly",
    unit: "Jitter (local, fraction)",
    caption: "Cycle-to-cycle pitch-period timing irregularity.",
  },
  shimmer_local: {
    label: "Shimmer — Yearly",
    unit: "Shimmer (local, fraction)",
    caption: "Cycle-to-cycle amplitude irregularity.",
  },
  hnr_db: {
    label: "Harmonics-to-Noise Ratio — Yearly",
    unit: "HNR (dB)",
    caption: "Higher = clearer/more tonal voice; lower = breathier/noisier.",
  },
  loudness_dynamics_db: {
    label: "Loudness Dynamics — Yearly",
    unit: "Loudness spread (dB)",
    caption:
      "Spread of frame loudness (RMS in dB) within a clip - animated volume " +
      "swings vs. a flat, even delivery.",
  },
  f1_hz: {
    label: "First Formant (F1) — Yearly",
    unit: "F1 (Hz)",
    caption:
      "Vocal-tract resonance most tied to jaw/tongue height - not a pitch " +
      "measure. Sensitive to the formant tracker's parameters and residual " +
      "isolation artifact; trust the relative shape.",
  },
  f2_hz: {
    label: "Second Formant (F2) — Yearly",
    unit: "F2 (Hz)",
    caption:
      "Vocal-tract resonance most tied to tongue front/back position - " +
      "with F1, the classic acoustic vowel-space axes.",
  },
  f3_hz: {
    label: "Third Formant (F3) — Yearly",
    unit: "F3 (Hz)",
    caption:
      "Higher vocal-tract resonance. Overall formant spacing (F1-F4) tracks " +
      "vocal tract length - the strongest acoustic correlate of perceived " +
      "voice maturity/body size measured here.",
  },
  f4_hz: {
    label: "Fourth Formant (F4) — Yearly",
    unit: "F4 (Hz)",
    caption: "Highest formant tracked - completes the F1-F4 picture with F1-F3.",
  },
};

let DATA = null;
let selectedTalents = new Set();
let branchFilter = "All";
let generationFilter = "All";

// Radar "timeline" mode: per-axis min-max range across EVERY talent and
// EVERY year (computed once at load, held fixed) so scrubbing the year
// slider moves points within a stable frame instead of the axes silently
// rescaling underneath you. Falls back to the corpus-wide rank percentile
// (radarOverall = true, the original behavior) when unchecked.
let RADAR_AXIS_RANGE = {};
let radarOverall = true;
let radarYear = null;
// Which metrics appear as radar axes — user-editable via checkboxes,
// defaults to every present yearly metric the first time the radar view
// is built (null means "not yet initialized", not "none selected").
let radarSelectedKeys = null;

// Trajectory view: X/Y (+ optional size) metric pickers and a from/to
// year range, all read straight from the already-present per-year
// "yearly" series — no new backend aggregation.
let ALL_YEARS = [];
let trajXKey = null;
let trajYKey = null;
let trajSizeKey = "none";
let trajFromYear = null;
let trajToYear = null;

let lastControlsMetric = null;

let tableSortColumn = "name";
let tableSortAscending = true;

function main() {
  // window.SITE_DATA comes from data.js (a <script>, not a fetch() —
  // fetch("data.json") is blocked by CORS when this page is opened via
  // file://, which is the whole point of a static, no-server site).
  DATA = window.SITE_DATA;
  selectedTalents = new Set(Object.keys(DATA.talents));
  assignFallbackColors(Object.keys(DATA.talents));
  computeRadarAxisRanges();
  computeYearRangeAndDefaultAxes();

  buildMetricPicker();
  buildFilterPickers();
  buildTalentList();

  // Delegated once on the stable container — buildTalentList() only ever
  // replaces its innerHTML (on every filter change too), so a listener
  // attached here keeps working without re-attaching (re-attaching inside
  // buildTalentList would stack a duplicate handler per filter change).
  document.getElementById("talent-list").addEventListener("change", (event) => {
    const name = event.target.value;
    if (event.target.checked) selectedTalents.add(name);
    else selectedTalents.delete(name);
    render();
  });

  document.getElementById("select-all").addEventListener("click", () => {
    selectedTalents = new Set(filteredNames());
    buildTalentList();
    render();
  });
  document.getElementById("select-none").addEventListener("click", () => {
    selectedTalents = new Set();
    buildTalentList();
    render();
  });
  const picker = document.getElementById("metric-picker");
  picker.addEventListener("change", () => {
    location.hash = picker.value;
    render();
  });

  // Deep-linkable views: #cute_mature, #yearly:brightness_hz, etc.
  const initial = location.hash.slice(1);
  if (initial && [...picker.options].some((o) => o.value === initial)) {
    picker.value = initial;
  }

  render();
}

function computeRadarAxisRanges() {
  RADAR_AXIS_RANGE = {};
  for (const key of Object.keys(YEARLY_METRICS)) {
    let min = Infinity;
    let max = -Infinity;
    for (const talent of Object.values(DATA.talents)) {
      for (const point of (talent.yearly && talent.yearly[key]) || []) {
        if (typeof point.median === "number") {
          min = Math.min(min, point.median);
          max = Math.max(max, point.median);
        }
      }
    }
    if (min <= max) RADAR_AXIS_RANGE[key] = { min, max };
  }
}

function computeYearRangeAndDefaultAxes() {
  const years = new Set();
  for (const talent of Object.values(DATA.talents)) {
    for (const points of Object.values(talent.yearly || {})) {
      for (const point of points) years.add(Number(point.year));
    }
  }
  ALL_YEARS = [...years].sort((a, b) => a - b);
  if (ALL_YEARS.length > 0) {
    radarYear = ALL_YEARS[ALL_YEARS.length - 1];
    trajFromYear = ALL_YEARS[0];
    trajToYear = ALL_YEARS[ALL_YEARS.length - 1];
  }
  const keys = presentYearlyKeys();
  trajXKey = keys.includes("median_f0") ? "median_f0" : keys[0];
  trajYKey = keys.includes("brightness_hz") ? "brightness_hz" : keys[1] || keys[0];
}

function presentYearlyKeys() {
  const present = new Set();
  for (const talent of Object.values(DATA.talents)) {
    for (const key of Object.keys(talent.yearly || {})) present.add(key);
  }
  return Object.keys(YEARLY_METRICS).filter((key) => present.has(key));
}

function buildMetricPicker() {
  const picker = document.getElementById("metric-picker");
  const options = [
    { value: "f0_monthly", label: "F0 (Pitch) — Monthly" },
    { value: "f0_quarterly", label: "F0 (Pitch) — Quarterly" },
  ];
  for (const key of presentYearlyKeys()) {
    options.push({ value: `yearly:${key}`, label: YEARLY_METRICS[key].label });
  }
  if (Object.keys(DATA.cute_mature).length > 0) {
    options.push({ value: "cute_mature", label: "Cute × Mature (F0 vs Brightness)" });
  }
  const anyPercentiles = Object.values(DATA.talents).some(
    (t) => Object.keys(t.percentiles || {}).length > 0
  );
  if (anyPercentiles) {
    options.push({ value: "radar", label: "Profile (Percentile Radar)" });
  }
  if (presentYearlyKeys().length >= 2 && ALL_YEARS.length > 0) {
    options.push({ value: "trajectory", label: "Trajectory (metric vs. metric, by year)" });
  }
  options.push({ value: "table", label: "Table (sortable)" });
  picker.innerHTML = options
    .map((o) => `<option value="${o.value}">${o.label}</option>`)
    .join("");
}

// Branch (EN/ID/DEV_IS/JP/Graduated/Unknown) is a coarse bucket derived
// server-side (site_data.py) from generation. "group" is a LIST of exact
// Holodex generation/unit strings (e.g. ["4th Generation (holoForce)"])
// — a list because a handful of talents (currently just Shirakami
// Fubuki: 1st Generation + GAMERS) hold more than one membership and
// should be filterable under either. Every talent has both fields,
// defaulted to ["Unknown"]/"Unknown" if the site was built without a
// roster.json (site_data.py rule 9), so these dropdowns are always
// populated, never missing a talent silently.
function distinctBranches() {
  return [...new Set(Object.values(DATA.talents).map((t) => t.branch || "Unknown"))].sort();
}

function buildFilterPickers() {
  const branchSel = document.getElementById("branch-filter");
  const genSel = document.getElementById("generation-filter");
  branchSel.innerHTML = ['<option value="All">All branches</option>']
    .concat(distinctBranches().map((b) => `<option value="${b}">${b}</option>`))
    .join("");
  // generation_order is computed server-side from real debut chronology
  // (site_data.py) — a plain alphabetical sort here would badly scramble
  // it (e.g. "DEV_IS ReGLOSS" (2023) sorting before "English -Myth-"
  // (2020)).
  genSel.innerHTML = ['<option value="All">All generations</option>']
    .concat((DATA.generation_order || []).map((g) => `<option value="${g}">${g}</option>`))
    .join("");
  branchSel.addEventListener("change", () => {
    branchFilter = branchSel.value;
    selectedTalents = new Set(filteredNames());
    buildTalentList();
    render();
  });
  genSel.addEventListener("change", () => {
    generationFilter = genSel.value;
    selectedTalents = new Set(filteredNames());
    buildTalentList();
    render();
  });
}

// Talents matching the current branch/generation filters (AND'd
// together) — the set the checkbox list displays, and what "All"/a
// filter change selects. Independent of selectedTalents: a filter change
// replaces the selection with exactly its matches (predictable "show me
// this branch" behavior), but a single checkbox toggle afterward only
// edits selectedTalents, not the filter. A multi-generation talent
// matches the generation filter if ANY of their memberships does.
function filteredNames() {
  return Object.keys(DATA.talents)
    .filter((name) => {
      const t = DATA.talents[name];
      const branchOk = branchFilter === "All" || (t.branch || "Unknown") === branchFilter;
      const genOk =
        generationFilter === "All" || (t.group || ["Unknown"]).includes(generationFilter);
      return branchOk && genOk;
    })
    .sort();
}

function buildTalentList() {
  const list = document.getElementById("talent-list");
  const names = filteredNames();
  list.innerHTML = names
    .map(
      (name) => `
      <label>
        <input type="checkbox" value="${name}" ${selectedTalents.has(name) ? "checked" : ""} />
        <span>${name}</span>
      </label>`
    )
    .join("");
}

function selectedNames() {
  return Object.keys(DATA.talents)
    .filter((name) => selectedTalents.has(name))
    .sort();
}

function render() {
  const metric = document.getElementById("metric-picker").value;
  const chart = document.getElementById("chart");
  const tableView = document.getElementById("table-view");
  const caption = document.getElementById("caption");
  const names = selectedNames();

  if (metric !== lastControlsMetric) {
    lastControlsMetric = metric;
    buildViewControls(metric);
  }

  const isTable = metric === "table";
  chart.style.display = isTable ? "none" : "";
  tableView.style.display = isTable ? "" : "none";

  if (names.length === 0) {
    Plotly.purge(chart);
    tableView.innerHTML = "";
    caption.textContent = "Select at least one talent.";
    return;
  }

  if (metric === "table") {
    renderTable(names);
    caption.textContent =
      "Click a column header to sort (click again to reverse). Percentile columns: 0=lowest, " +
      `100=highest vs. the whole registered corpus on that metric, independently per column.\n\n${QC_FOOTER}`;
  } else if (metric === "f0_monthly") {
    renderLineSeries(names, (t) => DATA.talents[t].monthly_f0_qc, "Median F0 (Hz)");
    caption.textContent = `${WHAT_IS_F0}\n\n${QC_FOOTER}`;
  } else if (metric === "f0_quarterly") {
    renderQuarterly(names);
    caption.textContent = `${WHAT_IS_F0}\n\n${QC_FOOTER}`;
  } else if (metric.startsWith("yearly:")) {
    const key = metric.slice("yearly:".length);
    renderYearly(names, key);
    caption.textContent =
      `${YEARLY_METRICS[key].caption}\n\n${QC_FOOTER}\n\n${percentileSummary(names, key)}`;
  } else if (metric === "cute_mature") {
    renderCuteMature(names);
    caption.textContent = CUTE_MATURE_CAPTION;
  } else if (metric === "radar") {
    if (radarSelectedKeys === null) radarSelectedKeys = new Set(presentYearlyKeys());
    if (radarSelectedKeys.size === 0) {
      Plotly.purge(chart);
      caption.textContent = "Select at least one metric (see the checkboxes above the chart).";
      return;
    }
    renderRadar(names, [...radarSelectedKeys]);
    caption.textContent = radarOverall
      ? RADAR_CAPTION
      : `${RADAR_CAPTION}\n\nShowing ${radarYear} only: each axis is min-max scaled ` +
        `against that metric's full range across every talent and every year (a fixed ` +
        `frame, so the shape's movement across years is meaningful) — not the overall ` +
        `rank percentile used in "Overall" mode.`;
  } else if (metric === "trajectory") {
    renderTrajectory(names);
    caption.textContent =
      "Each talent's path from year to year on the two chosen metrics (arrows show " +
      "direction of travel). Built from the same per-year values as the Yearly plots — " +
      `a year missing either metric for a talent is simply skipped.\n\n${QC_FOOTER}`;
  }
}

function buildViewControls(metric) {
  const el = document.getElementById("view-controls");
  if (metric === "radar") {
    const keys = presentYearlyKeys();
    if (radarSelectedKeys === null) radarSelectedKeys = new Set(keys);
    const metricsHtml = keys
      .map(
        (k) => `
        <label>
          <input type="checkbox" class="radar-metric-cb" value="${k}" ${radarSelectedKeys.has(k) ? "checked" : ""} />
          ${radarAxisLabel(k)}
        </label>`
      )
      .join("");
    el.innerHTML = `
      <div class="control-group">
        <label><input type="checkbox" id="radar-overall" ${radarOverall ? "checked" : ""} /> Overall (all years)</label>
      </div>
      <div class="control-group">
        <label for="radar-year">Year</label>
        <input type="range" id="radar-year" min="${ALL_YEARS[0]}" max="${ALL_YEARS[ALL_YEARS.length - 1]}"
          step="1" value="${radarYear}" ${radarOverall ? "disabled" : ""} />
        <span class="range-value" id="radar-year-value">${radarYear}</span>
      </div>
      <div class="metric-checklist-row">
        <span class="filter-label metric-checklist-label">Axes</span>
        <button type="button" id="radar-metrics-all" class="mini-btn">All</button>
        <button type="button" id="radar-metrics-none" class="mini-btn">None</button>
        <div class="metric-checklist">${metricsHtml}</div>
      </div>`;
    document.getElementById("radar-overall").addEventListener("change", (e) => {
      radarOverall = e.target.checked;
      document.getElementById("radar-year").disabled = radarOverall;
      render();
    });
    document.getElementById("radar-year").addEventListener("input", (e) => {
      radarYear = Number(e.target.value);
      document.getElementById("radar-year-value").textContent = radarYear;
      render();
    });
    el.querySelector(".metric-checklist").addEventListener("change", (e) => {
      if (e.target.checked) radarSelectedKeys.add(e.target.value);
      else radarSelectedKeys.delete(e.target.value);
      render();
    });
    document.getElementById("radar-metrics-all").addEventListener("click", () => {
      radarSelectedKeys = new Set(keys);
      buildViewControls(metric);
      render();
    });
    document.getElementById("radar-metrics-none").addEventListener("click", () => {
      radarSelectedKeys = new Set();
      buildViewControls(metric);
      render();
    });
  } else if (metric === "trajectory") {
    const keys = presentYearlyKeys();
    const opts = (selected) =>
      keys.map((k) => `<option value="${k}" ${k === selected ? "selected" : ""}>${radarAxisLabel(k)}</option>`).join("");
    const sizeOpts =
      `<option value="none" ${trajSizeKey === "none" ? "selected" : ""}>None</option>` + opts(trajSizeKey);
    const yearOpts = (selected) =>
      ALL_YEARS.map((y) => `<option value="${y}" ${y === selected ? "selected" : ""}>${y}</option>`).join("");
    el.innerHTML = `
      <div class="control-group"><label for="traj-x">X</label><select id="traj-x">${opts(trajXKey)}</select></div>
      <div class="control-group"><label for="traj-y">Y</label><select id="traj-y">${opts(trajYKey)}</select></div>
      <div class="control-group"><label for="traj-size">Size</label><select id="traj-size">${sizeOpts}</select></div>
      <div class="control-group"><label for="traj-from">From</label><select id="traj-from">${yearOpts(trajFromYear)}</select></div>
      <div class="control-group"><label for="traj-to">To</label><select id="traj-to">${yearOpts(trajToYear)}</select></div>`;
    document.getElementById("traj-x").addEventListener("change", (e) => {
      trajXKey = e.target.value;
      render();
    });
    document.getElementById("traj-y").addEventListener("change", (e) => {
      trajYKey = e.target.value;
      render();
    });
    document.getElementById("traj-size").addEventListener("change", (e) => {
      trajSizeKey = e.target.value;
      render();
    });
    document.getElementById("traj-from").addEventListener("change", (e) => {
      trajFromYear = Number(e.target.value);
      render();
    });
    document.getElementById("traj-to").addEventListener("change", (e) => {
      trajToYear = Number(e.target.value);
      render();
    });
  } else {
    el.innerHTML = "";
  }
}

function radarAxisLabel(key) {
  return YEARLY_METRICS[key].label.replace(/\s*—\s*Yearly$/, "");
}

// One row per talent: identity columns (name/branch/generation) plus a
// QC-pass-rate column and one percentile column per present yearly
// metric — percentiles, not raw values, so every numeric column is
// directly comparable/sortable on the same 0-100 scale regardless of
// the metric's native unit (Hz, dB, fraction, semitones, ...).
function tableColumns() {
  const cols = [
    { key: "name", label: "Talent", type: "text" },
    { key: "branch", label: "Branch", type: "text" },
    { key: "group", label: "Generation", type: "text" },
    { key: "qc_pass", label: "QC-pass %", type: "number" },
  ];
  for (const key of presentYearlyKeys()) {
    cols.push({ key: `pct:${key}`, label: `${radarAxisLabel(key)} pctl`, type: "number" });
  }
  return cols;
}

function tableCellValue(name, col) {
  const t = DATA.talents[name];
  if (col.key === "name") return name;
  if (col.key === "branch") return t.branch || "Unknown";
  if (col.key === "group") return (t.group || ["Unknown"]).join(", ");
  if (col.key === "qc_pass") {
    const s = t.qc_summary;
    return s && s.total > 0 ? (100 * s.qc_pass) / s.total : null;
  }
  if (col.key.startsWith("pct:")) {
    const value = (t.percentiles || {})[col.key.slice(4)];
    return value == null ? null : value;
  }
  return null;
}

function sortTableBy(columnKey) {
  if (tableSortColumn === columnKey) {
    tableSortAscending = !tableSortAscending;
  } else {
    tableSortColumn = columnKey;
    tableSortAscending = true;
  }
  render();
}

function renderTable(names) {
  const cols = tableColumns();
  const sortCol = cols.find((c) => c.key === tableSortColumn) || cols[0];
  const rows = names.map((name) => ({
    name,
    values: Object.fromEntries(cols.map((c) => [c.key, tableCellValue(name, c)])),
  }));
  // Missing values always sink to the bottom regardless of sort
  // direction — reversing the direction should never make "no data"
  // look like "the highest/lowest value".
  rows.sort((a, b) => {
    const av = a.values[sortCol.key];
    const bv = b.values[sortCol.key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = sortCol.type === "number" ? av - bv : String(av).localeCompare(String(bv));
    return tableSortAscending ? cmp : -cmp;
  });

  const headerHtml = cols
    .map((c) => {
      const arrow = c.key === tableSortColumn ? (tableSortAscending ? " ▲" : " ▼") : "";
      return `<th data-col="${c.key}">${c.label}${arrow ? `<span class="sort-arrow">${arrow}</span>` : ""}</th>`;
    })
    .join("");
  const bodyHtml = rows
    .map((row) => {
      const cells = cols
        .map((c) => {
          const value = row.values[c.key];
          if (c.key === "name") return `<td class="name-cell">${row.name}</td>`;
          if (value == null) return "<td>—</td>";
          return `<td>${c.type === "number" ? value.toFixed(0) : value}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  const wrap = document.getElementById("table-view");
  wrap.innerHTML = `<table class="data-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
  wrap.querySelector("thead").addEventListener("click", (event) => {
    const th = event.target.closest("th");
    if (th) sortTableBy(th.dataset.col);
  });
}

// Overall mode: axis value = corpus-wide rank percentile (site_data.py,
// unchanged). Per-year mode: axis value = that year's raw median,
// min-max scaled into 0-100 against RADAR_AXIS_RANGE — the fixed
// all-years/all-talents range per axis, computed once at load, so
// scrubbing the year slider moves the shape within a stable frame
// instead of the axes rescaling underneath you every year.
function radarValueForYear(name, key, year) {
  const points = (DATA.talents[name].yearly && DATA.talents[name].yearly[key]) || [];
  const point = points.find((p) => Number(p.year) === year);
  const range = RADAR_AXIS_RANGE[key];
  if (!point || !range || range.max === range.min) return null;
  return (100 * (point.median - range.min)) / (range.max - range.min);
}

function renderRadar(names, restrictToKeys) {
  const allKeys = restrictToKeys
    ? presentYearlyKeys().filter((key) => restrictToKeys.includes(key))
    : presentYearlyKeys();
  const valueFor = radarOverall
    ? (name, key) => (DATA.talents[name].percentiles || {})[key]
    : (name, key) => radarValueForYear(name, key, radarYear);
  const keys = allKeys.filter((key) =>
    names.some((name) => valueFor(name, key) != null)
  );
  const traces = names
    .map((name) => {
      const axisKeys = keys.filter((key) => valueFor(name, key) != null);
      return {
        type: "scatterpolar",
        name,
        r: axisKeys.map((key) => valueFor(name, key)),
        theta: axisKeys.map((key) => radarAxisLabel(key)),
        fill: "toself",
        opacity: 0.5,
        line: { color: talentColor(name) },
        marker: { color: talentColor(name) },
      };
    })
    .filter((trace) => trace.r.length > 0);
  Plotly.react("chart", traces, polarLayout());
}

function renderTrajectory(names) {
  const xKey = trajXKey;
  const yKey = trajYKey;
  const traces = names
    .map((name) => {
      const yearly = DATA.talents[name].yearly || {};
      const xByYear = new Map((yearly[xKey] || []).map((p) => [Number(p.year), p.median]));
      const yByYear = new Map((yearly[yKey] || []).map((p) => [Number(p.year), p.median]));
      const sizeByYear =
        trajSizeKey !== "none"
          ? new Map((yearly[trajSizeKey] || []).map((p) => [Number(p.year), p.median]))
          : null;
      const years = ALL_YEARS.filter(
        (y) =>
          y >= trajFromYear &&
          y <= trajToYear &&
          xByYear.has(y) &&
          yByYear.has(y) &&
          (!sizeByYear || sizeByYear.has(y))
      );
      const sizes = sizeByYear ? years.map((y) => sizeByYear.get(y)) : null;
      const scaledSizes = sizes
        ? (() => {
            const lo = Math.min(...sizes);
            const hi = Math.max(...sizes);
            return sizes.map((v) => (hi === lo ? 14 : 8 + (22 * (v - lo)) / (hi - lo)));
          })()
        : undefined;
      return {
        type: "scatter",
        mode: "lines+markers+text",
        name,
        x: years.map((y) => xByYear.get(y)),
        y: years.map((y) => yByYear.get(y)),
        text: years.map((y) => String(y)),
        textposition: "top center",
        textfont: { size: 9 },
        line: { color: talentColor(name) },
        marker: {
          color: talentColor(name),
          size: scaledSizes || 10,
          symbol: "arrow",
          angleref: "previous",
          standoff: scaledSizes ? undefined : 4,
        },
      };
    })
    .filter((trace) => trace.x.length > 0);
  Plotly.react(
    "chart",
    traces,
    layout({
      yLabel: YEARLY_METRICS[yKey].unit,
      xLabel: YEARLY_METRICS[xKey].unit,
      xType: "linear",
    })
  );
}

function renderLineSeries(names, seriesFn, yLabel) {
  const traces = names.map((name) => {
    const points = seriesFn(name) || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p[0]),
      y: points.map((p) => p[1]),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react("chart", traces, layout({ yLabel, xType: "category" }));
}

function renderQuarterly(names) {
  const traces = names.map((name) => {
    const points = DATA.talents[name].quarterly_f0 || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p.quarter),
      y: points.map((p) => p.mean),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react("chart", traces, layout({ yLabel: "Median F0 (Hz)", xType: "category" }));
}

// Percentile = this talent's rank (0-100) among every registered talent
// with QC-pass data on this exact metric (site_data.py's per-axis
// ranking, independent of the combined cute/mature score) — corpus-wide,
// so it does not change when the talent filter/selection narrows; only
// which talents' numbers are shown here does. Absent entirely (empty
// string) when fewer than 2 talents corpus-wide have this axis at all.
function percentileSummary(names, key) {
  const parts = names
    .filter((name) => key in (DATA.talents[name].percentiles || {}))
    .map((name) => `${name} ${Math.round(DATA.talents[name].percentiles[key])}`);
  if (parts.length === 0) return "";
  return `Percentile vs. corpus (this metric, 0=lowest 100=highest): ${parts.join(" · ")}`;
}

function renderYearly(names, key) {
  const traces = names.map((name) => {
    const points = (DATA.talents[name].yearly && DATA.talents[name].yearly[key]) || [];
    return {
      type: "scatter",
      mode: "lines+markers",
      name,
      x: points.map((p) => p.year),
      y: points.map((p) => p.median),
      connectgaps: false,
      marker: { color: talentColor(name) },
      line: { color: talentColor(name) },
    };
  });
  Plotly.react(
    "chart",
    traces,
    layout({ yLabel: YEARLY_METRICS[key].unit, xType: "category" })
  );
}

function renderCuteMature(names) {
  const rows = names
    .map((name) => ({ name, point: DATA.cute_mature[name] }))
    .filter((row) => row.point);
  const trace = {
    type: "scatter",
    mode: "markers+text",
    text: rows.map((r) => r.name),
    textposition: "top center",
    x: rows.map((r) => r.point.f0_mean),
    y: rows.map((r) => r.point.brightness_mean),
    marker: { size: 12, color: rows.map((r) => talentColor(r.name)) },
    hovertemplate:
      "%{text}<br>F0: %{x:.0f} Hz<br>Brightness: %{y:.0f} Hz<br>Percentile: %{customdata:.0f}<extra></extra>",
    customdata: rows.map((r) => r.point.percentile),
  };
  Plotly.react(
    "chart",
    [trace],
    layout({ yLabel: "Brightness (Hz)", xLabel: "Median F0 (Hz)", xType: "linear" })
  );
}

function layout({ yLabel, xLabel, xType }) {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e6e8ec" : "#1a1d23";
  const grid = dark ? "#2a2e38" : "#dde1e7";
  return {
    autosize: true,
    height: 480,
    margin: { t: 20, r: 20, b: 60, l: 60 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: fg },
    xaxis: {
      type: xType,
      title: xLabel || "",
      gridcolor: grid,
      tickangle: -45,
      // Plotly's default category order is "first seen across traces",
      // which interleaves each talent's own months/quarters/years in
      // whatever order their trace was added — not chronological. Every
      // category here is a sortable "YYYY-..." string, so ascending
      // string order IS chronological order.
      categoryorder: xType === "category" ? "category ascending" : undefined,
    },
    yaxis: { title: yLabel || "", gridcolor: grid },
    legend: { orientation: "h", y: -0.25 },
  };
}

function polarLayout() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e6e8ec" : "#1a1d23";
  const grid = dark ? "#2a2e38" : "#dde1e7";
  return {
    autosize: true,
    height: 520,
    margin: { t: 20, r: 40, b: 40, l: 40 },
    paper_bgcolor: "transparent",
    font: { color: fg },
    polar: {
      bgcolor: "transparent",
      radialaxis: { range: [0, 100], gridcolor: grid, color: fg },
      angularaxis: { gridcolor: grid, color: fg },
    },
    legend: { orientation: "h", y: -0.15 },
    showlegend: true,
  };
}

window.addEventListener("resize", () => {
  const chart = document.getElementById("chart");
  if (chart && chart.data) Plotly.Plots.resize(chart);
});

main();
