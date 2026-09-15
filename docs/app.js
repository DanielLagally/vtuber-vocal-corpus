"use strict";

// window.SITE_DATA_V2 comes from data.js (a <script>, not a fetch() —
// fetch("data.json") is blocked by CORS when this page is opened via
// file://, which is the whole point of a static, no-server site).
let DATA;

let selectedCompare = new Set();
let compareFamily = "pitch_level";
let compareMetric = "median_f0";
let compareMode = "series";
let compareGranularity = "quarterly";
let scatterX = "median_f0";
let scatterY = "brightness_hz";
let groupBy = "member";
let tableSort = { key: "typical", dir: -1 };

// ---------------------------------------------------------------------
// Colors — ported from v1's app.js unchanged (same curated brand-color
// map, same deterministic alphabetical fallback). Colors are identity:
// keeping one shared table here rather than a second hand-maintained
// copy is what keeps a talent's color consistent between v1 and v2.
// ---------------------------------------------------------------------

const TALENT_COLORS = {
  "Aki Rosenthal": "#4E7FFC",
  "AZKi": "#FC3488",
  "Hakos Baelz": "#D72517",
  "Koseki Bijou": "#6E5BF4",
  "Yuzuki Choco": "#FE739E",
  "Elizabeth Rose Bloodflame": "#C7383B",
  "Gigi Murin": "#FDB440",
  "Himemori Luna": "#F7ABD5",
  "Hakui Koyori": "#F09CC0",
  "Inugami Korone": "#FEE039",
  "Ninomae Ina'nis": "#62567E",
  "IRyS": "#8C1236",
  "Kikirara Vivi": "#FF90CC",
  "Natsuiro Matsuri": "#FDAB45",
  "Nerissa Ravencroft": "#2233FB",
  "Ookami Mio": "#C71E3E",
  "Ouro Kronii": "#20318B",
  "Shirogane Noel": "#ACBDC5",
  "Usada Pekora": "#7EC2FE",
  "Juufuutei Raden": "#3C7C71",
  "Isaki Riona": "#FE3480",
  "Takanashi Kiara": "#FF511C",
  "Takane Lui": "#B84A67",
  "Todoroki Hajime": "#B6B9FF",
  "Tokoyami Towa": "#BA92CA",
  "Yukihana Lamy": "#6ABADF",
  "Tokino Sora": "#266AFF",
  "Robocosan": "#D192FE",
  "Shirakami Fubuki": "#43BFEF",
  "Oozora Subaru": "#E5FB67",
  "Nekomata Okayu": "#B190FC",
  "Kiryu Coco": "#F38514",
  "Tsunomaki Watame": "#F9AFB2",
  "Omaru Polka": "#B92731",
  "Mizumiya Su": "#71E5FF",
};

const FALLBACK_PALETTE = [
  "#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
  "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
];
const _fallbackAssigned = new Map();
function assignFallbackColors(allNames) {
  let i = 0;
  for (const name of [...allNames].sort()) {
    if (TALENT_COLORS[name]) continue;
    _fallbackAssigned.set(name, FALLBACK_PALETTE[i % FALLBACK_PALETTE.length]);
    i += 1;
  }
}
function talentColor(name) {
  return TALENT_COLORS[name] || _fallbackAssigned.get(name) || "#888888";
}

// ---------------------------------------------------------------------
// Metric metadata — display label and unit only. Which family a metric
// belongs to, and whether it's robust/experimental, comes from
// DATA.families (reference/statistics.md and reference/measurement.md),
// not duplicated here.
// ---------------------------------------------------------------------

const METRIC_META = {
  median_f0: { label: "Median F0 (pitch)", unit: "Hz" },
  f0_iqr_semitones: { label: "F0 spread (IQR)", unit: "semitones" },
  dynamism_semitones: { label: "Pitch dynamism", unit: "semitones" },
  jitter_local: { label: "Jitter", unit: "fraction" },
  shimmer_local: { label: "Shimmer", unit: "fraction" },
  hnr_db: { label: "Harmonics-to-noise ratio", unit: "dB" },
  brightness_hz: { label: "Brightness", unit: "Hz" },
  f1_hz: { label: "Formant F1", unit: "Hz" },
  f2_hz: { label: "Formant F2", unit: "Hz" },
  f3_hz: { label: "Formant F3", unit: "Hz" },
  f4_hz: { label: "Formant F4", unit: "Hz" },
  voiced_fraction: { label: "Voiced fraction", unit: "" },
  loudness_dynamics_db: { label: "Loudness dynamics", unit: "dB" },
  background_ratio_db: { label: "Background ratio", unit: "dB" },
};

function allMetrics() {
  return DATA.families.flatMap((f) => f.metrics);
}

function familyOf(metric) {
  return DATA.families.find((f) => f.metrics.includes(metric));
}

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

function fmt(x) {
  if (!Number.isFinite(x)) return "—";
  return Math.abs(x) < 10 ? x.toFixed(2) : x.toFixed(0);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fillSelect(id, values) {
  document.getElementById(id).innerHTML = values
    .map((v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`)
    .join("");
}

function chartColors() {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return dark
    ? { fg: "#e6e8ec", grid: "#2a2e38" }
    : { fg: "#1a1d23", grid: "#dde1e7" };
}

// ---------------------------------------------------------------------
// Init and routing
// ---------------------------------------------------------------------

function main() {
  DATA = window.SITE_DATA_V2;
  assignFallbackColors(Object.keys(DATA.talents));
  document.getElementById("generated-at").textContent = DATA.generated_at;

  buildHomeFilters();
  buildFamilyPicker();
  buildScatterAxisPickers();

  wireNav();
  wireHomeControls();
  wireCompareControls();

  route();
  window.addEventListener("hashchange", route);
}

function route() {
  const hash = location.hash.slice(1);
  if (hash.startsWith("talent/")) {
    showProfile(decodeURIComponent(hash.slice("talent/".length)));
  } else if (hash === "compare") {
    showView("compare");
  } else {
    showView("home");
  }
}

function showView(view) {
  for (const el of document.querySelectorAll(".view")) el.hidden = true;
  document.getElementById(view + "-view").hidden = false;
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.classList.toggle("active", btn.dataset.view === view);
  }
  if (view === "home") buildTalentGrid();
  if (view === "compare") {
    renderChipPicker();
    renderCompare();
  }
}

function showProfile(name) {
  if (!DATA.talents[name]) {
    showView("home");
    return;
  }
  for (const el of document.querySelectorAll(".view")) el.hidden = true;
  document.getElementById("profile-view").hidden = false;
  for (const btn of document.querySelectorAll(".nav-btn")) btn.classList.remove("active");
  renderProfile(name);
}

function wireNav() {
  for (const btn of document.querySelectorAll(".nav-btn")) {
    btn.addEventListener("click", () => {
      location.hash = btn.dataset.view;
    });
  }
}

// ---------------------------------------------------------------------
// Home view
// ---------------------------------------------------------------------

function buildHomeFilters() {
  const branches = new Set();
  const gens = new Set();
  for (const t of Object.values(DATA.talents)) {
    branches.add(t.branch);
    for (const g of t.group) gens.add(g);
  }
  fillSelect("home-branch-filter", ["All branches", ...[...branches].sort()]);
  fillSelect(
    "home-generation-filter",
    ["All generations", ...DATA.generation_order.filter((g) => gens.has(g))]
  );
}

function wireHomeControls() {
  document.getElementById("talent-search").addEventListener("input", buildTalentGrid);
  document.getElementById("home-branch-filter").addEventListener("change", buildTalentGrid);
  document.getElementById("home-generation-filter").addEventListener("change", buildTalentGrid);
}

function sparklineSvg(points, color) {
  if (!points || points.length < 2) return "";
  const ys = points.map((p) => p.median);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const w = 100;
  const h = 22;
  const scaleX = (i) => (points.length === 1 ? w / 2 : (i / (points.length - 1)) * w);
  const scaleY = (y) => (maxY === minY ? h / 2 : h - ((y - minY) / (maxY - minY)) * h);
  const pts = ys.map((y, i) => `${scaleX(i).toFixed(1)},${scaleY(y).toFixed(1)}`).join(" ");
  return (
    `<svg class="talent-card-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">` +
    `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6"/></svg>`
  );
}

function talentCardHtml(name) {
  const t = DATA.talents[name];
  const legacy = t.legacy_fallback ? `<span class="legacy-tag">v1 only</span>` : "";
  const spark = sparklineSvg(t.metrics.median_f0.yearly, talentColor(name));
  return (
    `<button class="talent-card" data-name="${escapeHtml(name)}">` +
    `<span class="talent-swatch" style="background:${talentColor(name)}"></span>` +
    `<span class="talent-card-body">` +
    `<span class="talent-card-name">${escapeHtml(name)}${legacy}</span>` +
    `<span class="talent-card-meta">${escapeHtml(t.group[0] || "Unknown")} · ${t.n_pass}/${t.n_clips} clips</span>` +
    spark +
    `</span></button>`
  );
}

function buildTalentGrid() {
  const query = (document.getElementById("talent-search").value || "").toLowerCase();
  const branch = document.getElementById("home-branch-filter").value;
  const gen = document.getElementById("home-generation-filter").value;
  const grid = document.getElementById("talent-grid");
  const names = Object.keys(DATA.talents)
    .sort()
    .filter((name) => name.toLowerCase().includes(query))
    .filter((name) => branch === "All branches" || DATA.talents[name].branch === branch)
    .filter((name) => gen === "All generations" || DATA.talents[name].group.includes(gen));
  grid.innerHTML = names.map(talentCardHtml).join("");
  for (const card of grid.querySelectorAll(".talent-card")) {
    card.addEventListener("click", () => {
      location.hash = "talent/" + encodeURIComponent(card.dataset.name);
    });
  }
}

// ---------------------------------------------------------------------
// Profile view
// ---------------------------------------------------------------------

function metricRowHtml(name, metric) {
  const m = DATA.talents[name].metrics[metric];
  const meta = METRIC_META[metric];
  if (!m || !Number.isFinite(m.typical)) {
    return (
      `<div class="metric-row"><span class="metric-name">${escapeHtml(meta.label)}</span>` +
      `<span class="metric-sub">no data</span></div>`
    );
  }
  const ci = Number.isFinite(m.ci_low)
    ? ` · 95% CI ${fmt(m.ci_low)}–${fmt(m.ci_high)}`
    : "";
  const pct = m.percentile != null ? ` · ${Math.round(m.percentile)}th percentile` : "";
  const trend =
    m.trend && Number.isFinite(m.trend.slope_per_year)
      ? ` · ${m.trend.slope_per_year >= 0 ? "+" : ""}${m.trend.slope_per_year.toFixed(2)} ${meta.unit}/yr (r² ${m.trend.r2.toFixed(2)})`
      : "";
  const nf =
    m.noise_floor && m.noise_floor.median_abs_diff != null
      ? ` · noise floor ${fmt(m.noise_floor.median_abs_diff)} ${meta.unit}`
      : "";
  return (
    `<div class="metric-row"><span class="metric-name">${escapeHtml(meta.label)}</span>` +
    `<span><span class="metric-value">${fmt(m.typical)} ${escapeHtml(meta.unit)}</span>` +
    `<span class="metric-sub">${ci}${pct}${trend}${nf}</span></span></div>`
  );
}

function familyCardHtml(name, family) {
  const badge =
    family.robust === true ? "robust" : family.robust === false ? "experimental" : "context";
  const rows = family.metrics.map((metric) => metricRowHtml(name, metric)).join("");
  return (
    `<div class="family-card"><h3>${escapeHtml(family.label)} ` +
    `<span class="badge ${badge}">${badge}</span></h3>` +
    `<p class="family-note">${escapeHtml(family.note)}</p>${rows}</div>`
  );
}

function renderProfile(name) {
  const t = DATA.talents[name];
  const legacyNote = t.legacy_fallback
    ? `<p class="control-hint">This talent's source audio wasn't available for v2 ` +
      `re-measurement — every figure below is carried over from v1 rather than remeasured.</p>`
    : "";
  document.getElementById("profile-content").innerHTML =
    `<div class="profile-header">` +
    `<span class="profile-swatch" style="background:${talentColor(name)}"></span>` +
    `<h2>${escapeHtml(name)}</h2></div>` +
    `<p class="profile-meta">${escapeHtml(t.group.join(", ") || "Unknown")} (${escapeHtml(t.branch)}) &middot; ` +
    `${t.n_pass}/${t.n_clips} clips passing QC &middot; ${t.months_covered} months covered ` +
    `(${escapeHtml(t.first_month || "—")} to ${escapeHtml(t.last_month || "—")}) &middot; ` +
    `${Math.round(t.legacy_fraction * 100)}% legacy features</p>` +
    legacyNote +
    `<div class="profile-grid">` +
    `<div class="profile-families">${DATA.families.map((f) => familyCardHtml(name, f)).join("")}</div>` +
    `<div class="profile-radar-panel panel"><h2>Snapshot</h2>` +
    `<div id="profile-radar" class="chart" style="min-height:320px"></div>` +
    `<p class="control-hint">Percentile within the corpus, averaged per family. A family with ` +
    `no comparable talents sits at the neutral midpoint, not a measured value.</p></div></div>` +
    `<p class="profile-closing">This shows a typical value, its trend, and this talent's own ` +
    `same-month measurement noise floor for each metric — a trend smaller than the noise floor ` +
    `is not distinguishable from measurement error. It does not, on its own, establish a change ` +
    `in anyone's voice: career time and recording era are hard to separate in this material. See ` +
    `the repository's <code>reference/limitations.md</code> for what this corpus can and cannot ` +
    `support.</p>`;
  renderProfileRadar(name);
}

function renderProfileRadar(name) {
  const cats = DATA.families.map((f) => f.label);
  const vals = DATA.families.map((f) => {
    const pcts = f.metrics
      .map((m) => DATA.talents[name].metrics[m] && DATA.talents[name].metrics[m].percentile)
      .filter((v) => v != null);
    return pcts.length ? pcts.reduce((a, b) => a + b, 0) / pcts.length : 50;
  });
  const { fg, grid } = chartColors();
  const color = talentColor(name);
  Plotly.react(
    "profile-radar",
    [
      {
        type: "scatterpolar",
        r: [...vals, vals[0]],
        theta: [...cats, cats[0]],
        fill: "toself",
        line: { color },
        marker: { color },
      },
    ],
    {
      polar: {
        bgcolor: "transparent",
        radialaxis: { range: [0, 100], gridcolor: grid, color: fg },
        angularaxis: { gridcolor: grid, color: fg },
      },
      paper_bgcolor: "transparent",
      font: { color: fg, size: 10 },
      showlegend: false,
      margin: { t: 30, b: 30, l: 30, r: 30 },
    },
    { displayModeBar: false, responsive: true }
  );
}

// ---------------------------------------------------------------------
// Compare view
// ---------------------------------------------------------------------

// A graduated talent's stored `branch` is always "Graduated" (a status, not a
// region — see site_data.py) even though their `group` still carries their
// real generation. Branch CHIPS need the region a talent actually debuted
// in, or "JP" would silently exclude every graduated JP member — so this
// re-derives it from group[0] instead of trusting the stored field. Mirrors
// site_data.py's _branch_for_group; small and pure enough that duplicating
// it here (rather than threading it through data.js) is the simpler choice.
function regionOf(name) {
  const group = DATA.talents[name].group[0];
  if (!group) return DATA.talents[name].branch;
  if (group.startsWith("English")) return "EN";
  if (group.startsWith("Indonesia")) return "ID";
  if (group.startsWith("DEV_IS")) return "DEV_IS";
  return "JP";
}

function chipTokens() {
  const regions = [...new Set(Object.keys(DATA.talents).map(regionOf))].sort();
  const hasGraduated = Object.values(DATA.talents).some((t) => t.branch === "Graduated");
  return [
    { type: "all", label: "All" },
    ...regions.map((label) => ({ type: "region", label })),
    ...(hasGraduated ? [{ type: "graduated", label: "Graduated" }] : []),
    ...DATA.generation_order.map((label) => ({ type: "group", label })),
    ...Object.keys(DATA.talents).sort().map((label) => ({ type: "talent", label })),
  ];
}

function membersOf(token) {
  if (token.type === "talent") return [token.label];
  if (token.type === "all") return Object.keys(DATA.talents);
  if (token.type === "region") {
    return Object.keys(DATA.talents).filter((n) => regionOf(n) === token.label);
  }
  if (token.type === "graduated") {
    return Object.keys(DATA.talents).filter((n) => DATA.talents[n].branch === "Graduated");
  }
  return Object.keys(DATA.talents).filter((n) => DATA.talents[n].group.includes(token.label));
}

function renderChipPicker() {
  const query = (document.getElementById("compare-search").value || "").toLowerCase();
  const wrap = document.getElementById("chip-picker");
  const tokens = chipTokens().filter((tok) => tok.label.toLowerCase().includes(query));
  wrap.innerHTML = tokens
    .map((tok) => {
      const members = membersOf(tok);
      const allSelected = members.length > 0 && members.every((m) => selectedCompare.has(m));
      const swatch =
        tok.type === "talent"
          ? `<span class="chip-swatch" style="background:${talentColor(tok.label)}"></span>`
          : "";
      const cls = ["chip", tok.type !== "talent" ? "group" : "", allSelected ? "selected" : ""]
        .filter(Boolean)
        .join(" ");
      return (
        `<button type="button" class="${cls}" data-type="${tok.type}" ` +
        `data-label="${escapeHtml(tok.label)}">${swatch}${escapeHtml(tok.label)}</button>`
      );
    })
    .join("");
  for (const chip of wrap.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      const token = { type: chip.dataset.type, label: chip.dataset.label };
      const members = membersOf(token);
      const allSelected = members.every((m) => selectedCompare.has(m));
      for (const m of members) {
        if (allSelected) selectedCompare.delete(m);
        else selectedCompare.add(m);
      }
      renderChipPicker();
      renderCompare();
    });
  }
}

function buildFamilyPicker() {
  const el = document.getElementById("family-picker");
  el.innerHTML = DATA.families
    .map((f) => `<option value="${f.key}">${escapeHtml(f.label)}</option>`)
    .join("");
  el.value = compareFamily;
  el.addEventListener("change", () => {
    compareFamily = el.value;
    compareMetric = DATA.families.find((f) => f.key === compareFamily).metrics[0];
    buildMetricPicker();
    renderCompare();
  });
  buildMetricPicker();
}

function buildMetricPicker() {
  const fam = DATA.families.find((f) => f.key === compareFamily);
  const el = document.getElementById("metric-picker");
  el.innerHTML = fam.metrics
    .map((m) => `<option value="${m}">${escapeHtml(METRIC_META[m].label)}</option>`)
    .join("");
  el.value = compareMetric;
  el.onchange = () => {
    compareMetric = el.value;
    renderCompare();
  };
  document.getElementById("family-note").textContent = fam.note;
}

function buildScatterAxisPickers() {
  const opts = allMetrics()
    .map((m) => `<option value="${m}">${escapeHtml(METRIC_META[m].label)}</option>`)
    .join("");
  document.getElementById("scatter-x-picker").innerHTML = opts;
  document.getElementById("scatter-y-picker").innerHTML = opts;
  document.getElementById("scatter-x-picker").value = scatterX;
  document.getElementById("scatter-y-picker").value = scatterY;
}

function wireCompareControls() {
  document.getElementById("compare-search").addEventListener("input", renderChipPicker);
  document.getElementById("compare-select-none").addEventListener("click", () => {
    selectedCompare.clear();
    renderChipPicker();
    renderCompare();
  });
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.addEventListener("click", () => {
      compareMode = btn.dataset.mode;
      for (const b of document.querySelectorAll(".mode-btn")) b.classList.toggle("active", b === btn);
      document.getElementById("series-controls").hidden = compareMode !== "series";
      document.getElementById("scatter-controls").hidden = compareMode !== "scatter";
      renderCompare();
    });
  }
  document.getElementById("granularity-picker").addEventListener("change", (e) => {
    compareGranularity = e.target.value;
    renderCompare();
  });
  document.getElementById("group-by-picker").addEventListener("change", (e) => {
    groupBy = e.target.value;
    renderCompare();
  });
  document.getElementById("scatter-x-picker").addEventListener("change", (e) => {
    scatterX = e.target.value;
    renderCompare();
  });
  document.getElementById("scatter-y-picker").addEventListener("change", (e) => {
    scatterY = e.target.value;
    renderCompare();
  });
}

function updateValidityBanner() {
  const fam = DATA.families.find((f) => f.key === compareFamily);
  const banner = document.getElementById("validity-banner");
  if (fam.robust === false) {
    banner.className = "validity-banner experimental";
    banner.textContent = "Experimental — " + fam.note;
  } else if (fam.robust === null) {
    banner.className = "validity-banner";
    banner.textContent = fam.note;
  } else {
    banner.className = "validity-banner robust";
    banner.textContent = "";
  }
}

function renderCompare() {
  updateValidityBanner();
  const chart = document.getElementById("chart");
  const table = document.getElementById("table-view");
  const caption = document.getElementById("caption");
  if (selectedCompare.size === 0) {
    chart.hidden = false;
    table.hidden = true;
    Plotly.purge(chart);
    caption.textContent = "Select a talent or a group chip on the left to compare.";
    return;
  }
  if (compareMode === "series") {
    chart.hidden = false;
    table.hidden = true;
    renderCompareSeries();
  } else if (compareMode === "table") {
    chart.hidden = true;
    table.hidden = false;
    renderCompareTable();
  } else {
    chart.hidden = false;
    table.hidden = true;
    renderCompareScatter();
  }
}

// Plotly's default category order is "trace" — the order each x value is
// first seen across traces, not a sort. With several talents' spans
// stitched together that jumps back and forth in time. The period strings
// (YYYY-MM / YYYY-Qn / YYYY) are fixed-width and zero-padded, so a plain
// ascending string sort is chronological.
function periodXAxis(grid) {
  return { gridcolor: grid, tickangle: -45, type: "category", categoryorder: "category ascending" };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// ---------------------------------------------------------------------
// "Group by" — member / generation / branch — applies across all three
// views below. Member mode is one entry per selected talent, using that
// talent's own precomputed, server-side stats (real bootstrap CI, career
// trend). Generation/branch mode is one entry per group PRESENT AMONG THE
// CURRENT SELECTION, aggregated client-side from its members' series —
// there is no server-side "group" record, so its uncertainty is a plain
// min-max range rather than a bootstrap CI, and its trend is fit over
// calendar time (member-level trends use each talent's own career time,
// which isn't comparable across members once aggregated).
// ---------------------------------------------------------------------

// A talent can hold more than one generation (e.g. Shirakami Fubuki: 1st
// Generation AND GAMERS) — grouping by generation counts her toward both,
// matching how the chip picker already treats multi-group membership.
// Grouping by branch uses regionOf (real region, graduated-inclusive) for
// the same reason region chips do: "Graduated" is a status, not a region.
function groupKeysFor(name) {
  if (groupBy === "branch") return [regionOf(name)];
  const groups = DATA.talents[name].group;
  return groups && groups.length ? groups : ["Unknown"];
}

function selectionEntries() {
  if (groupBy === "member") {
    return [...selectedCompare].sort().map((name) => ({ label: name, members: [name], isGroup: false }));
  }
  const namesByGroup = new Map();
  for (const name of selectedCompare) {
    for (const key of groupKeysFor(name)) {
      if (!namesByGroup.has(key)) namesByGroup.set(key, []);
      namesByGroup.get(key).push(name);
    }
  }
  return [...namesByGroup.keys()]
    .sort()
    .map((label) => ({ label, members: namesByGroup.get(label), isGroup: true }));
}

function entryColor(entry) {
  return entry.isGroup ? paletteColorFor(entry.label) : talentColor(entry.label);
}

// Deterministic per-label (not per-position) color, so a group's color
// stays the same across renders regardless of what else is selected.
function paletteColorFor(label) {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  return FALLBACK_PALETTE[hash % FALLBACK_PALETTE.length];
}

// One talent (isGroup=false): pass its own series straight through, with
// its real per-point CI. Several talents (isGroup=true): aggregate across
// members at each period present in the CURRENT granularity, taking the
// median of whichever members have a value there — a period no member
// covers is a gap, never interpolated.
function aggregatedSeries(members, metric, granularity) {
  if (members.length === 1) {
    return { series: DATA.talents[members[0]].metrics[metric][granularity] || [], hasRealCi: true };
  }
  const byPeriod = new Map();
  for (const name of members) {
    for (const p of DATA.talents[name].metrics[metric][granularity] || []) {
      if (!Number.isFinite(p.median)) continue;
      if (!byPeriod.has(p.period)) byPeriod.set(p.period, []);
      byPeriod.get(p.period).push(p.median);
    }
  }
  const periods = [...byPeriod.keys()].sort();
  const series = periods.map((period) => {
    const values = byPeriod.get(period);
    return { period, median: median(values), min: Math.min(...values), max: Math.max(...values), n: values.length };
  });
  return { series, hasRealCi: false };
}

function periodToMonthIndex(period, granularity) {
  if (granularity === "yearly") return parseInt(period, 10) * 12;
  if (granularity === "monthly") {
    const [y, m] = period.split("-").map(Number);
    return y * 12 + (m - 1);
  }
  const [y, q] = period.split("-Q").map(Number);
  return y * 12 + (q - 1) * 3;
}

// Same estimator as analysis.py's theil_sen/career_trend: median of
// pairwise slopes, median intercept, ordinary r². Ported to JS because a
// group's trend can only be computed client-side — there is no server-side
// record for an ad-hoc selection of members.
function theilSenTrend(xs, ys) {
  const pairs = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pairs.length < 2) return null;
  const slopes = [];
  for (let i = 0; i < pairs.length; i++) {
    for (let j = i + 1; j < pairs.length; j++) {
      if (pairs[j][0] !== pairs[i][0]) slopes.push((pairs[j][1] - pairs[i][1]) / (pairs[j][0] - pairs[i][0]));
    }
  }
  if (!slopes.length) return null;
  const slope = median(slopes);
  const intercept = median(pairs.map(([x, y]) => y - slope * x));
  const meanY = pairs.reduce((a, [, y]) => a + y, 0) / pairs.length;
  const ssTot = pairs.reduce((a, [, y]) => a + (y - meanY) ** 2, 0);
  const ssRes = pairs.reduce((a, [x, y]) => a + (y - (intercept + slope * x)) ** 2, 0);
  return { slope_per_year: slope * 12, r2: ssTot > 0 ? 1 - ssRes / ssTot : null };
}

// One summary per entry, for the Table and Scatter views. Member entries
// use the precomputed, server-side per-talent stats directly (so numbers
// here always agree with the Profile page); group entries are aggregated
// from the currently selected granularity.
function summaryFor(entry, metric, granularity) {
  if (!entry.isGroup) {
    const name = entry.members[0];
    const m = DATA.talents[name].metrics[metric];
    const t = DATA.talents[name];
    return {
      typical: m.typical,
      low: m.ci_low,
      high: m.ci_high,
      isRange: false,
      trend: m.trend && Number.isFinite(m.trend.slope_per_year) ? m.trend : null,
      nPass: t.n_pass,
      nClips: t.n_clips,
      noiseFloor: m.noise_floor && m.noise_floor.median_abs_diff != null ? m.noise_floor.median_abs_diff : null,
    };
  }
  const { series } = aggregatedSeries(entry.members, metric, granularity);
  if (series.length === 0) {
    return { typical: NaN, low: NaN, high: NaN, isRange: true, trend: null, nPass: 0, nClips: 0, noiseFloor: null };
  }
  const values = series.map((p) => p.median);
  const xs = series.map((p) => periodToMonthIndex(p.period, granularity));
  let nPass = 0;
  let nClips = 0;
  const floors = [];
  for (const name of entry.members) {
    const t = DATA.talents[name];
    nPass += t.n_pass;
    nClips += t.n_clips;
    const nf = t.metrics[metric].noise_floor;
    if (nf && nf.median_abs_diff != null) floors.push(nf.median_abs_diff);
  }
  return {
    typical: median(values),
    low: Math.min(...values),
    high: Math.max(...values),
    isRange: true,
    trend: theilSenTrend(xs, values),
    nPass,
    nClips,
    noiseFloor: floors.length ? median(floors) : null,
  };
}

// Base band opacity shrinks as more entries overlap — several 95% CI bands
// stacked on each other turn opaque and unreadable well before six lines,
// which is what was reported. Inverse-sqrt keeps a visible band at n=1
// while backing off quickly over the first few additions and flattening out
// rather than vanishing. Hover restores one band to full visibility (see
// attachSeriesHover below) so any single member is still checkable.
function bandAlphaHex(n) {
  const frac = Math.max(0.05, Math.min(0.3, 0.3 / Math.sqrt(n)));
  return Math.round(frac * 255)
    .toString(16)
    .padStart(2, "0");
}
const BAND_HOVER_ALPHA_HEX = "6e"; // ~0.43, the hovered member's own band
const BAND_DIM_ALPHA_HEX = "08"; // ~0.03, every other band while one is hovered

// Highlights one entry's band on hovering its line, dims the rest, and
// restores the count-scaled default on unhover. Listeners are re-attached
// on every render (old ones removed first) since Plotly.react keeps the
// same <div> but trace indices change between renders.
function attachSeriesHover(chartEl, pairs) {
  chartEl.removeAllListeners("plotly_hover");
  chartEl.removeAllListeners("plotly_unhover");
  const withBand = pairs.filter((p) => p.bandIdx != null);
  if (withBand.length === 0) return;
  const bandIdxs = withBand.map((p) => p.bandIdx);
  chartEl.on("plotly_hover", (ev) => {
    const lineIdx = ev.points[0].curveNumber;
    const hit = pairs.find((p) => p.lineIdx === lineIdx);
    if (!hit) return;
    const colors = withBand.map((p) => p.color + (p.bandIdx === hit.bandIdx ? BAND_HOVER_ALPHA_HEX : BAND_DIM_ALPHA_HEX));
    Plotly.restyle(chartEl, { fillcolor: colors }, bandIdxs);
  });
  chartEl.on("plotly_unhover", () => {
    Plotly.restyle(chartEl, { fillcolor: withBand.map((p) => p.color + p.baseAlphaHex) }, bandIdxs);
  });
}

function renderCompareSeries() {
  const meta = METRIC_META[compareMetric];
  const { fg, grid } = chartColors();
  const entries = selectionEntries();
  const withData = entries
    .map((entry) => ({ entry, ...aggregatedSeries(entry.members, compareMetric, compareGranularity) }))
    .filter((e) => e.series.length > 0);
  const baseAlphaHex = bandAlphaHex(withData.length);

  const traces = [];
  const hoverPairs = [];
  for (const { entry, series, hasRealCi } of withData) {
    const color = entryColor(entry);
    const xs = series.map((p) => p.period);
    const upper = series.map((p) => (hasRealCi ? p.ci_high : p.max));
    const lower = series.map((p) => (hasRealCi ? p.ci_low : p.min));
    const hasBand = upper.some((v, i) => Number.isFinite(v) && Number.isFinite(lower[i]));
    let bandIdx = null;
    if (hasBand) {
      const bandUpper = series.map((p, i) => (Number.isFinite(upper[i]) ? upper[i] : p.median));
      const bandLower = series.map((p, i) => (Number.isFinite(lower[i]) ? lower[i] : p.median));
      bandIdx = traces.length;
      traces.push({
        x: [...xs, ...xs.slice().reverse()],
        y: [...bandUpper, ...bandLower.slice().reverse()],
        fill: "toself",
        fillcolor: color + baseAlphaHex,
        mode: "lines",
        line: { width: 0 },
        showlegend: false,
        hoverinfo: "skip",
        type: "scatter",
      });
    }
    const label = entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label;
    const lineIdx = traces.length;
    traces.push({
      x: xs,
      y: series.map((p) => p.median),
      mode: "lines+markers",
      name: label,
      line: { color },
      marker: { color, size: 5 },
      type: "scatter",
    });
    hoverPairs.push({ bandIdx, lineIdx, color, baseAlphaHex });
  }
  Plotly.react(
    "chart",
    traces,
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: fg },
      xaxis: periodXAxis(grid),
      yaxis: { title: meta.label + (meta.unit ? ` (${meta.unit})` : ""), gridcolor: grid },
      legend: { orientation: "h" },
      margin: { t: 20 },
    },
    { responsive: true, displayModeBar: false }
  );
  attachSeriesHover(document.getElementById("chart"), hoverPairs);
  const nf = DATA.corpus_noise_floor[compareMetric];
  const nfLine =
    nf && nf.median_abs_diff != null
      ? ` Corpus noise floor for this metric: ${fmt(nf.median_abs_diff)} ${meta.unit}` +
        (nf.median_abs_semitones != null ? ` (${nf.median_abs_semitones.toFixed(2)} semitones)` : "") +
        " — a difference smaller than this is not distinguishable from measurement noise."
      : "";
  const bandLine =
    groupBy === "member"
      ? "Shaded band = 95% bootstrap CI (quarterly/yearly only — a single month's own " +
        "uncertainty is better read from its noise floor than from a 2-3 clip bootstrap)."
      : "Shaded band = min–max range across each group's SELECTED members at that period " +
        "(not a confidence interval); a group with one selected member shows no band. Reflects " +
        "your current selection, not the whole corpus.";
  const hoverHint =
    withData.length > 1 ? " Bands fade as more lines overlap — hover a line to isolate its own band." : "";
  document.getElementById("caption").textContent = bandLine + hoverHint + nfLine;
}

const TABLE_COLUMNS = [
  { key: "name", label: "Talent" },
  { key: "typical", label: "Typical" },
  { key: "ci", label: "95% CI", sortKey: "low" },
  { key: "trend", label: "Trend/yr", sortKey: "trend" },
  { key: "r2", label: "r²" },
  { key: "clips", label: "Clips (pass/total)", sortKey: "nPass" },
  { key: "noise_floor", label: "Noise floor" },
];

function renderCompareTable() {
  const meta = METRIC_META[compareMetric];
  const grouped = groupBy !== "member";
  const entries = selectionEntries();
  const rows = entries.map((entry) => {
    const s = summaryFor(entry, compareMetric, compareGranularity);
    return {
      name: entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label,
      typical: s.typical,
      low: s.low,
      ci: Number.isFinite(s.low) ? `${fmt(s.low)}–${fmt(s.high)}` : "—",
      trend: s.trend ? s.trend.slope_per_year : null,
      r2: s.trend && s.trend.r2 != null ? s.trend.r2 : null,
      nPass: s.nPass,
      clips: `${s.nPass}/${s.nClips}`,
      noise_floor: s.noiseFloor,
    };
  });

  const sortKey = tableSort.key === "name" ? "name" : TABLE_COLUMNS.find((c) => c.key === tableSort.key)?.sortKey || tableSort.key;
  rows.sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const aMissing = av == null || av === "";
    const bMissing = bv == null || bv === "";
    if (aMissing && bMissing) return 0;
    if (aMissing) return 1; // missing values always sort last, in either direction
    if (bMissing) return -1;
    if (typeof av === "string") return av.localeCompare(bv) * tableSort.dir;
    return (av - bv) * tableSort.dir;
  });

  const headerHtml = TABLE_COLUMNS.map((col) => {
    const label = col.key === "name" ? (grouped ? "Group" : "Talent") : col.key === "ci" ? (grouped ? "Range" : "95% CI") : col.label;
    const arrow = tableSort.key === col.key ? `<span class="sort-arrow">${tableSort.dir === 1 ? "▲" : "▼"}</span>` : "";
    return `<th data-key="${col.key}">${escapeHtml(label)}${arrow}</th>`;
  }).join("");
  const bodyHtml = rows
    .map(
      (r) =>
        `<tr><td class="name-cell">${escapeHtml(r.name)}</td><td>${fmt(r.typical)}</td>` +
        `<td>${r.ci}</td><td>${r.trend != null ? r.trend.toFixed(2) : "—"}</td>` +
        `<td>${r.r2 != null ? r.r2.toFixed(2) : "—"}</td><td>${r.clips}</td>` +
        `<td>${r.noise_floor != null ? fmt(r.noise_floor) : "—"}</td></tr>`
    )
    .join("");
  document.getElementById("table-view").innerHTML =
    `<table class="data-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
  for (const th of document.querySelectorAll(".data-table th")) {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (tableSort.key === key) tableSort.dir *= -1;
      else tableSort = { key, dir: key === "name" ? 1 : -1 };
      renderCompareTable();
    });
  }
  document.getElementById("caption").textContent = grouped
    ? `Typical (${meta.unit || "unitless"}) = median across each group's selected members' ` +
      `${compareGranularity} series (set on the Time series tab — it still governs this ` +
      "aggregation even though the granularity control is hidden outside that view); trend is " +
      "Theil-Sen over calendar time (not career time, which isn't comparable once members are " +
      "pooled). Click a column header to sort."
    : `Typical (${meta.unit || "unitless"}) = median of month medians. Trend = Theil-Sen slope ` +
      "over career months, fit on the talent's own monthly series. Click a column header to sort.";
}

function renderCompareScatter() {
  const xMeta = METRIC_META[scatterX];
  const yMeta = METRIC_META[scatterY];
  const { fg, grid } = chartColors();
  const entries = selectionEntries();
  const points = entries
    .map((entry) => ({
      name: entry.isGroup ? `${entry.label} (n=${new Set(entry.members).size})` : entry.label,
      x: summaryFor(entry, scatterX, compareGranularity).typical,
      y: summaryFor(entry, scatterY, compareGranularity).typical,
      color: entryColor(entry),
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  Plotly.react(
    "chart",
    [
      {
        type: "scatter",
        mode: "markers+text",
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        text: points.map((p) => p.name),
        textposition: "top center",
        textfont: { size: 9, color: fg },
        marker: { size: 12, color: points.map((p) => p.color) },
      },
    ],
    {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: fg },
      xaxis: { title: xMeta.label + (xMeta.unit ? ` (${xMeta.unit})` : ""), gridcolor: grid },
      yaxis: { title: yMeta.label + (yMeta.unit ? ` (${yMeta.unit})` : ""), gridcolor: grid },
      margin: { t: 20 },
    },
    { responsive: true, displayModeBar: false }
  );
  document.getElementById("caption").textContent =
    (groupBy === "member"
      ? "Each point is a talent's typical value on each axis (median of month medians)."
      : `Each point is a group's typical value on each axis, aggregated across its selected ` +
        `members' ${compareGranularity} series (set on the Time series tab).`) +
    " Pick any two metrics — replaces v1's hardcoded Cute×Mature pair with a general " +
    "comparison. A composite index built from correlated axes is exploratory by " +
    "construction; this plots the raw metrics instead of inventing one.";
}

main();
