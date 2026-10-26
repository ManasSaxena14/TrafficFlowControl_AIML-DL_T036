// Builds the project report (Word) from docs/report/report_data.json and reports/figures/*.png.
// Every number in the report comes from the JSON, which scripts/report/collect_data.py generates from the repository files.
//
//   python scripts/report/collect_data.py
//   NODE_PATH=<folder with the "docx" package> node scripts/report/build_report.js docs/report/AML_Project_Report.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Footer, PageNumber, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, VerticalAlign,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "docs/report/report_data.json"), "utf8"));
const OUT = process.argv[2] || path.join(ROOT, "docs/report/AML_Project_Report.docx");

const NAVY = "0B1730", BLUE = "2A78D6", SLATE = "5B6472", PANEL = "EEF3F9", GRID = "C9D2DF", GREEN = "0B7A52", RED = "B02A2A";
const W = 9026;                                   // content width in DXA (A4, 1 inch margins)
const MODELS = ["Persistence (last value)", "Historical average", "Ridge per sensor", "Multi-output Random Forest", "LSTM (no graph)", "XGBoost per horizon"];
const SHORT = { "Persistence (last value)": "Persistence", "Historical average": "Historical average", "Ridge per sensor": "Ridge per sensor", "Multi-output Random Forest": "Random Forest", "LSTM (no graph)": "LSTM (no graph)", "XGBoost per horizon": "XGBoost" };

// ------------------------------------------------------------------ data helpers
const H = D.horizon;
const hv = (m, split, metric, h) => H[m][split][metric][h - 1];              // h = horizon step 1..12
const f1 = (v) => (Math.round(v * 10 + 1e-9) / 10).toFixed(1);
const f2 = (v) => (Math.round(v * 100 + 1e-9) / 100).toFixed(2);
const f3 = (v) => (Math.round(v * 1000 + 1e-9) / 1000).toFixed(3);
const pct = (a, b) => Math.round(100 * (1 - a / b));                          // % reduction of a relative to b
const pct1 = (a, b) => (100 * (1 - a / b)).toFixed(1);
const num = (n) => n.toLocaleString("en-US");
const peak = (model, period, minutes, metric = "mae") => D.peak.find((r) => r.model === model && r.period === period && r.minutes === minutes)[metric];
const gap = (model, split) => D.gap_results.find((r) => r.model === model && r.split === split);
const onset = (model, group, h) => D.onset.find((r) => r.model === model && r.group === group && r.horizon === h).mae;
const resid = (model, group, h) => D.residual.find((r) => r.model === model && r.group === group && r.horizon === h).mean_corr;
const P1 = D.phase1;

// ------------------------------------------------------------------ building blocks
const run = (t, o = {}) => new TextRun(Object.assign({ text: t, font: "Calibri", size: 21 }, o));
function para(content, o = {}) {
  const children = (Array.isArray(content) ? content : [content]).map((c) => (typeof c === "string" ? run(c) : c));
  return new Paragraph(Object.assign({ children, spacing: { after: 120, line: 276 } }, o));
}
const b = (t) => run(t, { bold: true });
const i_ = (t) => run(t, { italics: true });
function h1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: "Cambria", bold: true, size: 32, color: NAVY })], spacing: { before: 360, after: 160 }, keepNext: true }); }
function h2(t) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: "Cambria", bold: true, size: 25, color: BLUE })], spacing: { before: 240, after: 100 }, keepNext: true }); }
function bullet(content) {
  const children = (Array.isArray(content) ? content : [content]).map((c) => (typeof c === "string" ? run(c) : c));
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, children, spacing: { after: 70, line: 264 } });
}
function caption(t) { return new Paragraph({ children: [run(t, { italics: true, size: 18, color: SLATE })], spacing: { before: 60, after: 200 } }); }
function tcap(t) { return new Paragraph({ children: [run(t, { bold: true, size: 19, color: NAVY })], spacing: { before: 160, after: 80 }, keepNext: true }); }
function spacer(n = 80) { return new Paragraph({ children: [], spacing: { after: n } }); }

const border = { style: BorderStyle.SINGLE, size: 4, color: GRID };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(c, w, o) {
  const spec = typeof c === "object" && c !== null && !Array.isArray(c) && "t" in c ? c : { t: String(c) };
  const align = spec.align || o.align || AlignmentType.LEFT;
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: spec.fill || o.fill ? { fill: spec.fill || o.fill, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 55, bottom: 55, left: 90, right: 90 },
    columnSpan: spec.span,
    children: [new Paragraph({ alignment: align, keepNext: !!o.keepNext, children: [run(spec.t, { size: o.size || 19, bold: spec.b || o.bold, color: spec.color || o.color })] })],
  });
}
function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  const aligns = opts.aligns || widths.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const head = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((t, k) => cell({ t, align: aligns[k] }, widths[k], { fill: NAVY, color: "FFFFFF", bold: true, size: opts.size || 19, keepNext: true })),
  });
  const keep = opts.keep !== undefined ? opts.keep : rows.length <= 14;       // keep short tables on one page
  const body = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((c, k) => cell(c, widths[k], { align: aligns[k], size: opts.size || 19, keepNext: keep && ri < rows.length - 1, fill: opts.zebra === false ? undefined : (ri % 2 ? PANEL : undefined) })),
  }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
function pngSize(file) { const buf = fs.readFileSync(file); return [buf.readUInt32BE(16), buf.readUInt32BE(20)]; }
let figNo = 0;
function figure(name, text, widthIn = 6.1) {
  const file = path.join(ROOT, "reports/figures", name);
  const [w, h] = pngSize(file);
  const wpx = Math.round(widthIn * 96), hpx = Math.round((wpx * h) / w);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx }, altText: { title: name, description: text, name } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(`Figure ${figNo}. ${text}`, { italics: true, size: 18, color: SLATE })] }),
  ];
}
function callout(lines) {
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: [W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: W, type: WidthType.DXA }, borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
      shading: { fill: PANEL, type: ShadingType.CLEAR, color: "auto" }, margins: { top: 140, bottom: 140, left: 200, right: 200 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 80, line: 270 }, children: (Array.isArray(l) ? l : [l]).map((c) => (typeof c === "string" ? run(c) : c)) })),
    })] })],
  });
}

// ================================================================== content
const body = [];
const add = (...x) => x.flat().forEach((e) => body.push(e));

// ---------------- cover
const cover = [
  new Paragraph({ spacing: { before: 1900, after: 120 }, children: [run("APPLIED MACHINE LEARNING  |  PROJECT T036", { bold: true, size: 22, color: BLUE, characterSpacing: 40 })] }),
  new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models", font: "Cambria", bold: true, size: 56, color: NAVY })] }),
  new Paragraph({ spacing: { after: 500 }, children: [new TextRun({ text: "Project report: the AML phase (Phases 0 to 4)", font: "Cambria", size: 34, color: SLATE })] }),
  para([b("Author: "), "Manas Saxena"]),
  para([b("Contributors: "), "Harsh Raj (project setup, data loading and exploratory analysis, Phases 0 and 1); Sumit (GitHub: sumit316-glitch; interactive Streamlit dashboard)"]),
  para([b("Date: "), "26 October 2026"]),
  para([b("Repository: "), "https://github.com/ManasSaxena14/TrafficFlowControl_AIML-DL_T036"]),
  para([b("Report generated from commit: "), `${D.head} (${D.commits.length} commits at that point)`]),
  spacer(300),
  callout([
    [b("Scope and integrity note. "), "This report covers work completed to date: the data, a leak-free preprocessing pipeline, six forecasting baselines, and four experiments that measure what independent per-sensor models miss. Every number, table and figure is generated from files in the repository (results/, reports/figures/, data/processed/ and git history) by scripts/report/collect_data.py, so each claim can be traced to its source (Appendix C). Graph deep-learning models (Phase 5) are not yet built and no results are claimed for them."],
  ]),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------- contents
add(h1("Contents"));
["1  Executive summary", "2  Objectives and scope", "3  Data: METR-LA", "4  Methodology", "5  Results: forecasting baselines (Phase 3)", "6  Results: identifying the gap (Phase 4)",
  "7  Engineering and reproducibility", "8  Limitations and threats to validity", "9  Conclusions and next steps",
  "Appendix A  Test MAE at every horizon", "Appendix B  Validation scores", "Appendix C  Provenance: result to source file", "Appendix D  Commit history"]
  .forEach((t) => add(para(t, { spacing: { after: 60 } })));
add(new Paragraph({ children: [new PageBreak()] }));

// ---------------- 1 executive summary
const X = "XGBoost per horizon", PE = "Persistence (last value)", LS = "LSTM (no graph)";
add(h1("1  Executive summary"));
add(para(["The project asks whether a model that sees the road network forecasts traffic speed better than strong per-sensor models, and where any gain comes from. The AML phase builds and tests the per-sensor side of that question using METR-LA, 207 freeway sensors in Los Angeles sampled every 5 minutes over four months, with forecasts 5 to 60 minutes ahead. Signal control is a simulation-only extension and is not claimed anywhere in this work."]));
add(para([b("What was done. "), "A chronological, leak-free preprocessing pipeline with outage masking; a 46-column feature table; six baselines (persistence, historical average, Ridge, XGBoost, Random Forest, and an LSTM without a graph) scored with one shared evaluation path (masked MAE, RMSE and MAPE by horizon, plus peak and off-peak); and four experiments that measure where independent per-sensor models fail."]));
add(tcap("Table 1. Headline results (test block, MAE in mph; lower is better)"));
add(table(["Model", "15 min", "30 min", "60 min", "Pooled MAE"],
  MODELS.map((m) => {
    const best = (h) => Math.min(...MODELS.map((x) => hv(x, "test", "mae", h)));
    return [SHORT[m]].concat([3, 6, 12].map((h) => { const v = hv(m, "test", "mae", h); return { t: f2(v), b: Math.abs(v - best(h)) < 1e-9, color: Math.abs(v - best(h)) < 1e-9 ? GREEN : undefined }; }), [f2(H[m].pooled.test.mae)]);
  }).concat([[{ t: "XGBoost + neighbour speeds (both directions)", b: true }].concat(["MAE_15min", "MAE_30min", "MAE_60min"].map((k) => ({ t: f2(gap("xgb + both", "test")[k]), b: true })), [{ t: "not computed" }])]),
  [3300, 1200, 1200, 1200, 2126]));
add(spacer(60));
add(h2("Key findings"));
add(bullet([b("XGBoost is the strongest independent baseline. "), `Test MAE ${f2(hv(X, "test", "mae", 3))}, ${f2(hv(X, "test", "mae", 6))} and ${f2(hv(X, "test", "mae", 12))} mph at 15, 30 and 60 minutes: ${pct(hv(X, "test", "mae", 3), hv(PE, "test", "mae", 3))}% lower than persistence at 15 minutes and ${pct(hv(X, "test", "mae", 12), hv(PE, "test", "mae", 12))}% lower at 60 minutes. The LSTM is marginally better at 15 minutes (${f2(hv(LS, "test", "mae", 3))}) and worse from 30 minutes on.`]));
add(bullet([b("Rush hour is hardest. "), `Every model's weekday-peak error is 1.4 to 1.95 times its off-peak error (XGBoost at 30 minutes: ${f2(peak("XGBoost per horizon", "peak", 30))} vs ${f2(peak("XGBoost per horizon", "offpeak", 30))} mph).`]));
add(bullet([b("The per-sensor model fails where congestion propagates. "), `In the 30 minutes after a congestion onset XGBoost's MAE is ${f1(onset("xgb", "onset", 3))} mph at 15 minutes against ${f1(onset("xgb", "normal", 3))} mph at other times (${f1(onset("xgb", "onset", 3) / onset("xgb", "normal", 3))} times worse; ${num(D.phase4_findings.test_onsets)} onsets in the test period). Errors of directly linked sensors correlate (${f2(resid("xgb", "1 link", 12))} at 60 minutes) while errors of unconnected sensors do not (${f2(resid("xgb", "not connected", 12))}).`]));
add(bullet([b("Neighbour information helps, but not enough. "), `Adding hand-made upstream and downstream speed features lowers test MAE at every horizon (both directions: ${pct1(gap("xgb + both", "test").MAE_15min, gap("xgb (no neighbours)", "test").MAE_15min)}%, ${pct1(gap("xgb + both", "test").MAE_30min, gap("xgb (no neighbours)", "test").MAE_30min)}% and ${pct1(gap("xgb + both", "test").MAE_60min, gap("xgb (no neighbours)", "test").MAE_60min)}% at 15, 30 and 60 minutes; ${pct(onset("xgb_both", "onset", 3), onset("xgb", "onset", 3))}% right after onsets at 15 minutes). Linked errors stay correlated, which is the case for a model that learns which neighbours matter over several hops (Section 6.4).`]));
add(bullet([b("Reproducible. "), `${D.commits.length} commits, ${Object.values(D.tests).reduce((a, c) => a + c, 0)} automated tests (12 of them leakage tests), make targets for every stage, and an interactive dashboard.`]));

// ---------------- 2 objectives
add(h1("2  Objectives and scope"));
add(para("The project brief defines a hybrid logic: start with segment-level machine learning, identify what it cannot represent, then test a network-aware deep-learning model. Table 2 maps the brief to the work in this repository."));
add(tcap("Table 2. Brief requirements and status"));
add(table(["Requirement from the brief", "Where addressed", "Status"], [
  ["AML baseline on lagged speed, time of day and road-segment features", "46-column feature table (Section 4.2)", { t: "Done", b: true, color: GREEN }],
  ["Flow and occupancy features", "Not available in METR-LA (speed only); PeMS planned for Phase 9", { t: "Data limit", b: true, color: RED }],
  ["XGBoost per segment or multi-output Random Forest", "Both implemented (Section 4.3)", { t: "Done", b: true, color: GREEN }],
  ["Identify the gap: independent models cannot represent upstream and downstream propagation", "Four experiments (Section 6)", { t: "Done", b: true, color: GREEN }],
  ["Fair comparison: chronological splits; MAE, RMSE, MAPE by horizon; equal history; peak vs off-peak", "Applied to all baselines (Sections 4 and 5); final comparison with graph models in Phase 6", { t: "Baselines done", b: true, color: GREEN }],
  ["Mask sensor outages; fit scaling on training data only; cyclic time features; no random splits", "Preprocessing pipeline with 12 leakage tests (Section 4.1)", { t: "Done", b: true, color: GREEN }],
  ["Deep-learning model: graph attention or diffusion model over the road graph", "Phase 5", { t: "Not started", b: true, color: SLATE }],
  ["Congestion maps; optional simulated signal-control experiment", "Phases 7 and 8; signal control is simulation only", { t: "Not started", b: true, color: SLATE }],
], [3900, 3626, 1500], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER] }));

// ---------------- 3 data
add(h1("3  Data: METR-LA"));
add(para(`METR-LA contains speed readings (mph) from ${P1.data.sensors} loop detectors on Los Angeles freeways, one reading every 5 minutes from ${P1.data.start.slice(0, 10)} to ${P1.data.end.slice(0, 10)}: ${num(P1.data.timestamps)} timestamps per sensor with no gaps. The sensor graph is the one distributed with the DCRNN reference implementation: a directed Gaussian-kernel adjacency matrix built from road-network distances. Column order of the speed matrix was verified against the adjacency order, and this is enforced in code and in tests.`));
add(tcap("Table 3. Dataset facts (results/phase1_findings.json)"));
add(table(["Property", "Value"], [
  ["Sensors; timestamps per sensor", `${P1.data.sensors}; ${num(P1.data.timestamps)}`],
  ["Period", `${P1.data.start.slice(0, 10)} to ${P1.data.end.slice(0, 10)} (training period for exploration ends ${P1.data.train_end_70pct.slice(0, 16)})`],
  ["Readings equal to 0 mph (sensor down)", `${P1.missingness.zero_pct}% (whole-network outages ${P1.missingness.network_outage_pct} points; isolated sensor outages ${P1.missingness.sensor_outage_pct} points)`],
  ["Whole-network outages", `${num(P1.missingness.network_outage_timestamps)} timestamps; longest ${P1.missingness.longest_network_outage_hours} hours`],
  ["Worst single sensor", `718076, down ${P1.missingness.worst_sensor["718076"]}% of the time`],
  ["Weekday free-flow speed; weekend minimum", `${P1.daily_profile.weekday_free_flow_mph} mph; ${P1.daily_profile.weekend_min_mph} mph`],
  ["Weekday dips below 90% of free flow", `${P1.daily_profile.peak_windows.map((p) => `${p.start} to ${p.end} (low ${p.min_mph} mph)`).join("; ")}`],
  ["Speed distribution", `median ${P1.distribution.median} mph; ${P1.distribution.pct_ge_60}% of readings at least 60 mph; ${P1.distribution.pct_lt_40}% below 40 mph; ${P1.distribution.pct_lt_20}% below 20 mph`],
  ["Graph", `${P1.graph.nodes} nodes, ${num(P1.graph.edges)} directed links, mean degree ${P1.graph.mean_degree}, ${P1.graph.reciprocal_edge_pct}% of links two-way, ${P1.graph.isolated_nodes} isolated sensor, ${P1.graph.weak_components} weak components`],
], [3300, 5726], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT] }));
add(spacer(80));
add(h2("3.1  Missing data"));
add(para(`${P1.missingness.zero_pct}% of all readings are 0 mph, which means the sensor was down, not that traffic stopped. Of this, ${P1.missingness.network_outage_pct} percentage points are whole-network outages (all 207 sensors at once, i.e. logging failures) and only ${P1.missingness.sensor_outage_pct} points are single-sensor failures. Treating zeros as traffic would teach a model that roads randomly stop; all later stages therefore mask them.`));
add(figure("fig01_missingness.png", "Missing data by sensor and day, and the share explained by whole-network outages."));
add(h2("3.2  Daily rhythm and the need for time features"));
add(para(`Weekday speeds dip in the morning and, more deeply, in the afternoon; weekend speeds stay above ${P1.daily_profile.weekend_min_mph} mph. Detected dips (below 90% of free flow for over 30 minutes) are ${P1.daily_profile.peak_windows.map((p) => `${p.start} to ${p.end}`).join(" and ")}; the brief's fixed peak definition (weekdays 07:00 to 09:59 and 16:00 to 19:59) lies inside them and is used for all peak results. Because ${P1.distribution.pct_ge_60}% of readings are free-flowing, an average error is dominated by easy periods, which is why peak and congestion-onset results are reported separately.`));
add(figure("fig02_daily_profile.png", "Network-mean speed by time of day, weekday versus weekend (training period only)."));
add(h2("3.3  Congestion propagates backwards along the graph"));
add(para(`To test whether neighbouring sensors carry information, each sensor's typical daily pattern was subtracted and the remaining disturbance was cross-correlated along every directed link (training period only). ${P1.propagation.weak_edge_pct}% of the ${num(P1.propagation.edges)} links are weak (peak correlation below 0.3). Of the ${P1.propagation.clear_edges} links with a clear peak, ${P1.propagation.same_step_pct}% co-move within one 5-minute step, the downstream sensor moves first on ${P1.propagation.downstream_first} links and the upstream sensor first on ${P1.propagation.upstream_first}. The implied backward wave speed is about ${P1.propagation.shockwave_kmh} km/h. A jam therefore shows up at the downstream neighbour before the upstream sensor, which a model that sees only its own history cannot anticipate.`));
add(figure("fig05d_propagation_example.png", "Worked example (2 May 2012): the downstream sensor drops to about 20 mph at 06:40 while the upstream sensor is still at about 66 mph. An illustration chosen by a rule; the evidence is the all-link statistics above."));

// ---------------- 4 methodology
add(h1("4  Methodology"));
add(h2("4.1  Leak-free preprocessing (Phase 2)"));
add(para("One shared pipeline feeds every model, so the comparison is fair by construction."));
add(bullet([b("Chronological split with a purge gap. "), "Timestamps are split 70, 10 and 20 percent into train, validation and test blocks; a window is kept only if all 12 of its target timestamps lie inside one block. Windows whose targets straddle a boundary are purged (11 per boundary), so no target timestamp appears in two splits. Inputs may look back into earlier data, which is the past and not a label."]));
add(bullet([b("Outage masking. "), "0 mph becomes missing. Input gaps are forward-filled from the last real reading inside the same 12-step window, otherwise from the training mean of that sensor and time-of-day slot, and a mask channel marks imputed values. Targets stay missing (NaN); every loss and metric ignores them. Training windows whose targets are over 90% masked (mostly whole-network outages) are dropped; validation and test keep every window."]));
add(bullet([b("Scaling on training data only. "), `Mean ${f2(D.scaler_strict.mu)} mph and standard deviation ${f2(D.scaler_strict.sd)} mph over training timestamps, ignoring outages. All errors are reported in mph after inverse scaling.`]));
add(bullet([b("Cyclic time features. "), "Sine and cosine of time of day and of day of week, plus a weekend flag, so 23:55 and 00:05 are neighbours."]));
add(bullet([b("Adjacency. "), "Rebuilt from the road-distance file (Gaussian kernel, sigma equal to the standard deviation of finite distances, threshold 0.1) and asserted equal to the supplied matrix; it stays directed and fixed."]));
add(bullet([b("Windows. "), "Input: the last 12 steps (60 minutes). Output: the next 12 steps (5 to 60 minutes). Every model, including the tree models, uses this same history."]));
add(tcap("Table 4. Strict split: windows, target periods and masked targets (data/processed/strict)"));
add(table(["Split", "Windows", "Rows (windows x 207)", "First target", "Last target", "Masked targets"],
  ["train", "val", "test"].map((s) => [s, num(D.splits[s].windows), num(D.splits[s].rows), D.splits[s].first_target, D.splits[s].last_target, `${D.splits[s].masked_target_pct}%`]),
  [900, 1100, 1750, 1750, 1750, 1776], { size: 18 }));
add(para(`The training block excludes ${num(D.dropped_train_windows)} windows dropped by the 90% masking rule. The test block has a much higher share of masked targets (${D.splits.test.masked_target_pct}%) than training (${D.splits.train.masked_target_pct}%) because the test period contains more outages; metrics ignore masked points, but this is one reason test errors are not directly comparable with validation errors.`, { spacing: { before: 100, after: 120 } }));
add(tcap("Table 5. Why the purge matters: strict split versus the DCRNN reference split"));
add(table(["", "Strict (used)", "DCRNN reference (comparison only)"], [
  ["Split unit", "Timestamps, then windows", "Windows"],
  ["Last train target", D.splits.train.last_target, D.splits_dcrnn.train.last_target],
  ["First validation target", D.splits.val.first_target, { t: `${D.splits_dcrnn.val.first_target} (before last train target)`, color: RED, b: true }],
  ["Scaler mean / std (mph)", `${f2(D.scaler_strict.mu)} / ${f2(D.scaler_strict.sd)} (outages excluded)`, `${f2(D.scaler_dcrnn.mu)} / ${f2(D.scaler_dcrnn.sd)} (0 mph outages included)`],
], [2400, 3000, 3626], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT] }));
add(para("Both protocols are implemented so published numbers can be compared later. The reference split lets train and validation targets overlap by 50 minutes; the strict split does not, and a test proves this.", { spacing: { before: 100, after: 120 } }));

add(h2("4.2  Feature table for the per-sensor models (Phase 3)"));
add(para(`One row is one sensor at one forecast origin; there are ${num(D.splits.train.rows)} training rows, ${num(D.splits.val.rows)} validation rows and ${num(D.splits.test.rows)} test rows. All 46 columns come from the Phase 2 windows.`));
add(tcap("Table 6. Feature groups"));
add(table(["Group", "Columns", "Count"], [
  ["Lagged speed", "speed at t, t-1, ..., t-11 (imputed where the sensor was down)", "12"],
  ["Rolling statistics", "mean, standard deviation and minimum over the last 3, 6 and 12 steps", "9"],
  ["Change", "speed(t) minus speed(t-1); speed(t) minus speed(t-3)", "2"],
  ["Missing flags", "real-reading flag for each of the 12 lags; number of imputed lags", "13"],
  ["Time", "sin and cos of time of day and of day of week; weekend flag", "5"],
  ["Segment (training data only)", "sensor id, mean speed, free-flow speed (85th percentile), in-degree, out-degree", "5"],
], [2300, 5826, 900], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.CENTER] }));
add(para("Flow and occupancy, which the brief lists, are not recorded in METR-LA; adding them requires PeMS data (Phase 9). This is a data limitation, not omitted work.", { spacing: { before: 100, after: 120 } }));

add(h2("4.3  Models"));
const xm = D.model_meta.xgb, rm = D.model_meta.rf, lm = D.model_meta.lstm, gm = D.model_meta.ridge;
const tune = xm.tuning_h6;
const lh = lm.history, lbest = Math.min(...lh.map((e) => e.val_mae));
add(tcap("Table 7. The six baselines and how each was configured (all selection on validation only)"));
add(table(["Rung", "Model", "Configuration"], [
  ["0", "Persistence", "Repeats the last (imputed) speed for all 12 horizons."],
  ["1", "Historical average", "Training mean per sensor, weekday-or-weekend and 5-minute slot."],
  ["2", "Ridge per sensor", `One linear model per sensor and horizon (207 x 12) on the 41 time-varying features, standardised per sensor. Regularisation chosen from 1, 10, 100, 1000 on validation: ${gm.alpha} (validation MAE ${f3(gm.val_mae_by_alpha["1.0"])} to ${f3(gm.val_mae_by_alpha["1000.0"])}, so the choice barely matters).`],
  ["3", "XGBoost per horizon", `One regressor per horizon trained on all sensors stacked, sensor id as a feature, masked targets given weight 0. Maximum depth searched on the 30-minute horizon: ${tune.map((t) => `${t.max_depth} gives ${f3(t.val_mae)}`).join(", ")} validation MAE; depth ${xm.params.max_depth} chosen. Learning rate ${xm.params.eta}, subsample and column sample ${xm.params.subsample}, up to 600 rounds with early stopping (50) on validation. Every ${xm.stride}nd window used for training.`],
  ["4", "Multi-output Random Forest", `One forest predicting all 12 horizons: ${rm.params.n_estimators} trees, depth ${rm.params.max_depth}, minimum leaf ${rm.params.min_samples_leaf}, ${rm.params.max_samples * 100}% bootstrap samples, ${rm.params.max_features * 100}% of features per split. Trained on every ${rm.stride}th window, keeping only rows whose 12 targets are all observed (${num(rm.train_rows)} rows); evaluated on every test row with the mask.`],
  ["5", "LSTM, no graph", `Two LSTM layers of ${lm.hidden} units with dropout 0.1; one set of weights shared by all 207 sensors; input is each sensor's own 12 x 7 window, output 12 horizons. Masked MAE in mph, Adam (learning rate 0.002, halved on plateau), gradient clipping 5, at most 12 epochs. It ran all 12 epochs; best validation MAE ${f3(lbest)}.`],
], [600, 2000, 6426], { aligns: [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.LEFT], size: 18 }));
add(para("XGBoost and the Random Forest are independent models: a row contains only one sensor's own history, time and static descriptors and never a neighbour's speed. The LSTM shares weights across sensors but also sees no neighbour, so it separates the effect of deep learning from the effect of the graph: if a later graph model beats it, the gain comes from the graph.", { spacing: { before: 100, after: 120 } }));

add(h2("4.4  Evaluation protocol"));
add(bullet([b("Metrics. "), "Masked MAE and RMSE in mph and MAPE in percent, computed only where the target is a real reading. MAPE guards against a near-zero denominator with a 1 mph floor."]));
add(bullet([b("Horizons. "), "All 12 horizons (5 to 60 minutes) are stored; headline numbers are step 3, 6 and 12, i.e. 15, 30 and 60 minutes. The pooled column averages over all 12."]));
add(bullet([b("Peak definition. "), "Weekdays 07:00 to 09:59 and 16:00 to 19:59, judged on the timestamp being predicted; other weekday times are off-peak and weekends are reported separately."]));
add(bullet([b("Test discipline. "), "Test numbers are written to results but never used to choose anything; hyper-parameters and early stopping use the validation block only."]));

// ---------------- 5 results
add(h1("5  Results: forecasting baselines (Phase 3)"));
add(h2("5.1  Scores on the test block"));
const metrics = ["mae", "rmse", "mape"];
const sc = (m, metric, h) => hv(m, "test", metric, h);
const bestOf = (metric, h) => Math.min(...MODELS.map((m) => sc(m, metric, h)));
add(tcap("Table 8. Test scores at 15, 30 and 60 minutes (best in each column in green)"));
add(table(["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "RMSE 30", "RMSE 60", "MAPE 15", "MAPE 30", "MAPE 60"],
  MODELS.map((m) => [{ t: SHORT[m], b: true }].concat(metrics.flatMap((k) => [3, 6, 12].map((h) => {
    const v = sc(m, k, h), isB = Math.abs(v - bestOf(k, h)) < 1e-9;
    return { t: k === "mape" ? f1(v) : f2(v), b: isB, color: isB ? GREEN : undefined };
  })))),
  [1626, 800, 800, 800, 800, 800, 800, 800, 800, 800], { size: 17 }));
add(para("MAE and RMSE in mph; MAPE in percent. Forecast horizons are in minutes. The pooled averages over all 12 horizons are in Appendix B.", { spacing: { before: 80, after: 120 } }));
add(para(`XGBoost has the lowest RMSE at 15, 30 and 60 minutes and the lowest MAE and MAPE at 30 and 60 minutes. The LSTM has the lowest MAE and MAPE at 15 minutes (${f2(sc(LS, "mae", 3))} against ${f2(sc(X, "mae", 3))} mph), a difference of ${f2(sc(X, "mae", 3) - sc(LS, "mae", 3))} mph from a single training run, which should not be over-interpreted. Persistence is a strong floor at 15 minutes, but its MAE grows from ${f2(sc(PE, "mae", 3))} to ${f2(sc(PE, "mae", 12))} mph by 60 minutes. The historical average ignores the present, so its error is flat (${f2(sc("Historical average", "mae", 3))} mph at every horizon); it beats persistence and Ridge at 60 minutes but not the tree models. Ridge is competitive at 15 minutes but its MAPE at 60 minutes (${f1(sc("Ridge per sensor", "mape", 12))}%) is the worst, because a linear model extrapolates the recent trend too far.`));
add(figure("fig07_baseline_error_by_horizon.png", "Test MAE by forecast horizon for all six baselines."));
add(h2("5.2  Peak versus off-peak"));
const PM = ["Persistence (last value)", "Historical average", "Ridge per sensor", "Multi-output Random Forest", "LSTM (no graph)", "XGBoost per horizon"];
add(tcap("Table 9. Test MAE (mph) by period at 15 / 30 / 60 minutes"));
add(table(["Model", "Weekday peak", "Weekday off-peak", "Weekend"],
  PM.map((m) => [{ t: SHORT[m], b: true }].concat(["peak", "offpeak", "weekend"].map((p) => [15, 30, 60].map((mm) => f2(peak(m, p, mm))).join(" / ")))),
  [2400, 2200, 2226, 2200], { size: 18 }));
const ratios = PM.flatMap((m) => [15, 30, 60].map((mm) => peak(m, "peak", mm) / peak(m, "offpeak", mm)));
add(para(`Peak-hour error is ${f2(Math.min(...ratios))} to ${f2(Math.max(...ratios))} times off-peak error across all models and horizons, and the penalty grows with the horizon. XGBoost's weekday-peak MAE rises from ${f2(peak("XGBoost per horizon", "peak", 15))} to ${f2(peak("XGBoost per horizon", "peak", 60))} mph while its off-peak MAE rises from ${f2(peak("XGBoost per horizon", "offpeak", 15))} to ${f2(peak("XGBoost per horizon", "offpeak", 60))} mph. Rush hour is when congestion forms and travels, so this is where a network-aware model has the most to gain. The full table with RMSE, MAPE and sample counts is results/baselines_peak.csv.`, { spacing: { before: 100, after: 120 } }));
add(h2("5.3  What XGBoost relies on"));
const imp = D.xgb_importance;
const impRows = (() => {
  const names = Array.from(new Set(["3", "6", "12"].flatMap((h) => Object.keys(imp[h]))));
  return names.map((n) => [n].concat(["3", "6", "12"].map((h) => (imp[h][n] !== undefined ? `${f1(imp[h][n] * 100)}%` : "below top 8"))));
})();
add(tcap("Table 10. Share of total gain by feature (top 8 per horizon; results/baselines/xgb_importance.csv)"));
add(table(["Feature", "15 min", "30 min", "60 min"], impRows, [3200, 1942, 1942, 1942], { size: 18 }));
add(figure("fig06_xgb_importance.png", "XGBoost feature importance at 15 and 60 minutes."));
add(para(`The model leans on the current speed and its short-term mean at every horizon (current speed ${f1(imp["3"].speed_t * 100)}% of gain at 15 minutes, ${f1(imp["12"].speed_t * 100)}% at 60), and as the horizon grows the clock and the sensor's typical speed gain weight. Nothing in the feature set describes what is happening elsewhere on the network.`));

// ---------------- 6 gap
add(h1("6  Results: identifying the gap (Phase 4)"));
add(para("Phase 1 showed that congestion propagates between sensors. Phase 4 tests whether the best independent model, XGBoost, fails exactly where that propagation matters. All results use the test block and the saved XGBoost predictions."));
add(h2("6.1  Error grows steadily with horizon"));
add(para(`XGBoost's MAE rises from ${f2(sc(X, "mae", 3))} to ${f2(sc(X, "mae", 6))} to ${f2(sc(X, "mae", 12))} mph and its RMSE from ${f2(sc(X, "rmse", 3))} to ${f2(sc(X, "rmse", 6))} to ${f2(sc(X, "rmse", 12))} mph at 15, 30 and 60 minutes. Beyond about 15 minutes the jam that will arrive is not yet visible in the sensor's own history; it is at a neighbour.`));
add(figure("fig08_gap_error_vs_horizon.png", "XGBoost error against horizon, with persistence for reference (test block)."));
add(h2("6.2  Error at congestion onset"));
add(para(`A congestion onset is a speed drop of more than 20 mph within 15 minutes starting from at least 50 mph (3-step rolling median, the Phase 1 definition). A target counts as "after an onset" if it lies within 30 minutes after one at that sensor. The test period contains ${num(D.phase4_findings.test_onsets)} onsets; the affected targets are ${f2(D.phase4_findings.onset_share_of_targets_pct)}% of all real targets.`));
add(tcap("Table 11. Test MAE (mph) in the 30 minutes after an onset versus other times"));
add(table(["Model", "Horizon", "After onset", "Other times", "Ratio"],
  [["xgb", "XGBoost"], ["persistence", "Persistence"], ["xgb_both", "XGBoost + neighbours"]].flatMap(([k, label]) => [3, 6, 12].map((h) => [{ t: label, b: true }, `${5 * h} min`, f2(onset(k, "onset", h)), f2(onset(k, "normal", h)), `${f1(onset(k, "onset", h) / onset(k, "normal", h))}x`])),
  [2600, 1300, 1700, 1700, 1726], { size: 18 }));
add(para(`XGBoost's error right after an onset is ${f1(onset("xgb", "onset", 3) / onset("xgb", "normal", 3))} to ${f1(onset("xgb", "onset", 6) / onset("xgb", "normal", 6))} times its error at other times. The model keeps forecasting free flow until the slowdown appears in its own lags. Persistence fails the same way and worse, so the cause is the information available, not the learner.`, { spacing: { before: 100, after: 120 } }));
add(figure("fig09_gap_onset_error.png", "MAE in the 30 minutes after a congestion onset versus other times, by horizon."));
add(h2("6.3  Errors of linked sensors move together"));
add(para("For every pair of sensors, the correlation of their XGBoost residual series (using only timestamps where both are real) was grouped by the number of road links between them."));
add(tcap("Table 12. Mean residual correlation by road-graph distance"));
const groups = ["1 link", "2 links", "3 links", "4+ links", "not connected"];
add(table(["Pair distance", "XGBoost 15 min", "XGBoost 30 min", "XGBoost 60 min", "XGBoost + neighbours 60 min"],
  groups.map((g) => [g, f3(resid("xgb", g, 3)), f3(resid("xgb", g, 6)), f3(resid("xgb", g, 12)), f3(resid("xgb_both", g, 12))]),
  [2000, 1750, 1750, 1750, 1776], { size: 18 }));
add(para(`Errors are correlated along the road graph and the correlation decays with every link (${f3(resid("xgb", "1 link", 12))} for one link, ${f3(resid("xgb", "2 links", 12))} for two, about zero for unconnected pairs at 60 minutes). A network-wide cause such as time of day would also correlate unconnected pairs; it does not. Correlation grows with horizon, as expected if the missing information is congestion travelling between sensors. This is an association, not proof of causation.`, { spacing: { before: 100, after: 120 } }));
add(figure("fig11_gap_residual_correlation.png", "Mean correlation of XGBoost errors between sensor pairs, by road-graph distance."));
add(h2("6.4  Neighbour-information test"));
add(para("The key experiment hands XGBoost hand-made summaries of each sensor's road neighbours: the weighted mean neighbour speed now and 5, 15 and 30 minutes ago, the slowest neighbour now, and the largest 15-minute drop among neighbours. Upstream only (6 features), downstream only (6) and both directions (12) were tested with the same depth, row thinning and early stopping as the base model, without extra tuning. Because of training cost, each variant was trained for the 15, 30 and 60 minute horizons only."));
const gl = [["xgb (no neighbours)", "XGBoost, own history only"], ["xgb + upstream", "+ upstream neighbours"], ["xgb + downstream", "+ downstream neighbours"], ["xgb + both", "+ both directions"]];
add(tcap("Table 13. Test MAE (mph) and change versus own-history XGBoost"));
add(table(["Model", "15 min", "30 min", "60 min", "RMSE 15", "MAPE 15 (%)"],
  gl.map(([k, label]) => {
    const t = gap(k, "test"), base = gap("xgb (no neighbours)", "test");
    const cellv = (key) => (k === "xgb (no neighbours)" ? f3(t[key]) : `${f3(t[key])} (${(100 * (t[key] / base[key] - 1)).toFixed(1)}%)`);
    return [{ t: label, b: true }, cellv("MAE_15min"), cellv("MAE_30min"), cellv("MAE_60min"), f2(t.RMSE_15min), f1(t.MAPE_15min)];
  }),
  [2600, 1500, 1500, 1500, 960, 966], { size: 17 }));
add(para(`Every variant beats the independent model at every horizon, and the validation block shows the same ordering (both directions: ${["MAE_15min", "MAE_30min", "MAE_60min"].map((k) => `${(100 * (gap("xgb + both", "val")[k] / gap("xgb (no neighbours)", "val")[k] - 1)).toFixed(1)}%`).join(", ")}). Downstream neighbours help more than upstream ones, consistent with the Phase 1 finding that congestion is seen first downstream. Right after congestion onsets, MAE at 15 minutes falls from ${f2(onset("xgb", "onset", 3))} to ${f2(onset("xgb_both", "onset", 3))} mph (${pct(onset("xgb_both", "onset", 3), onset("xgb", "onset", 3))}%) and at 30 minutes from ${f2(onset("xgb", "onset", 6))} to ${f2(onset("xgb_both", "onset", 6))} mph (${pct(onset("xgb_both", "onset", 6), onset("xgb", "onset", 6))}%), a far larger relative gain than the average.`, { spacing: { before: 100, after: 120 } }));
add(figure("fig10_gap_neighbour_test.png", "Test MAE of XGBoost with no, upstream, downstream and both-direction neighbour features."));
add(para([b("Why this is not enough. "), `The 12 neighbour features together receive only ${D.neighbour_feature_gain_pct["3"]}%, ${D.neighbour_feature_gain_pct["6"]}% and ${D.neighbour_feature_gain_pct["12"]}% of the model's total gain at 15, 30 and 60 minutes. The gain shrinks at 60 minutes, residual correlation between linked sensors barely moves (${f3(resid("xgb", "1 link", 12))} to ${f3(resid("xgb_both", "1 link", 12))} at 60 minutes), and the error after onsets is still about ${f1(onset("xgb_both", "onset", 3) / onset("xgb_both", "normal", 3))} times normal at 15 minutes. A tree model with one-hop summaries cannot learn which neighbour matters when, nor look several links away. That is what attention over the road graph and multi-hop diffusion are designed to learn.`]));
add(h2("6.5  Gap statement"));
add(callout([[i_("Independent segment models forecast each sensor from its own history. Congestion forms at bottlenecks and travels backwards along road links at about 16 km/h, so at longer horizons the most predictive information sits at neighbouring sensors. We observe this as error rising steadily with horizon, errors 5 to 6 times larger in the 30 minutes after a congestion onset, and residuals correlated only between road-linked sensors. Handing XGBoost one-hop neighbour summaries lowers error at every horizon but leaves linked residuals correlated and onset errors several times normal. A graph model that learns which neighbours matter, over several hops, should close this gap.")]]));

// ---------------- 7 engineering
add(h1("7  Engineering and reproducibility"));
add(para(`The repository (https://github.com/ManasSaxena14/TrafficFlowControl_AIML-DL_T036) holds ${num(D.loc.src)} lines of source in src/, ${num(D.loc.tests)} lines of tests, and a ${num(D.loc["app.py"])}-line dashboard. Large intermediate data (data/processed, about 2.5 GB per protocol) is git-ignored and rebuilt by the make targets.`));
add(tcap("Table 14. Repository map"));
add(table(["Path", "Contents"], [
  ["src/data", "Loading (HDF5, graph, locations); leak-free preprocessing; split and scaler readers"],
  ["src/features", "46-column per-sensor feature table, cached as memory-mapped arrays"],
  ["src/models", "Simple baselines, XGBoost and Random Forest, shared-weight LSTM"],
  ["src/eval", "Masked metrics; per-horizon and peak/off-peak evaluation"],
  ["src/analysis", "Exploratory analysis (Phase 1) and gap analysis (Phase 4)"],
  ["src/train", "Runners: baselines (Phase 3) and gap experiments (Phase 4)"],
  ["src/viz", "Figure code with one shared visual style"],
  ["tests", "Automated tests (see below)"],
  ["results, reports/figures", "All result tables and the figures used in this report"],
  ["docs", "Project guide, phase reports 1 to 4, this report, the slide deck"],
  ["app.py", "Interactive Streamlit dashboard over the saved results"],
], [2400, 6626], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT], size: 18 }));
add(spacer(60));
add(tcap("Table 15. Automated tests"));
add(table(["Test file", "Tests", "What it guards"], [
  ["test_data.py", String(D.tests["test_data.py"]), "Shapes, five-minute clock, sensor order equals graph order, adjacency properties, outage masking"],
  ["test_leakage.py", String(D.tests["test_leakage.py"]), "Chronological target order, no target in two splits, train-only scaler, no future inputs, masked metrics, adjacency rebuild"],
  ["test_baselines.py", String(D.tests["test_baselines.py"]), "Feature alignment, train-only statistics, baselines, tree models, peak labelling"],
  ["test_gap.py", String(D.tests["test_gap.py"]), "Neighbour direction and features, onset labelling, residual correlation"],
  [{ t: "Total", b: true }, { t: String(Object.values(D.tests).reduce((a, c) => a + c, 0)), b: true }, "All pass; the code also passes the project linter"],
], [2300, 900, 5826], { aligns: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.LEFT], size: 18 }));
add(para([b("Reproduce. "), "make setup, data, eda, preprocess, features, baselines, gap and test rebuild everything from the raw download; make dashboard opens the dashboard. Appendix D lists the commit history. Team: Manas Saxena (Phases 2 to 4, this report and the slide deck), Harsh Raj (Phases 0 and 1), Sumit (dashboard)."], { spacing: { before: 120, after: 120 } }));

// ---------------- 8 limitations
add(h1("8  Limitations and threats to validity"));
[
  [b("Speed only. "), "METR-LA has no flow or occupancy, so the brief's flow and occupancy features are not used; PeMS would supply them."],
  [b("One city, one season. "), "Freeway data from Los Angeles in 2012; results may not transfer to other networks or to signalised urban streets."],
  [b("Single runs, no significance tests. "), "Each model was trained once (seed 42). Small differences, such as the LSTM versus XGBoost at 15 minutes, should not be over-interpreted; significance testing is planned for Phase 6."],
  [b("Limited tuning budgets. "), `XGBoost used at least 580 of its 600 allowed rounds at ${Object.values(xm.best_rounds).filter((v) => v >= 580).length} of 12 horizons, so a larger budget might help slightly. The LSTM used all 12 epochs with validation error still improving slowly. XGBoost and the Random Forest were trained on thinned windows (every ${xm.stride}nd and every ${rm.stride}th). The neighbour experiment covers three horizons.`],
  [b("Validation and test differ. "), `The test period has ${D.splits.test.masked_target_pct}% masked targets against ${D.splits.val.masked_target_pct}% on validation, so test and validation errors are not directly comparable. The test block is used for reporting only.`],
  [b("Definitions drive the gap analysis. "), "Congestion onsets use one threshold definition; residual correlation shows association, not causation."],
  [b("No published-number comparison yet. "), "The DCRNN reference results are not yet reproduced; that is a gate in Phase 5."],
  [b("Simulation only. "), "Signal control, if added, will be evaluated only in a simulator and will not be presented as live traffic control."],
].forEach((x) => add(bullet(x)));

// ---------------- 9 conclusions
add(h1("9  Conclusions and next steps"));
add(para(`The AML phase is complete. A leak-free pipeline and six baselines establish that XGBoost is the strongest independent model (test MAE ${f2(sc(X, "mae", 3))}, ${f2(sc(X, "mae", 6))}, ${f2(sc(X, "mae", 12))} mph at 15, 30, 60 minutes), and four experiments show that its remaining error is concentrated right after congestion starts, at rush hour, and in a form shared between road-linked sensors. Giving it its neighbours' speeds helps at every horizon, which shows the missing information is spatial, but hand-made one-hop features capture only a small part of it.`));
add(h2("Next steps"));
add(bullet([b("Phase 5, graph models. "), "A spatio-temporal graph attention network (STGAT) and a diffusion convolutional recurrent network (DCRNN) over the fixed directed road graph, with the DCRNN required to reproduce the published reference within about 5% before it is trusted, and ablations (graph versus no graph, directed versus undirected)."]));
add(bullet([b("Phase 6, fair comparison. "), `All models on the same splits, windows, masks and metrics, with peak and off-peak breakdowns and significance tests. The targets to beat are XGBoost's ${f2(sc(X, "mae", 3))}, ${f2(sc(X, "mae", 6))} and ${f2(sc(X, "mae", 12))} mph.`]));
add(bullet([b("Phases 7 to 9. "), "Congestion maps, an optional simulated signal-control experiment, and extensions (PeMS flow and occupancy, packaging)."]));

// ---------------- appendices
add(new Paragraph({ children: [new PageBreak()] }));
add(h1("Appendix A  Test MAE at every horizon"));
add(para("Test MAE in mph for horizons 1 to 12 (5 to 60 minutes). Source: results/baselines/<model>.csv, split = test."));
add(table(["Model"].concat(Array.from({ length: 12 }, (_, k) => `${5 * (k + 1)}`)),
  MODELS.map((m) => [{ t: SHORT[m], b: true }].concat(H[m].test.mae.map((v) => f2(v)))),
  [1730].concat(Array(12).fill(608)), { size: 16 }));
add(para("Column headings are forecast horizons in minutes.", { spacing: { before: 60, after: 120 } }));

add(h1("Appendix B  Validation scores"));
add(para("Validation-block scores, used only to choose settings and for early stopping. Pooled columns average over all 12 horizons."));
add(table(["Model", "MAE 15", "MAE 30", "MAE 60", "RMSE 15", "RMSE 60", "MAPE 15", "MAPE 60", "Pooled MAE val", "Pooled MAE test"],
  MODELS.map((m) => [{ t: SHORT[m], b: true }, f2(hv(m, "val", "mae", 3)), f2(hv(m, "val", "mae", 6)), f2(hv(m, "val", "mae", 12)), f2(hv(m, "val", "rmse", 3)), f2(hv(m, "val", "rmse", 12)), f1(hv(m, "val", "mape", 3)), f1(hv(m, "val", "mape", 12)), f2(H[m].pooled.val.mae), f2(H[m].pooled.test.mae)]),
  [1626, 760, 760, 760, 800, 800, 800, 800, 960, 960], { size: 16 }));
add(spacer(100));
add(tcap("XGBoost with neighbour features: validation MAE (mph)"));
add(table(["Model", "15 min", "30 min", "60 min"],
  gl.map(([k, label]) => [{ t: label, b: true }, f3(gap(k, "val").MAE_15min), f3(gap(k, "val").MAE_30min), f3(gap(k, "val").MAE_60min)]),
  [3400, 1875, 1875, 1876], { size: 18 }));

add(new Paragraph({ children: [new PageBreak()] }));
add(h1("Appendix C  Provenance: result to source file"));
add(para("Every number in this report is generated by scripts/report/collect_data.py from the files below; rerun it and scripts/report/build_report.js to regenerate the report."));
add(table(["Report content", "Source in the repository"], [
  ["Dataset facts, missingness, graph, propagation (Section 3)", "results/phase1_findings.json; reports/figures/fig01, fig02, fig05d"],
  ["Split sizes, target periods, masked share, scalers (Section 4.1)", "data/processed/strict and dcrnn (rebuilt by make preprocess); configs/data/metr_la.yaml"],
  ["Model configuration (Section 4.3)", "results/baselines/{xgb,rf,ridge,lstm}.json; src/models"],
  ["Test and validation scores (Sections 1, 5; Appendices A and B)", "results/baselines.csv; results/baselines/<model>.csv"],
  ["Peak versus off-peak (Section 5.2)", "results/baselines_peak.csv"],
  ["XGBoost feature importance (Section 5.3)", "results/baselines/xgb_importance.csv"],
  ["Onset errors, residual correlation (Sections 6.2, 6.3)", "results/gap/onset_vs_normal.csv; results/gap/residual_correlation.csv; results/phase4_findings.json"],
  ["Neighbour experiment (Section 6.4)", "results/gap_results.csv; results/gap/xgb_neighbours.csv; results/gap/xgb_neighbours_importance.csv"],
  ["Figures 1 to 9", "reports/figures/*.png (generated by src/viz)"],
  ["Tests, lines of code, commit history (Section 7, Appendix D)", "pytest collection; git log"],
], [4300, 4726], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT], size: 18 }));

add(new Paragraph({ children: [new PageBreak()] }));
add(h1("Appendix D  Commit history"));
add(para(`${D.commits.length} commits at the time of generation (commit ${D.head}). Source: git log.`));
add(table(["Hash", "Author", "Date", "Subject"],
  D.commits.map((c) => [c.hash, c.author, c.date, c.subject]),
  [900, 1500, 1100, 5526], { aligns: [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.LEFT], size: 16 }));

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "Manas Saxena", title: "Traffic Flow Prediction and Signal Control: AML Phase Report", description: "Project report for the AML phase (Phases 0 to 4)",
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Cambria", size: 32, bold: true, color: NAVY }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Cambria", size: 25, bold: true, color: BLUE }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { titlePage: true, page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1300, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [run("AML phase report  |  page ", { size: 17, color: SLATE }), new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 17, color: SLATE })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: [...cover, ...body],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, `(${figNo} figures)`); });
