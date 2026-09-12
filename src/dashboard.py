"""
Builds a single self-contained HTML file (no server, no external dependencies,
no internet needed) that lets a reviewer browse:
  - headline + per-output metrics vs. naive baselines
  - facility top-k accuracy and confidence-gated nudging tradeoff
  - model-family comparison + rolling-origin cross-validation stability
  - error analysis (accuracy by facility, top confusions)
  - the full predicted-vs-actual table, filterable and sortable

All data is embedded inline as JSON at build time (fetching sibling files
over file:// is blocked by browser CORS rules, so this avoids that entirely
-- double-clicking the file just works).

Run: python -m src.dashboard
Output: outputs/dashboard.html
"""
import json
import pandas as pd

METRICS_PATH = "outputs/metrics.json"
COMPARISON_PATH = "outputs/model_comparison.json"
REVIEW_PATH = "outputs/predictions_review.csv"
OUT_PATH = "outputs/dashboard.html"


def main():
    metrics = json.load(open(METRICS_PATH))
    try:
        comparison = json.load(open(COMPARISON_PATH))
    except FileNotFoundError:
        comparison = None
    review = pd.read_csv(REVIEW_PATH)
    review_records = review.to_dict(orient="records")

    payload = {"metrics": metrics, "comparison": comparison, "review": review_records}
    data_json = json.dumps(payload, allow_nan=False)

    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    with open(OUT_PATH, "w") as fh:
        fh.write(html)
    print(f"Wrote {OUT_PATH} ({len(html)/1024:.0f} KB, {len(review_records)} prediction rows embedded)")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Facility Usage Prediction &mdash; Review Dashboard</title>
<style>
  :root {
    --bg: #f7f8fa; --panel: #ffffff; --text: #1a1f27; --muted: #6b7280;
    --border: #e5e7eb; --accent: #3b6fa0; --accent-soft: #eaf1f8;
    --good: #1a7f37; --good-bg: #dcfce7; --bad: #b42318; --bad-bg: #fee2e2; --warn-bg: #fef3c7;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14171c; --panel: #1c2028; --text: #e7e9ec; --muted: #9aa2af;
      --border: #2b313b; --accent: #6fa4d6; --accent-soft: #1e2c3a;
      --good: #4ade80; --good-bg: #113322; --bad: #f87171; --bad-bg: #3a1717; --warn-bg: #3a2f10;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .subtitle { color: var(--muted); margin: 0 0 26px; font-size: 13px; }
  h2 { font-size: 15px; margin: 36px 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .kpi .label { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }
  .kpi .value { font-size: 24px; font-weight: 650; margin-top: 4px; }
  .kpi .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
  table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--panel); }
  th:hover { color: var(--accent); }
  tbody tr:hover { background: var(--accent-soft); }
  .badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
  .badge.good { background: var(--good-bg); color: var(--good); }
  .badge.bad { background: var(--bad-bg); color: var(--bad); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 12.5px; }
  .bar-label { width: 150px; flex-shrink: 0; color: var(--muted); text-align: right; }
  .bar-track { flex: 1; background: var(--border); border-radius: 4px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--accent); border-radius: 4px; }
  .bar-fill.muted { background: var(--muted); opacity: .5; }
  .bar-value { width: 50px; font-variant-numeric: tabular-nums; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
  .controls input, .controls select { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--panel); color: var(--text); font-size: 13px; }
  .controls input { flex: 1; min-width: 160px; }
  .table-scroll { overflow-x: auto; max-height: 560px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
  .pager { display: flex; gap: 8px; align-items: center; margin-top: 10px; font-size: 12.5px; color: var(--muted); }
  .pager button { padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--panel); color: var(--text); cursor: pointer; }
  .pager button:disabled { opacity: .4; cursor: default; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
  code { background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Facility Usage Prediction &mdash; Review Dashboard</h1>
  <p class="subtitle">Predicted vs. actual on the held-out chronological test set. Generated from <code>outputs/metrics.json</code>, <code>outputs/model_comparison.json</code>, and <code>outputs/predictions_review.csv</code> &mdash; fully static, no server required.</p>

  <div class="kpi-grid" id="kpi-grid"></div>

  <h2>Per-output accuracy vs. naive baselines</h2>
  <div class="panel" id="baseline-bars"></div>

  <div class="two-col">
    <div>
      <h2>Facility: top-k accuracy</h2>
      <div class="panel" id="topk-bars"></div>
    </div>
    <div>
      <h2>Confidence-gated nudging</h2>
      <div class="panel" id="confidence-bars"></div>
      <p class="note">If the system only nudges residents when the model is confident, accuracy on those nudges rises &mdash; at the cost of covering fewer residents.</p>
    </div>
  </div>

  <div class="two-col">
    <div>
      <h2>Facility accuracy by facility</h2>
      <div class="panel" id="facility-acc-bars"></div>
    </div>
    <div>
      <h2>Top confusions</h2>
      <div class="panel"><table id="confusion-table"><tbody></tbody></table></div>
    </div>
  </div>

  <h2 id="model-comparison-heading" style="display:none">Model comparison &amp; stability</h2>
  <div class="panel" id="model-comparison" style="display:none"></div>

  <h2>Predicted vs. actual (every held-out booking)</h2>
  <div class="panel">
    <div class="controls">
      <input id="search" type="text" placeholder="Search resident or record ref&hellip;">
      <select id="facility-filter"><option value="">All facilities</option></select>
      <select id="match-filter">
        <option value="">Any match status</option>
        <option value="4">All 4 correct</option>
        <option value="partial">Partially correct (1-3)</option>
        <option value="0">0 correct</option>
      </select>
    </div>
    <div class="table-scroll"><table id="review-table"><thead></thead><tbody></tbody></table></div>
    <div class="pager">
      <button id="prev-page">&larr; Prev</button>
      <span id="page-info"></span>
      <button id="next-page">Next &rarr;</button>
      <span id="row-count" style="margin-left:auto"></span>
    </div>
  </div>
</div>

<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const M = DATA.metrics, CMP = DATA.comparison, REVIEW = DATA.review;
const pct = (x, d=1) => x === null || x === undefined ? '&mdash;' : (x * 100).toFixed(d) + '%';
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };

// ---------------- KPI cards ----------------
function renderKpis() {
  const grid = document.getElementById('kpi-grid');
  const cards = [
    ['Facility accuracy', pct(M.per_output.facility.accuracy), `top-2 ${pct(M.per_output.facility.top2_accuracy)} &middot; top-3 ${pct(M.per_output.facility.top3_accuracy)}`],
    ['Usage day accuracy', pct(M.per_output.usage_day_of_week.exact_accuracy), `within &plusmn;1 day: ${pct(M.per_output.usage_day_of_week.within_1_day_accuracy)}`],
    ['Usage hour accuracy', pct(M.per_output.usage_hour.exact_accuracy), `MAE ${M.per_output.usage_hour.mae_hours.toFixed(1)}h`],
    ['Notification match rate', pct(M.per_output.notification_time_leadtime.match_within_tolerance_rate), `median error ${M.per_output.notification_time_leadtime.median_absolute_error_hours.toFixed(1)}h`],
    ['Avg. outputs correct', M.overall.mean_outputs_correct_out_of_4.toFixed(2) + ' / 4', `${pct(M.overall.all_4_correct_rate)} rows fully correct`],
    ['Held-out bookings', M.test_rows.toLocaleString(), `chronological holdout`],
  ];
  cards.forEach(([label, value, sub]) => {
    const k = el('div', 'kpi');
    k.appendChild(el('div', 'label', label));
    k.appendChild(el('div', 'value', value));
    k.appendChild(el('div', 'sub', sub));
    grid.appendChild(k);
  });
}

function barRow(container, label, value, opts) {
  opts = opts || {};
  const row = el('div', 'bar-row');
  row.appendChild(el('div', 'bar-label', label));
  const track = el('div', 'bar-track');
  const fill = el('div', 'bar-fill' + (opts.muted ? ' muted' : ''));
  fill.style.width = Math.max(2, Math.min(100, value * 100)) + '%';
  track.appendChild(fill);
  row.appendChild(track);
  row.appendChild(el('div', 'bar-value', pct(value)));
  container.appendChild(row);
}

function renderBaselines() {
  const c = document.getElementById('baseline-bars');
  const rows = [
    ['Facility (model)', M.per_output.facility.accuracy, false],
    ['Facility (global mode baseline)', M.baselines.always_predict_global_mode.facility_accuracy, true],
    ['Usage day (model)', M.per_output.usage_day_of_week.exact_accuracy, false],
    ['Usage day (global mode baseline)', M.baselines.always_predict_global_mode.dow_accuracy, true],
    ['Usage hour (model)', M.per_output.usage_hour.exact_accuracy, false],
    ['Usage hour (global mode baseline)', M.baselines.always_predict_global_mode.hour_accuracy, true],
  ];
  rows.forEach(([label, value, muted]) => barRow(c, label, value, {muted}));
}

function renderTopk() {
  const c = document.getElementById('topk-bars');
  ['top1_accuracy', 'top2_accuracy', 'top3_accuracy'].forEach(k => {
    barRow(c, k.replace('_accuracy', '').toUpperCase(), M.per_output.facility[k]);
  });
}

function renderConfidence() {
  const c = document.getElementById('confidence-bars');
  const rows = M.error_analysis.confidence_gated_nudging;
  rows.forEach(r => {
    if (r.facility_accuracy_when_nudged === null) return;
    barRow(c, `conf &ge; ${r.min_confidence} (${pct(r.coverage_rate, 0)} of residents)`, r.facility_accuracy_when_nudged);
  });
}

function renderFacilityAcc() {
  const c = document.getElementById('facility-acc-bars');
  const entries = Object.entries(M.error_analysis.facility_accuracy_by_true_facility).sort((a, b) => b[1] - a[1]);
  entries.forEach(([f, v]) => barRow(c, f, v));
}

function renderConfusions() {
  const tbody = document.querySelector('#confusion-table tbody');
  Object.entries(M.error_analysis.top_facility_confusions).forEach(([k, v]) => {
    const tr = el('tr');
    tr.appendChild(el('td', null, k));
    tr.appendChild(el('td', null, v + ' rows'));
    tbody.appendChild(tr);
  });
}

function renderModelComparison() {
  if (!CMP) return;
  document.getElementById('model-comparison-heading').style.display = '';
  const c = document.getElementById('model-comparison');
  c.style.display = '';
  const mc = CMP.model_family_comparison, cv = CMP.rolling_origin_cv;
  c.appendChild(el('p', null, `<strong>Model family (facility prediction, same split):</strong> Random Forest ${pct(mc.random_forest.accuracy)} accuracy (macro-F1 ${mc.random_forest.macro_f1.toFixed(2)}) vs. Hist Gradient Boosting ${pct(mc.hist_gradient_boosting.accuracy)} (macro-F1 ${mc.hist_gradient_boosting.macro_f1.toFixed(2)}).`));
  c.appendChild(el('p', null, `<strong>Rolling-origin cross-validation</strong> (${cv.folds.length} sequential time windows): facility accuracy ${(cv.facility_accuracy_mean*100).toFixed(1)}% &plusmn; ${(cv.facility_accuracy_std*100).toFixed(1)}% &mdash; consistent across time, not a lucky single split.`));
  const table = el('table');
  const thead = el('thead', null, '<tr><th>Fold</th><th>Test window</th><th>Train rows</th><th>Test rows</th><th>Accuracy</th></tr>');
  table.appendChild(thead);
  const tbody = el('tbody');
  cv.folds.forEach(f => {
    const tr = el('tr');
    [f.fold, `${f.test_window_start.slice(0,10)} &rarr; ${f.test_window_end.slice(0,10)}`, f.train_rows, f.test_rows, pct(f.accuracy)].forEach(v => tr.appendChild(el('td', null, String(v))));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  c.appendChild(table);
}

// ---------------- predictions table ----------------
const COLUMNS = ['record_ref','resident_id','pred_facility','pred_facility_confidence','pred_usage_day','pred_usage_time','pred_nudge_day','pred_nudge_time','actual_facility','actual_usage_day','actual_usage_time','actual_booked_day','actual_booked_time','matches_out_of_4'];
let state = { sortCol: 'record_ref', sortDir: 1, page: 0, pageSize: 50 };

function matchBadge(n) {
  const cls = n === 4 ? 'good' : (n === 0 ? 'bad' : '');
  return cls ? `<span class="badge ${cls}">${n} / 4</span>` : `${n} / 4`;
}

function filteredRows() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const facility = document.getElementById('facility-filter').value;
  const matchF = document.getElementById('match-filter').value;
  return REVIEW.filter(r => {
    if (q && !(r.record_ref.toLowerCase().includes(q) || r.resident_id.toLowerCase().includes(q))) return false;
    if (facility && r.pred_facility !== facility && r.actual_facility !== facility) return false;
    if (matchF === '4' && r.matches_out_of_4 !== 4) return false;
    if (matchF === '0' && r.matches_out_of_4 !== 0) return false;
    if (matchF === 'partial' && (r.matches_out_of_4 === 0 || r.matches_out_of_4 === 4)) return false;
    return true;
  });
}

function renderTable() {
  let rows = filteredRows();
  rows.sort((a, b) => {
    const av = a[state.sortCol], bv = b[state.sortCol];
    if (typeof av === 'number') return (av - bv) * state.sortDir;
    return String(av).localeCompare(String(bv)) * state.sortDir;
  });
  const totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, totalPages - 1);
  const pageRows = rows.slice(state.page * state.pageSize, (state.page + 1) * state.pageSize);

  const thead = document.querySelector('#review-table thead');
  thead.innerHTML = '';
  const trh = el('tr');
  COLUMNS.forEach(c => {
    const th = el('th', null, c.replace(/_/g, ' ') + (state.sortCol === c ? (state.sortDir === 1 ? ' ↑' : ' ↓') : ''));
    th.onclick = () => { state.sortDir = state.sortCol === c ? -state.sortDir : 1; state.sortCol = c; renderTable(); };
    trh.appendChild(th);
  });
  thead.appendChild(trh);

  const tbody = document.querySelector('#review-table tbody');
  tbody.innerHTML = '';
  pageRows.forEach(r => {
    const tr = el('tr');
    COLUMNS.forEach(c => {
      const val = c === 'matches_out_of_4' ? matchBadge(r[c]) : r[c];
      tr.appendChild(el('td', null, String(val)));
    });
    tbody.appendChild(tr);
  });

  document.getElementById('page-info').textContent = `Page ${state.page + 1} of ${totalPages}`;
  document.getElementById('row-count').textContent = `${rows.length.toLocaleString()} rows`;
  document.getElementById('prev-page').disabled = state.page === 0;
  document.getElementById('next-page').disabled = state.page >= totalPages - 1;
}

function initControls() {
  const facilities = [...new Set(REVIEW.map(r => r.actual_facility))].sort();
  const sel = document.getElementById('facility-filter');
  facilities.forEach(f => sel.appendChild(el('option', null, f)).value = f);
  document.getElementById('search').addEventListener('input', () => { state.page = 0; renderTable(); });
  sel.addEventListener('change', () => { state.page = 0; renderTable(); });
  document.getElementById('match-filter').addEventListener('change', () => { state.page = 0; renderTable(); });
  document.getElementById('prev-page').addEventListener('click', () => { state.page--; renderTable(); });
  document.getElementById('next-page').addEventListener('click', () => { state.page++; renderTable(); });
}

renderKpis();
renderBaselines();
renderTopk();
renderConfidence();
renderFacilityAcc();
renderConfusions();
renderModelComparison();
initControls();
renderTable();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
