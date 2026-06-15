"""
rule_viz.py
-----------
Generate a self-contained interactive HTML rule-matrix visualization.

The output is a single ``.html`` file with:
  - D3.js loaded from CDN (one external dependency)
  - JSON data embedded inline
  - Dark GitHub-style theme
  - Model tabs, sort controls, precision/support filter sliders
  - Rule matrix: rows = rules, columns = features used by that model
  - Right-side metric bars: precision (RdYlGn), support (blue), importance (orange)
  - Hover tooltip with full rule description and all stats

Functions
---------
rule_matrix_html    Build HTML string for one or more RuleSets.
save_rule_report    Convenience wrapper that writes the HTML to a file.

Examples
--------
>>> from ml_elements.rules import RuleExtractor
>>> from ml_elements.rule_viz import save_rule_report
>>> rs = RuleExtractor().from_model(tree_model, X_tr, y_tr, feature_names, "Tree")
>>> save_rule_report([rs], "outputs/rule_report.html", title="Credit Fraud Rules")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .rules import RuleSet


# ─── Public API ─────────────────────────────────────────────────────────────


def rule_matrix_html(
    rulesets: Sequence[RuleSet],
    title: str = "Rule Explorer",
) -> str:
    """
    Build a self-contained interactive HTML rule matrix.

    Parameters
    ----------
    rulesets : sequence of RuleSet
        One or more rule sets (one tab per model).
    title : str
        Page title shown in the header.

    Returns
    -------
    str
        Full HTML document as a string.
    """
    data_json = json.dumps([rs.to_dict() for rs in rulesets], indent=None)
    return _HTML_TEMPLATE.format(
        title=_esc(title),
        data_json=data_json,
    )


def save_rule_report(
    rulesets: Sequence[RuleSet],
    path: str | Path,
    title: str = "Rule Explorer",
) -> Path:
    """Write the HTML report to *path* and return the resolved Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rule_matrix_html(rulesets, title=title), encoding="utf-8")
    return path


# ─── Helpers ────────────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ─── HTML Template ──────────────────────────────────────────────────────────
#
# {title}     → page title (HTML-escaped)
# {data_json} → JSON array of RuleSet.to_dict()

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
/* ── Reset & base ──────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0d1117;
  color: #c9d1d9;
  font-family: ui-monospace, "SF Mono", Consolas, "Liberation Mono", monospace;
  font-size: 13px;
  line-height: 1.5;
}}
a {{ color: #58a6ff; }}

/* ── Layout ────────────────────────────────────────────────────────── */
#app {{ max-width: 1600px; margin: 0 auto; padding: 24px 20px; }}

/* ── Header ────────────────────────────────────────────────────────── */
header {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 16px;
  margin-bottom: 20px;
  border-bottom: 1px solid #21262d;
  padding-bottom: 14px;
}}
header h1 {{
  font-size: 1.25rem;
  font-weight: 600;
  color: #e6edf3;
  flex: 1 1 auto;
}}

/* ── Tabs ──────────────────────────────────────────────────────────── */
#tabs {{
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}}
.tab-btn {{
  background: #161b22;
  border: 1px solid #30363d;
  color: #8b949e;
  padding: 5px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: all .15s;
}}
.tab-btn:hover {{ border-color: #58a6ff; color: #c9d1d9; }}
.tab-btn.active {{
  background: #1f6feb;
  border-color: #1f6feb;
  color: #fff;
  font-weight: 600;
}}

/* ── Controls ──────────────────────────────────────────────────────── */
#controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: center;
  margin-bottom: 14px;
  padding: 10px 14px;
  background: #161b22;
  border: 1px solid #21262d;
  border-radius: 8px;
}}
.ctrl-group {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.ctrl-group label {{ color: #8b949e; font-size: 11px; white-space: nowrap; }}
.ctrl-group span {{ color: #e6edf3; font-size: 11px; min-width: 36px; }}
input[type=range] {{
  -webkit-appearance: none;
  appearance: none;
  width: 110px;
  height: 4px;
  background: #30363d;
  border-radius: 2px;
  outline: none;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px; height: 14px;
  background: #1f6feb;
  border-radius: 50%;
  cursor: pointer;
}}
select {{
  background: #21262d;
  border: 1px solid #30363d;
  color: #c9d1d9;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}}

/* ── Stats banner ──────────────────────────────────────────────────── */
#stats {{
  font-size: 11px;
  color: #8b949e;
  margin-bottom: 10px;
  height: 18px;
}}

/* ── Table wrapper ─────────────────────────────────────────────────── */
#matrix-wrap {{
  overflow-x: auto;
  border: 1px solid #21262d;
  border-radius: 8px;
}}
table {{
  border-collapse: collapse;
  width: max-content;
  min-width: 100%;
}}

/* ── Table header ──────────────────────────────────────────────────── */
thead th {{
  background: #161b22;
  color: #8b949e;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .04em;
  padding: 8px 12px;
  white-space: nowrap;
  border-bottom: 1px solid #21262d;
  user-select: none;
  cursor: pointer;
  position: sticky;
  top: 0;
  z-index: 2;
}}
thead th:hover {{ color: #e6edf3; }}
thead th.sort-asc::after  {{ content: " ▲"; color: #58a6ff; }}
thead th.sort-desc::after {{ content: " ▼"; color: #58a6ff; }}
thead th.feat-col {{ min-width: 120px; max-width: 200px; }}
thead th.metric-col {{ min-width: 130px; text-align: right; }}
thead th:first-child {{ min-width: 52px; text-align: center; cursor: default; }}

/* ── Table body ────────────────────────────────────────────────────── */
tbody tr {{
  border-bottom: 1px solid #161b22;
  transition: background .1s;
}}
tbody tr:hover {{ background: #161b22; }}
tbody tr.hidden {{ display: none; }}
td {{
  padding: 6px 12px;
  vertical-align: middle;
  white-space: nowrap;
}}
td.rule-id {{
  text-align: center;
  color: #6e7681;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .05em;
}}
td.cond-cell {{
  font-size: 11.5px;
  border-radius: 4px;
  padding: 4px 10px;
}}
td.cond-cell.has-cond {{
  font-weight: 500;
}}
td.cond-cell.no-cond {{
  color: #30363d;
  text-align: center;
}}
td.metric-cell {{ text-align: right; }}

/* ── Mini bars ─────────────────────────────────────────────────────── */
.bar-wrap {{
  display: flex;
  align-items: center;
  gap: 6px;
  justify-content: flex-end;
}}
.bar-track {{
  width: 70px;
  height: 6px;
  background: #21262d;
  border-radius: 3px;
  overflow: hidden;
  flex-shrink: 0;
}}
.bar-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width .25s;
}}
.bar-label {{
  font-size: 11px;
  min-width: 38px;
  color: #e6edf3;
}}

/* ── Tooltip ───────────────────────────────────────────────────────── */
#tooltip {{
  position: fixed;
  pointer-events: none;
  background: #1c2128;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  color: #c9d1d9;
  max-width: 340px;
  box-shadow: 0 8px 24px rgba(0,0,0,.5);
  z-index: 100;
  opacity: 0;
  transition: opacity .12s;
}}
#tooltip.visible {{ opacity: 1; }}
#tooltip .tt-rule {{ font-weight: 600; color: #e6edf3; margin-bottom: 6px; word-break: break-word; white-space: normal; }}
#tooltip .tt-row {{ display: flex; justify-content: space-between; gap: 20px; margin-top: 3px; color: #8b949e; }}
#tooltip .tt-row span {{ color: #c9d1d9; }}

/* ── Empty state ───────────────────────────────────────────────────── */
#empty-msg {{
  display: none;
  text-align: center;
  color: #6e7681;
  padding: 40px;
  font-size: 13px;
}}
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>{title}</h1>
  </header>
  <div id="tabs"></div>
  <div id="controls">
    <div class="ctrl-group">
      <label>Sort by</label>
      <select id="sort-by">
        <option value="importance">Importance</option>
        <option value="precision">Precision</option>
        <option value="support">Support</option>
        <option value="rule_id">Rule #</option>
      </select>
    </div>
    <div class="ctrl-group">
      <label>Min precision</label>
      <input type="range" id="min-prec" min="0" max="1" step="0.01" value="0">
      <span id="min-prec-val">0.00</span>
    </div>
    <div class="ctrl-group">
      <label>Min support</label>
      <input type="range" id="min-sup" min="0" max="1000" step="1" value="0">
      <span id="min-sup-val">0</span>
    </div>
  </div>
  <div id="stats"></div>
  <div id="matrix-wrap">
    <table id="matrix">
      <thead id="matrix-head"></thead>
      <tbody id="matrix-body"></tbody>
    </table>
    <div id="empty-msg">No rules match the current filters.</div>
  </div>
</div>
<div id="tooltip"></div>

<script>
/* ── Data ─────────────────────────────────────────────────────────────── */
const ALL_DATA = {data_json};

/* ── State ────────────────────────────────────────────────────────────── */
let activeIdx = 0;
let sortKey = "importance";
let sortDir = -1;  // -1 = descending
let minPrec = 0;
let minSup = 0;

/* ── D3 color scale for precision ─────────────────────────────────────── */
const precColor = d3.scaleSequential(d3.interpolateRdYlGn).domain([0.2, 0.9]);
const supportColor = "#4a9eff";
const importColor  = "#f0883e";

/* ── Tabs ─────────────────────────────────────────────────────────────── */
function buildTabs() {{
  const container = document.getElementById("tabs");
  container.innerHTML = "";
  ALL_DATA.forEach((rs, i) => {{
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === activeIdx ? " active" : "");
    btn.textContent = `${{rs.model_name}} (${{rs.model_type}})`;
    btn.onclick = () => {{ activeIdx = i; render(); }};
    container.appendChild(btn);
  }});
}}

/* ── Column helpers ───────────────────────────────────────────────────── */
function usedFeatures(rules) {{
  const seen = new Set();
  rules.forEach(r => r.conditions.forEach(c => seen.add(c.feature)));
  return [...seen].sort();
}}

function condMap(rule) {{
  const m = {{}};
  rule.conditions.forEach(c => {{
    const existing = m[c.feature];
    const text = `${{c.op}} ${{Number(c.value).toFixed(4)}}`;
    m[c.feature] = existing ? existing + " & " + text : text;
  }});
  return m;
}}

/* ── Mini bar HTML ────────────────────────────────────────────────────── */
function miniBar(pct, color, label) {{
  return `<div class="bar-wrap">
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    <span class="bar-label">${{label}}</span>
  </div>`;
}}

/* ── Filter & sort rules ──────────────────────────────────────────────── */
function applyFilters(rules) {{
  return rules.filter(r => r.precision >= minPrec && r.support >= minSup);
}}

function sortRules(rules) {{
  return [...rules].sort((a, b) => sortDir * (a[sortKey] - b[sortKey]));
}}

/* ── Max support (for bar scaling) ───────────────────────────────────── */
function maxSupport(rules) {{
  return d3.max(rules, r => r.support) || 1;
}}

/* ── Build header ─────────────────────────────────────────────────────── */
function buildHeader(features) {{
  const head = document.getElementById("matrix-head");
  head.innerHTML = "";
  const tr = document.createElement("tr");

  const thId = document.createElement("th");
  thId.textContent = "#";
  thId.onclick = () => setSort("rule_id", tr.querySelectorAll("th"));
  tr.appendChild(thId);

  features.forEach(f => {{
    const th = document.createElement("th");
    th.textContent = f;
    th.className = "feat-col";
    th.title = `Feature: ${{f}}`;
    tr.appendChild(th);
  }});

  [["precision","Precision"], ["support","Support"], ["importance","Importance"]].forEach(([k,lbl]) => {{
    const th = document.createElement("th");
    th.textContent = lbl;
    th.className = "metric-col";
    th.dataset.sort = k;
    th.onclick = () => setSort(k, tr.querySelectorAll("th"));
    tr.appendChild(th);
  }});

  head.appendChild(tr);
  updateSortIndicators(tr.querySelectorAll("th"));
}}

function setSort(key, allTh) {{
  if (sortKey === key) {{ sortDir *= -1; }}
  else {{ sortKey = key; sortDir = -1; }}
  updateSortIndicators(allTh);
  render();
}}

function updateSortIndicators(allTh) {{
  allTh.forEach(th => {{
    th.classList.remove("sort-asc","sort-desc");
    if (th.dataset.sort === sortKey) {{
      th.classList.add(sortDir === -1 ? "sort-desc" : "sort-asc");
    }}
  }});
}}

/* ── Build body ───────────────────────────────────────────────────────── */
function buildBody(rules, features, maxSup) {{
  const body = document.getElementById("matrix-body");
  body.innerHTML = "";
  const emptyMsg = document.getElementById("empty-msg");

  if (rules.length === 0) {{
    emptyMsg.style.display = "block";
    return;
  }}
  emptyMsg.style.display = "none";

  rules.forEach(rule => {{
    const cm = condMap(rule);
    const tr = document.createElement("tr");

    // Rule ID
    const tdId = document.createElement("td");
    tdId.className = "rule-id";
    tdId.textContent = `R${{rule.rule_id}}`;
    tr.appendChild(tdId);

    // Feature condition cells
    features.forEach(f => {{
      const td = document.createElement("td");
      td.className = "cond-cell";
      if (cm[f]) {{
        const c = precColor(rule.precision);
        td.textContent = cm[f];
        td.classList.add("has-cond");
        td.style.background = hexWithAlpha(c, 0.18);
        td.style.color = c;
        td.style.borderLeft = `3px solid ${{c}}`;
      }} else {{
        td.textContent = "—";
        td.classList.add("no-cond");
      }}
      tr.appendChild(td);
    }});

    // Precision
    const tdPrec = document.createElement("td");
    tdPrec.className = "metric-cell";
    tdPrec.innerHTML = miniBar(
      rule.precision * 100,
      precColor(rule.precision),
      (rule.precision * 100).toFixed(1) + "%"
    );
    tr.appendChild(tdPrec);

    // Support
    const tdSup = document.createElement("td");
    tdSup.className = "metric-cell";
    tdSup.innerHTML = miniBar(
      (rule.support / maxSup) * 100,
      supportColor,
      rule.support.toLocaleString()
    );
    tr.appendChild(tdSup);

    // Importance
    const tdImp = document.createElement("td");
    tdImp.className = "metric-cell";
    tdImp.innerHTML = miniBar(
      rule.importance * 100,
      importColor,
      (rule.importance * 100).toFixed(1) + "%"
    );
    tr.appendChild(tdImp);

    // Tooltip
    tr.addEventListener("mouseenter", (e) => showTooltip(e, rule));
    tr.addEventListener("mousemove",  (e) => moveTooltip(e));
    tr.addEventListener("mouseleave", hideTooltip);

    body.appendChild(tr);
  }});
}}

/* ── Tooltip ──────────────────────────────────────────────────────────── */
const tooltip = document.getElementById("tooltip");

function showTooltip(e, rule) {{
  tooltip.innerHTML = `
    <div class="tt-rule">${{rule.description}}</div>
    <div class="tt-row">Precision <span>${{(rule.precision*100).toFixed(1)}}%</span></div>
    <div class="tt-row">Support   <span>${{rule.support.toLocaleString()}} samples</span></div>
    <div class="tt-row">Importance<span>${{(rule.importance*100).toFixed(1)}}%</span></div>
    <div class="tt-row">Prediction<span>${{(rule.prediction*100).toFixed(1)}}% positive</span></div>
  `;
  tooltip.classList.add("visible");
  moveTooltip(e);
}}

function moveTooltip(e) {{
  let x = e.clientX + 14, y = e.clientY - 10;
  if (x + 360 > window.innerWidth) x = e.clientX - 360;
  if (y + 140 > window.innerHeight) y = e.clientY - 140;
  tooltip.style.left = x + "px";
  tooltip.style.top  = y + "px";
}}

function hideTooltip() {{
  tooltip.classList.remove("visible");
}}

/* ── Color utils ──────────────────────────────────────────────────────── */
function hexWithAlpha(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

/* ── Stats banner ─────────────────────────────────────────────────────── */
function updateStats(shown, total, rs) {{
  document.getElementById("stats").textContent =
    `${{shown}} of ${{total}} rules shown  ·  ${{rs.model_type}}  ·  ` +
    `n_train = ${{rs.n_train.toLocaleString()}}  ·  ` +
    `base rate = ${{(rs.positive_rate*100).toFixed(1)}}%`;
}}

/* ── Controls ─────────────────────────────────────────────────────────── */
document.getElementById("sort-by").addEventListener("change", e => {{
  sortKey = e.target.value;
  render();
}});

document.getElementById("min-prec").addEventListener("input", e => {{
  minPrec = parseFloat(e.target.value);
  document.getElementById("min-prec-val").textContent = minPrec.toFixed(2);
  render();
}});

document.getElementById("min-sup").addEventListener("input", e => {{
  minSup = parseInt(e.target.value);
  document.getElementById("min-sup-val").textContent = minSup;
  render();
}});

/* ── Main render ──────────────────────────────────────────────────────── */
function render() {{
  buildTabs();

  const rs = ALL_DATA[activeIdx];
  const allRules = rs.rules;
  const features = usedFeatures(allRules);

  // Update support slider max
  const supMax = d3.max(allRules, r => r.support) || 100;
  const supSlider = document.getElementById("min-sup");
  supSlider.max = supMax;
  if (minSup > supMax) {{ minSup = 0; supSlider.value = 0; document.getElementById("min-sup-val").textContent = 0; }}

  buildHeader(features);

  const filtered = applyFilters(allRules);
  const sorted   = sortRules(filtered);
  const maxSup   = maxSupport(allRules);

  buildBody(sorted, features, maxSup);
  updateStats(sorted.length, allRules.length, rs);
}}

/* ── Init ─────────────────────────────────────────────────────────────── */
render();
</script>
</body>
</html>
"""
