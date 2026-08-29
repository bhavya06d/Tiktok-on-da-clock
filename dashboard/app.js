/* Shared renderer for the live dashboard and the published snapshot.
   Expects a global `RUN_DATA` (snapshot) OR polls /api/run/<name> (live). */

const FMT = (x, d = 4) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const $ = (sel, el = document) => el.querySelector(sel);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

let chart = null;
let openRows = new Set();

const CSS = (name, fb) => {
  try { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fb; }
  catch (e) { return fb; }
};
const THEME = () => ({
  accent: CSS("--accent", "#b26a00"),
  series: CSS("--series", "#3f7d94"),
  muted: CSS("--muted", "#5c6673"),
  bad: CSS("--bad", "#b3341f"),
  line: CSS("--line", "#dde1e7"),
  ink: CSS("--ink", "#1a1f26"),
  band: CSS("--band", "rgba(178,106,0,.09)"),
});

function summarise(data) {
  const its = data.iterations.filter(i => i.metrics);
  const best = its.reduce((b, i) => Math.max(b, i.metrics.primary), -Infinity);
  const bestIt = its.find(i => i.metrics && i.metrics.primary === best);
  const base = (data.meta && data.meta.baseline_primary) || 0.6016;
  const oracle = (data.meta && data.meta.oracle_primary) || 0.8484;
  const s = data.summary || {};
  return {
    best: its.length ? best : null,
    bestIter: bestIt ? bestIt.iter : null,
    bestMethod: bestIt ? bestIt.method : "—",
    delta: its.length ? best - base : 0,
    headroom: its.length ? (best - base) / (oracle - base) : 0,
    base, oracle,
    iters: data.iterations.length,
    failures: data.iterations.filter(i => !i.metrics).length,
    interventions: s.manual_interventions != null ? s.manual_interventions : 0,
    wall: s.wall_clock_seconds,
    tokens: s.llm_tokens ? s.llm_tokens.total : (data.iterations.reduce((a, i) => a + ((i.tokens && (i.tokens.input + i.tokens.output)) || 0), 0)),
    status: data.status,
  };
}

function render(data) {
  if (!data || !data.exists) { $("#app").innerHTML = '<div class="empty">No run log yet. Start <code>python run_agent.py</code>.</div>'; return; }
  const S = summarise(data);

  // ---- header / hero ----
  $("#run-name").textContent = data.name;
  const badge = $("#status-badge");
  badge.textContent = S.status;
  badge.className = "badge " + S.status;

  $("#hero").innerHTML = "";
  const heroCards = [
    ["best val primary", FMT(S.best), `iter ${S.bestIter ?? "—"} · ${S.bestMethod}`, "accent"],
    ["Δ over FM baseline", (S.delta >= 0 ? "+" : "") + FMT(S.delta), `baseline ${FMT(S.base)}`, S.delta > 0 ? "good" : ""],
    ["% of oracle headroom", (S.headroom * 100).toFixed(1) + "%", `ceiling ${FMT(S.oracle)}`, ""],
    ["iterations", String(S.iters), S.status === "converged" ? "converged (ε=0.002, N=3)" : "running", ""],
  ];
  for (const [k, v, sub, cls] of heroCards) {
    const c = el("div", "hero-card " + cls);
    c.append(el("div", "hc-label", k), el("div", "hc-value", v), el("div", "hc-sub", sub));
    $("#hero").append(c);
  }

  // ---- secondary stats ----
  $("#stats").innerHTML = "";
  const stats = [
    ["failures recovered", S.failures, "crashes the loop caught & rolled back"],
    ["manual interventions", S.interventions, "target: 0"],
    ["wall-clock", S.wall != null ? (S.wall / 60).toFixed(1) + " min" : "—", "unattended"],
    ["LLM tokens", S.tokens ? S.tokens.toLocaleString() : "0", "offline planner = 0"],
  ];
  for (const [k, v, sub] of stats) {
    const c = el("div", "stat");
    c.append(el("div", "s-value", String(v)), el("div", "s-label", k), el("div", "s-sub", sub));
    $("#stats").append(c);
  }

  drawChart(data, S);
  drawTimeline(data);
  drawEvents(data);
}

function drawChart(data, S) {
  const labels = data.iterations.map(i => "it " + i.iter);
  const pts = data.iterations.map(i => (i.metrics ? i.metrics.primary : null));
  const runningBest = [];
  let b = -Infinity;
  for (const i of data.iterations) { if (i.metrics) b = Math.max(b, i.metrics.primary); runningBest.push(b === -Infinity ? null : b); }
  const ctx = $("#chart").getContext("2d");
  const T = THEME();
  const mkLine = (y, color, dash) => ({ label: "", data: labels.map(() => y), borderColor: color, borderWidth: 1, borderDash: dash, pointRadius: 0, fill: false });

  const cfg = {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "running best", data: runningBest, borderColor: T.accent, backgroundColor: T.band, borderWidth: 2, fill: true, pointRadius: 0, tension: .15 },
        {
          label: "iteration primary", data: pts, borderColor: T.series, borderWidth: 1.5, tension: .15, spanGaps: false,
          pointRadius: data.iterations.map(i => (i.metrics ? (i.accepted ? 6 : 4) : 5)),
          pointStyle: data.iterations.map(i => (i.metrics ? (i.accepted ? "circle" : "crossRot") : "crossRot")),
          pointBackgroundColor: data.iterations.map(i => (!i.metrics ? T.bad : i.accepted ? T.accent : "transparent")),
          pointBorderColor: data.iterations.map(i => (!i.metrics ? T.bad : T.series)),
        },
        { ...mkLine(S.base, T.muted, [5, 4]), label: "FM baseline" },
        { ...mkLine(S.oracle, T.accent, [2, 3]), label: "oracle ceiling" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { min: Math.min(S.base - 0.02, ...pts.filter(x => x != null)) - 0.005, max: Math.max(S.oracle + 0.01, ...runningBest.filter(x => x != null)), grid: { color: T.line }, ticks: { color: T.muted } },
        x: { grid: { display: false }, ticks: { color: T.muted } },
      },
      plugins: {
        legend: { labels: { color: T.ink, filter: it => it.text, boxWidth: 12, boxHeight: 2 }, position: "bottom" },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const i = data.iterations[items[0].dataIndex];
              if (!i) return "";
              return [`method: ${i.method}`, `decision: ${i.decision || (i.accepted ? "KEEP" : "DISCARD")}`, "", (i.hypothesis || "").slice(0, 220)];
            },
          },
        },
      },
    },
  };
  if (chart) { chart.data = cfg.data; chart.options = cfg.options; chart.update(); }
  else chart = new Chart(ctx, cfg);
}

function pill(txt, cls) { const p = el("span", "pill " + (cls || ""), txt); return p; }

function drawTimeline(data) {
  const box = $("#timeline"); box.innerHTML = "";
  [...data.iterations].reverse().forEach(it => {
    const row = el("div", "iter");
    const head = el("div", "iter-head");
    const state = !it.metrics ? "fail" : it.accepted ? "keep" : "discard";
    head.append(
      el("span", "iter-idx", "#" + it.iter),
      pill(it.method || "?", "method"),
      pill(state === "keep" ? "KEEP" : state === "discard" ? "DISCARD" : "FAILED", state),
    );
    const m = it.metrics;
    head.append(el("span", "iter-score", m ? `primary ${FMT(m.primary)}  ·  gauc ${FMT(m.gauc)}  ·  ndcg@5 ${FMT(m.ndcg5)}` : (it.error || "").split("\n")[0].slice(0, 80)));
    if (it.duration_s) head.append(el("span", "iter-dur", (it.duration_s) + "s"));
    row.append(head);

    const body = el("div", "iter-body");
    body.append(el("div", "h-label", "hypothesis"), el("div", "hypo", it.hypothesis || "—"));
    body.append(el("div", "h-label", "decision"), el("div", "decision", it.decision || "—"));
    if (it.diff) { const d = el("pre", "diff"); d.append(...colorDiff(it.diff)); body.append(el("div", "h-label", "code diff"), d); }
    if (!it.metrics && it.stderr_tail) { const e = el("pre", "trace"); e.textContent = it.stderr_tail.slice(-1600); body.append(el("div", "h-label", "traceback (fed back to the agent)"), e); }
    row.append(body);

    if (openRows.has(it.iter)) row.classList.add("open");
    head.onclick = () => { row.classList.toggle("open"); row.classList.contains("open") ? openRows.add(it.iter) : openRows.delete(it.iter); };
    box.append(row);
  });
}

function colorDiff(text) {
  return text.split("\n").map(l => {
    let c = "d-ctx";
    if (l.startsWith("+") && !l.startsWith("+++")) c = "d-add";
    else if (l.startsWith("-") && !l.startsWith("---")) c = "d-del";
    else if (l.startsWith("@@")) c = "d-hunk";
    return el("span", c, l + "\n");
  });
}

function drawEvents(data) {
  const box = $("#events"); box.innerHTML = "";
  if (!data.events.length) { box.append(el("div", "muted", "no recovery events")); return; }
  data.events.forEach(e => {
    const r = el("div", "evt");
    r.append(pill("iter " + e.iter, "method"), el("span", "evt-name", e.event), el("span", "evt-detail", e.detail || ""));
    box.append(r);
  });
}

/* ---- boot ---- */
async function live() {
  let name = new URLSearchParams(location.search).get("run");
  if (!name) { const r = await fetch("/api/runs").then(x => x.json()); name = (r.runs || [])[0]; }
  if (!name) { render(null); return; }
  const tick = async () => {
    try { render(await fetch("/api/run/" + name).then(x => x.json())); }
    catch (e) { /* keep last */ }
  };
  await tick();
  setInterval(tick, 2500);
}

if (typeof RUN_DATA !== "undefined") render(RUN_DATA);
else live();
