/* Pipeline explorer. Data: window.VQS_EXAMPLES (static/examples.js), recorded research traces of
   Qwen3-VL-2B on its own parses. Nothing here calls a model. */
(() => {
  const E = window.VQS_EXAMPLES || [];
  if (!E.length) return;
  const byId = Object.fromEntries(E.map(e => [e.id, e]));
  const DOM = { natural: "Photo", charts: "Chart", diagrams: "Diagram", infographics: "Infographic" };
  const OUT = { kept: "Kept", fact: "Dropped at fact check", blind: "Dropped at blind gate" };
  const OUT_SHORT = { kept: "✓ Kept", fact: "✗ Fact check", blind: "✗ Blind gate" };
  const PAL = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2", "#dc2626", "#65a30d",
               "#9333ea", "#0d9488", "#ea580c", "#4f46e5"];
  const TOUR = [
    ["p39", "Chart arithmetic, every operand re-read", "#2563eb"],
    ["p37", "Photo: which furniture is green?", "#059669"],
    ["p25", "L5: compare two aggregates", "#7c3aed"],
    ["p12", "A misread parse caught by the fact check", "#dc2626"],
    ["p36", "Answerable without the image", "#d97706"],
    ["p38", "Infographic difference", "#db2777"],
    ["g12", "An error a human rater caught", "#6d28d9"],
  ].filter(([id]) => byId[id]);

  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const norm = s => String(s ?? "").trim().toLowerCase().replace(/\s+/g, " ");
  const fmtNum = v => typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(1) : +v.toFixed(3)) : esc(v);
  const compact = v => Math.abs(v) >= 1e9 ? +(v / 1e9).toFixed(1) + "B" : Math.abs(v) >= 1e6 ? +(v / 1e6).toFixed(1) + "M"
    : Math.abs(v) >= 1e4 ? +(v / 1e3).toFixed(0) + "k" : +(+v).toFixed(2);

  /* ------------------------------------------------------------ gallery */
  const state = { dom: "all", out: "all", q: "", all: false };
  const tourIds = new Set(TOUR.map(t => t[0]));
  // tour picks first, then the traced set, then the graded set, interleaving the four domains
  const base = (() => {
    const rest = E.filter(e => !tourIds.has(e.id)).sort((a, b) => (a.src === b.src ? 0 : a.src === "trace" ? -1 : 1));
    const buckets = {};
    rest.forEach(e => (buckets[e.domain] = buckets[e.domain] || []).push(e));
    const out = TOUR.map(t => byId[t[0]]);
    for (let i = 0; Object.values(buckets).some(b => b.length); i++)
      for (const d of ["charts", "natural", "infographics", "diagrams"]) if (buckets[d] && buckets[d].length) out.push(buckets[d].shift());
    return out;
  })();
  let order = base.slice();

  function matches(e) {
    if (state.dom !== "all" && e.domain !== state.dom) return false;
    if (state.out === "human" ? !e.human : state.out !== "all" && e.decision !== state.out) return false;
    if (state.q) {
      const hay = norm([e.q.shipped, e.q.template, e.answer, e.family, DOM[e.domain]].join(" "));
      return state.q.split(" ").every(w => hay.includes(w));
    }
    return true;
  }

  function chipRow(el, items, key) {
    el.innerHTML = items.map(([v, label, n]) =>
      `<button class="chip" type="button" data-v="${v}" aria-pressed="${state[key] === v}">${label}<span class="n">${n}</span></button>`).join("");
    el.querySelectorAll(".chip").forEach(b => b.addEventListener("click", () => { state[key] = b.dataset.v; state.all = false; render(); }));
  }

  function render() {
    const cnt = (f) => E.filter(f).length;
    chipRow($("#fDom"), [["all", "All", E.length], ...Object.keys(DOM).map(d => [d, DOM[d] + "s", cnt(e => e.domain === d)])], "dom");
    chipRow($("#fOut"), [["all", "Any outcome", E.length], ["kept", "Kept", cnt(e => e.decision === "kept")],
      ["fact", "Fact check drop", cnt(e => e.decision === "fact")], ["blind", "Blind gate drop", cnt(e => e.decision === "blind")],
      ["human", "Human-graded", cnt(e => e.human)]], "out");
    order = base.filter(matches);
    const shown = state.all ? order : order.slice(0, 12);
    $("#xGrid").innerHTML = shown.map(card).join("") ||
      `<p class="note" style="grid-column:1/-1">No example matches these filters.</p>`;
    $("#xGrid").querySelectorAll(".xcard").forEach(c => c.addEventListener("click", () => openViewer(c.dataset.id)));
    $("#xCount").textContent = `Showing ${shown.length} of ${order.length} matching examples (${E.length} in total).`;
    $("#xMore").innerHTML = order.length > shown.length
      ? `<button class="btn" type="button" id="xMoreBtn">Show all ${order.length}</button>` : "";
    const mb = $("#xMoreBtn");
    if (mb) mb.addEventListener("click", () => { state.all = true; render(); });
  }

  function card(e) {
    const h = e.human ? `<span class="tag human ${e.human.correct ? "" : "bad"}">${e.human.correct ? "Human ✓" : "Human ✗"}</span>` : "";
    return `<button class="xcard" type="button" data-id="${e.id}" aria-label="Open example: ${esc(e.q.shipped)}">
      <div class="xthumb"><img loading="lazy" src="${e.thumb}" alt=""><span class="st d-${e.decision}">${OUT_SHORT[e.decision]}</span></div>
      <div class="xbody">
        <div class="xmeta"><span class="tag dom-${e.domain}">${DOM[e.domain]}</span><span class="tag lvl">L${e.level}</span>${h}</div>
        <div class="xq">${esc(e.q.shipped)}</div>
        <div class="xans">Answer <b>${esc(e.answer)}</b></div>
      </div></button>`;
  }

  /* ------------------------------------------------------------ viewer */
  const dlg = $("#viewer");
  let cur = null;

  function openViewer(id, replay = true) {
    const e = byId[id];
    if (!e) return;
    cur = id;
    const idx = order.findIndex(x => x.id === id);
    const h = e.human ? `<span class="tag human ${e.human.correct ? "" : "bad"}">${e.human.correct ? "Human check ✓" : "Human check ✗"}</span>` : "";
    dlg.innerHTML = `
      <div class="vhead">
        <div class="vtitle" id="vTitle"><span class="tag dom-${e.domain}">${DOM[e.domain]}</span><span class="tag lvl">${esc(e.levelName)}</span>
          <span class="fam">${esc(e.family)}</span><span class="st d-${e.decision}">${OUT[e.decision]}</span>${h}</div>
        <div class="vnav">
          <button class="vbtn" type="button" id="vPrev" aria-label="Previous example" ${idx <= 0 ? "disabled" : ""}>←</button>
          <button class="vbtn" type="button" id="vNext" aria-label="Next example" ${idx < 0 || idx >= order.length - 1 ? "disabled" : ""}>→</button>
          <button class="vbtn" type="button" id="vClose" aria-label="Close">✕</button>
        </div>
      </div>
      <div class="vbody">
        <div class="vstage">${stage(e)}</div>
        <div class="vpanels" id="vPanels">${stepper(e)}${panels(e)}</div>
      </div>`;
    wireViewer(e, idx);
    if (!dlg.open) dlg.showModal();
    $("#vPanels").scrollTop = 0;
    history.replaceState(null, "", "#ex-" + id);
    if (replay) animate(); else dlg.querySelectorAll(".step-pill").forEach(p => p.classList.add("on"));
  }

  function wireViewer(e, idx) {
    $("#vClose").addEventListener("click", () => dlg.close());
    $("#vPrev").addEventListener("click", () => idx > 0 && openViewer(order[idx - 1].id));
    $("#vNext").addEventListener("click", () => idx < order.length - 1 && openViewer(order[idx + 1].id));
    dlg.querySelectorAll(".step-pill").forEach(p => p.addEventListener("click", () => {
      const t = document.getElementById(p.dataset.to);
      if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    const tog = $("#vBoxes");
    if (tog) tog.addEventListener("change", () => $("#vImg").classList.toggle("noboxes", !tog.checked));
    dlg.querySelectorAll(".obj[data-oid]").forEach(o => {
      const on = v => dlg.querySelectorAll(`[data-oid="${o.dataset.oid}"]`).forEach(x => x.classList.toggle("hl", v));
      o.addEventListener("mouseenter", () => on(true));
      o.addEventListener("mouseleave", () => on(false));
    });
    const raw = $("#vRaw");
    if (raw) raw.addEventListener("click", () => {
      const pre = $("#vJson");
      pre.hidden = !pre.hidden;
      raw.textContent = pre.hidden ? "Show raw parse" : "Hide raw parse";
    });
    dlg.querySelectorAll(".guess").forEach(g => g.addEventListener("click", () => g.classList.toggle("open")));
    const rp = $("#vReplay");
    if (rp) rp.addEventListener("click", animate);
  }

  function animate() {
    const pills = [...dlg.querySelectorAll(".step-pill")];
    pills.forEach(p => p.classList.remove("on"));
    pills.forEach((p, i) => setTimeout(() => p.classList.add("on"), 160 + i * 170));
  }

  dlg.addEventListener("close", () => { history.replaceState(null, "", location.pathname + location.search); cur = null; });
  dlg.addEventListener("click", ev => { if (ev.target === dlg) dlg.close(); });
  document.addEventListener("keydown", ev => {
    if (!dlg.open || !cur) return;
    if (ev.key === "ArrowRight") $("#vNext")?.click();
    if (ev.key === "ArrowLeft") $("#vPrev")?.click();
  });

  /* ------------------------------------------------------------ stage (image + question) */
  function stage(e) {
    const boxes = e.domain === "natural" ? photoBoxes(e) : "";
    const bar = e.domain === "natural"
      ? `<div class="stagebar"><label class="switch"><input type="checkbox" id="vBoxes" checked> Show parsed boxes</label><span>hover an object to find it</span></div>`
      : `<div class="stagebar"><span>Image as the parser saw it</span></div>`;
    return `<div class="imgwrap" id="vImg"><img src="${e.img}" alt="Input image" width="${e.w}" height="${e.h}">${boxes}</div>${bar}
      <div class="qbox"><div class="lbl">Question the solver trains on</div><div class="qq">${esc(e.q.shipped)}</div>
      <div class="aa">Computed answer <b>${esc(e.answer)}</b></div></div>`;
  }

  function objects(e) {
    return (e.parse.objects || []).map((o, i) => ({ ...o, color: PAL[i % PAL.length], oid: `o${i}` }));
  }

  function focusNames(e) {
    const q = [e.q.shipped, JSON.stringify(e.program.operands || {}), e.answer].map(norm).join(" ");
    return new Set(objects(e).filter(o => o.object_name && new RegExp(`\\b${norm(o.object_name).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(s|es)?\\b`).test(q)).map(o => o.oid));
  }

  function photoBoxes(e) {
    const focus = focusNames(e);
    const obs = objects(e).filter(o => Array.isArray(o.box_0_1000) && o.box_0_1000.length === 4);
    const rects = obs.map(o => {
      let [x0, y0, x1, y1] = o.box_0_1000.map(v => Math.max(0, Math.min(1000, +v)));
      if (x1 < x0) [x0, x1] = [x1, x0];
      if (y1 < y0) [y0, y1] = [y1, y0];
      o._b = [x0, y0, x1, y1];
      const w = focus.has(o.oid) ? 3.5 : 2;
      return `<rect data-oid="${o.oid}" class="bx" x="${x0}" y="${y0}" width="${Math.max(x1 - x0, 2)}" height="${Math.max(y1 - y0, 2)}"
        fill="${o.color}" fill-opacity="${focus.has(o.oid) ? 0.16 : 0.06}" stroke="${o.color}" stroke-width="${w}" vector-effect="non-scaling-stroke" rx="4"/>`;
    }).join("");
    const labels = obs.map(o => `<span class="blabel" data-oid="${o.oid}" style="left:${o._b[0] / 10}%;top:${Math.max(o._b[1] / 10, 3)}%;background:${o.color}">${esc(o.object_name)}</span>`).join("");
    return `<svg class="boxes" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">${rects}</svg>${labels}`;
  }

  /* ------------------------------------------------------------ stepper + panels */
  function stepper(e) {
    const rewrote = e.q.shipped !== e.q.template;
    const fact = !e.facts.length ? "skip" : e.fact_ok ? "ok" : "bad";
    const blind = e.decision === "fact" ? "skip" : e.blind.passed ? "ok" : "bad";
    const steps = [
      ["pParse", "Parse", "ok", "◧"], ["pProg", "Program", "ok", "ƒ"],
      ["pQ", "Rewrite", rewrote ? "ok" : "skip", "✎"],
      ["pFact", "Fact check", fact, fact === "bad" ? "✗" : fact === "ok" ? "✓" : "–"],
      ["pBlind", "Blind gate", blind, blind === "bad" ? "✗" : blind === "ok" ? "✓" : "–"],
      ["pOut", "Outcome", e.decision === "kept" ? "ok" : "bad", e.decision === "kept" ? "✓" : "✗"]];
    return `<div class="stepper" role="list">${steps.map(([to, t, s, ic]) =>
      `<button type="button" class="step-pill ${s}" data-to="${to}" role="listitem"><span class="ic">${ic}</span>${t}</button>`).join("")}</div>`;
  }

  function panels(e) {
    return [parsePanel(e), programPanel(e), questionPanel(e), factPanel(e), blindPanel(e), outcomePanel(e)].join("");
  }

  const panel = (id, n, title, right, body) =>
    `<section class="panel" id="${id}"><h4><span class="num">${n}</span>${title}${right ? `<span class="right">${right}</span>` : ""}</h4>${body}</section>`;

  /* ---- 1. parse */
  function parsePanel(e) {
    const body = e.domain === "natural" ? photoParse(e) : e.domain === "charts" ? chartParse(e)
      : e.domain === "diagrams" ? diagramParse(e) : infoParse(e);
    return panel("pParse", 1, "Structured parse", "schema-constrained, greedy",
      `<p class="hint">The parser's output for this image, under the ${DOM[e.domain].toLowerCase()} schema. Programs read only this record.</p>${body}
       <button class="linkbtn" type="button" id="vRaw">Show raw parse</button><pre class="json" id="vJson" hidden>${hlJson(e.parse)}</pre>`);
  }

  function photoParse(e) {
    const focus = focusNames(e), obs = objects(e);
    const names = Object.fromEntries((e.parse.objects || []).map(o => [o.object_id, o.object_name]));
    const list = obs.map(o => `<div class="obj ${focus.has(o.oid) ? "focus" : ""}" data-oid="${o.oid}"><span class="sw" style="background:${o.color}"></span>
      <span class="nm">${esc(o.object_name)}</span>${(o.attributes || []).map(a => `<span class="at">${esc(a)}</span>`).join("")}</div>`).join("");
    const rels = (e.parse.relations || []).map(r => `<span class="rel">${esc(names[r.subject_id] ?? "?")} <i>${esc(r.predicate)}</i> ${esc(names[r.object_id] ?? "?")}</span>`).join("");
    return `<div class="ptitle">${obs.length} objects <span>· ${(e.parse.relations || []).length} relations</span></div>
      <div class="objs">${list}</div>${rels ? `<div class="rels">${rels}</div>` : ""}`;
  }

  // cells the program read, and cells a read-back probe checked
  function cellMarks(e) {
    const reads = (e.program.reads || []).map(r => String(r).replace(/^CELL\(/, "").replace(/\)$/, ""));
    const probes = e.facts.map(f => {
      const m = /value (?:of|for) (.+?)(?: for (.+?))?\?/.exec(f.probe || "");
      return m ? { x: norm(m[1]), s: m[2] ? norm(m[2]) : null, ok: f.passed, got: f.got } : null;
    }).filter(Boolean);
    return (series, cat) => {
      const read = reads.some(r => norm(r).startsWith(norm(`${series},${cat},`)));
      const pr = probes.find(p => p.x === norm(cat) && (!p.s || p.s === norm(series)));
      return { read, fail: pr && !pr.ok, checked: !!pr, got: pr ? pr.got : null };
    };
  }

  function chartParse(e) {
    const P = e.parse, S = P.series || [];
    const cats = [];
    S.forEach(s => (s.points || []).forEach(p => { if (!cats.includes(p.category)) cats.push(p.category); }));
    const vals = S.flatMap(s => (s.points || []).map(p => p.numeric_value).filter(v => typeof v === "number"));
    const mark = cellMarks(e);
    let svg = "";
    if (vals.length && cats.length <= 30) {
      const W = 560, H = 200, L = 44, B = 42, T = 8, max = Math.max(0, ...vals), min = Math.min(0, ...vals);
      const y = v => T + (H - T - B) * (max - v) / ((max - min) || 1);
      const gw = (W - L - 6) / cats.length, bw = Math.max(2, Math.min(26, (gw - 4) / Math.max(S.length, 1)));
      let bars = "";
      S.forEach((s, si) => (s.points || []).forEach(p => {
        if (typeof p.numeric_value !== "number") return;
        const ci = cats.indexOf(p.category), m = mark(s.series_name, p.category);
        const x = L + ci * gw + (gw - bw * S.length) / 2 + si * bw, y0 = y(0), y1 = y(p.numeric_value);
        const stroke = m.fail ? `stroke="#dc2626" stroke-width="2.5"` : m.read ? `stroke="#111827" stroke-width="1.5"` : "";
        bars += `<rect x="${x}" y="${Math.min(y0, y1)}" width="${bw - 1}" height="${Math.max(Math.abs(y1 - y0), 1)}" rx="2" fill="${PAL[si % PAL.length]}" fill-opacity="${m.read || m.fail ? 1 : 0.55}" ${stroke}><title>${esc(s.series_name)} · ${esc(p.category)}: ${esc(p.printed_value)}</title></rect>`;
      }));
      const step = Math.ceil(cats.length / 14);
      const labels = cats.map((c, i) => i % step ? "" : `<text x="${L + i * gw + gw / 2}" y="${H - B + 14}" text-anchor="end" transform="rotate(-35 ${L + i * gw + gw / 2} ${H - B + 14})">${esc(String(c).slice(0, 14))}</text>`).join("");
      const axis = `<line x1="${L}" x2="${W}" y1="${y(0)}" y2="${y(0)}" stroke="currentColor" stroke-opacity=".25"/>
        <text x="${L - 4}" y="${y(max) + 4}" text-anchor="end">${compact(max)}</text>${min < 0 ? `<text x="${L - 4}" y="${y(min)}" text-anchor="end">${compact(min)}</text>` : ""}`;
      svg = `<svg class="pchart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Bars re-plotted from the parsed values">${axis}${bars}${labels}</svg>`;
    }
    const rank = m => m.fail ? 0 : m.checked ? 1 : m.read ? 2 : 3;
    const rows = S.flatMap(s => (s.points || []).map(p => ({ s, p, m: mark(s.series_name, p.category) })))
      .sort((a, b) => rank(a.m) - rank(b.m)).map(({ s, p, m }) => {
      return `<tr class="${m.fail ? "fail" : m.read || m.checked ? "read" : ""}"><td class="mk">${m.fail ? "✗" : m.checked ? "✓" : m.read ? "•" : ""}</td>
        ${S.length > 1 ? `<td>${esc(s.series_name)}</td>` : ""}<td>${esc(p.category)}</td><td>${esc(p.printed_value)}</td></tr>`;
    }).join("");
    return `<div class="ptitle">${esc(P.title || "Untitled chart")} <span>· ${S.length} series · ${cats.length} categories</span></div>${svg}
      <div style="max-height:230px;overflow:auto"><table class="ptable"><thead><tr><th></th>${S.length > 1 ? "<th>Series</th>" : ""}<th>Category</th><th>Printed value</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="legend-mini"><span style="--c:#111827">read by the program</span><span style="--c:#059669">re-read and confirmed</span><span style="--c:#dc2626">re-read and contradicted</span></div>`;
  }

  function infoParse(e) {
    const P = e.parse, mark = cellMarks(e);
    const rank = m => m.fail ? 0 : m.checked ? 1 : m.read ? 2 : 3;
    const rows = (P.entries || []).map(x => ({ x, m: mark("", x.entry_label) })).sort((a, b) => rank(a.m) - rank(b.m)).map(({ x, m }) => {
      return `<tr class="${m.fail ? "fail" : m.read || m.checked ? "read" : ""}"><td class="mk">${m.fail ? "✗" : m.checked ? "✓" : m.read ? "•" : ""}</td>
        <td>${esc(x.entry_label)}</td><td>${esc(x.printed_value)}</td><td>${esc(x.unit)}</td></tr>`;
    }).join("");
    return `<div class="ptitle">${esc(P.title || "Untitled")} <span>· ${(P.entries || []).length} entries</span></div>
      <div style="max-height:260px;overflow:auto"><table class="ptable"><thead><tr><th></th><th>Entry</th><th>Printed value</th><th>Unit</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function diagramParse(e) {
    const P = e.parse, N = P.nodes || [], Ed = P.has_arrows ? (P.edges || []) : [];
    const W = 520, H = Math.min(260, 110 + N.length * 16), cx = W / 2, cy = H / 2, r = Math.min(H / 2 - 22, 26 + N.length * 9);
    const pos = Object.fromEntries(N.map((n, i) => [n.node_id, [cx + r * 1.6 * Math.cos(2 * Math.PI * i / N.length - Math.PI / 2), cy + r * Math.sin(2 * Math.PI * i / N.length - Math.PI / 2)]]));
    const edges = Ed.map(ed => {
      const a = pos[ed.from_node_id], b = pos[ed.to_node_id];
      if (!a || !b) return "";
      const dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1, k = 22 / d;
      return `<line x1="${a[0] + dx * k}" y1="${a[1] + dy * k}" x2="${b[0] - dx * k}" y2="${b[1] - dy * k}" stroke="#7c3aed" stroke-width="1.8" marker-end="url(#arr)"/>`;
    }).join("");
    const nodes = N.map((n, i) => {
      const [x, y] = pos[n.node_id], t = String(n.node_label).slice(0, 26), w = Math.max(40, t.length * 6.4 + 14);
      return `<g><rect x="${x - w / 2}" y="${y - 11}" width="${w}" height="22" rx="11" fill="${PAL[i % PAL.length]}" fill-opacity=".14" stroke="${PAL[i % PAL.length]}"/><text x="${x}" y="${y + 4}" text-anchor="middle">${esc(t)}</text></g>`;
    }).join("");
    return `<div class="ptitle">${esc(P.title || "Untitled diagram")} <span>· ${N.length} nodes · ${Ed.length} arrows${P.has_arrows ? "" : " (has_arrows = false, so edges are ignored)"}</span></div>
      <svg class="pgraph" viewBox="0 0 ${W} ${H}" role="img" aria-label="Parsed diagram graph"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#7c3aed"/></marker></defs>${edges}${nodes}</svg>`;
  }

  /* ---- 2. program */
  function programPanel(e) {
    const ops = e.program.operands && typeof e.program.operands === "object" ? Object.entries(e.program.operands) : [];
    const kv = ops.filter(([, v]) => v !== null && v !== "" && v !== "__none__").map(([k, v]) =>
      `<span><b>${esc(k)}=</b>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</span>`).join("");
    const reads = (e.program.reads || []).length;
    const code = e.program.executor ? `<pre class="code">${hlPy(e.program.executor)}</pre>` : "";
    return panel("pProg", 2, "Template program", esc(e.levelName),
      `<p class="hint">A fixed program written once by hand. It reads the parse and computes the answer; no model is asked for it.</p>
       ${kv ? `<div class="kv">${kv}</div>` : ""}${code}
       <div class="computed"><span>Computed answer</span><b>${esc(e.answer)}</b>${reads ? `<span class="note" style="margin:0">from ${reads} parsed cell${reads > 1 ? "s" : ""}</span>` : ""}</div>`);
  }

  /* ---- 3. question rewrite */
  function questionPanel(e) {
    const q = e.q;
    let guard = "";
    if (q.raw != null) {
      const rej = q.rejected && q.rejected.length;
      guard = `<div class="row"><span class="k">Rewrite</span><span class="v">${esc(q.raw)}</span></div>
        <div class="row"><span class="k">Guard</span><span class="v">${rej ? `<span class="guard no">✗ rejected: ${esc(q.rejected.join(", "))}</span>`
          : q.raw === q.template ? `<span class="guard ok">✓ unchanged</span>` : `<span class="guard ok">✓ accepted</span>`}</span></div>`;
    }
    return panel("pQ", 3, "Question rewrite", q.shipped !== q.template ? "paraphrased" : "template wording kept",
      `<p class="hint">The parser rewrites the template question in natural wording; a rewrite that adds a content word, drops the question mark or names the answer is rejected.</p>
       <div class="flow"><div class="row"><span class="k">Template</span><span class="v mono">${esc(q.template)}</span></div>${guard}
       <div class="row"><span class="k">Shipped</span><span class="v">${esc(q.shipped)}</span></div></div>`);
  }

  /* ---- 4. fact check */
  function factPanel(e) {
    if (!e.facts.length) {
      return panel("pFact", 4, "Fact check", "nothing to check",
        `<p class="hint">This program records no cell-level claim, so the question passes through the fact check unchanged.</p>`);
    }
    const ok = e.facts.filter(f => f.passed).length;
    const items = e.facts.map(f => {
      const cls = f.kind === "READ" ? "info" : f.passed ? "ok" : "bad";
      const ic = f.kind === "READ" ? "?" : f.passed ? "✓" : "✗";
      const cmp = f.kind === "READ"
        ? `<span>model read <b>${fmtNum(f.got)}</b></span>`
        : f.kind === "ARITH"
          ? `<span>recomputed <b>${fmtNum(f.got)}</b></span><span>program <b>${fmtNum(f.expected)}</b></span>`
          : `<span>parse says <b>${fmtNum(f.expected)}</b></span><span>model read <b>${fmtNum(f.got)}</b></span>`;
      return `<div class="fact ${cls}"><span class="ic">${ic}</span><div><div class="pr">${esc(f.probe)}</div><div class="cmp">${cmp}</div></div></div>`;
    }).join("");
    const verdict = e.fact_ok
      ? `<div class="verdict ok">Every checked fact holds, so the question moves on.</div>`
      : `<div class="verdict bad">A read-back contradicts the parse, so the computed answer cannot be trusted: the question is dropped.</div>`;
    return panel("pFact", 4, "Fact check", `${ok}/${e.facts.length} passed`,
      `<p class="hint">The same model looks at the image again and reads back each value the program used; arithmetic is redone from the read-backs.</p>
       <div class="facts-list">${items}</div>${verdict}`);
  }

  /* ---- 5. blind gate */
  function blindPanel(e) {
    const b = e.blind, k = b.hits.filter(Boolean).length, n = b.guesses.length;
    const gs = b.guesses.map((g, i) => `<div class="guess ${b.hits[i] ? "hit" : ""}" title="Click to expand"><span class="gi">${b.hits[i] ? "=" : "≠"}</span><div class="gt">${esc(g)}</div></div>`).join("");
    const segs = Array.from({ length: n }, (_, i) => `<div class="seg ${i < k ? "on" : ""}"></div>`).join("");
    const reached = e.decision !== "fact";
    const verdict = !reached
      ? `<div class="verdict warn">Not reached: the question was already dropped by the fact check. Recorded guesses shown for reference.</div>`
      : b.passed
        ? `<div class="verdict ok">${k} of ${n} blind answers match (≤ 1/2): the question needs the image, so it is kept.</div>`
        : `<div class="verdict bad">${k} of ${n} blind answers match (> 1/2): the wording alone gives the answer away, so it is dropped.</div>`;
    return panel("pBlind", 5, "Blind gate", `${k}/${n} match the answer`,
      `<p class="hint">The same model answers the question ${n} times with the image withheld. Gold: <b>${esc(e.answer)}</b>.</p>
       <div class="guesses" style="${reached ? "" : "opacity:.6"}">${gs}</div>
       <div class="meter"><div class="track">${segs}<div class="th" title="λ = 1/2"></div></div><div class="ticks"><span>0</span><span>λ = 1/2</span><span>${n}</span></div></div>${verdict}`);
  }

  /* ---- 6. outcome */
  function outcomePanel(e) {
    const t = e.decision === "kept"
      ? ["✓", "Kept for training", "It goes on to the difficulty band and GRPO, rewarded by exact match with the computed answer."]
      : e.decision === "fact"
        ? ["✗", "Dropped at the fact check", "A misread parse would have produced a wrong label; the claim-level check stops it."]
        : ["✗", "Dropped at the blind gate", "The question is answerable from its wording, so it would teach nothing about the image."];
    const hum = e.human
      ? `<div class="verdict ${e.human.correct ? "ok" : "bad"}" style="margin-top:12px">Human rater: ${e.human.correct ? "the answer is correct." : "the answer is wrong."}${e.human.note ? ` <span style="font-weight:500">(${esc(e.human.note)})</span>` : ""}</div>` : "";
    return panel("pOut", 6, "Outcome", `<button class="linkbtn" type="button" id="vReplay">Replay ↻</button>`,
      `<div class="outcome"><span class="big d-${e.decision}">${t[0]}</span><div><div class="t">${t[1]}</div><div class="s">${t[2]}</div></div></div>${hum}`);
  }

  /* ------------------------------------------------------------ highlighting */
  function hlPy(src) {
    const re = /(#[^\n]*)|(f?"(?:\\.|[^"\\])*"|f?'(?:\\.|[^'\\])*')|\b(if|elif|else|return|for|in|not|and|or|None|True|False|def|lambda|while|is)\b|\b(\d+(?:\.\d+)?)\b|\b(len|sum|sorted|max|min|int|float|abs|round|str|set|list|any|all|range|zip|enumerate)\b/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      out += `<span class="${m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "n" : "p"}">${esc(m[0])}</span>`;
      last = re.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  function hlJson(obj) {
    const src = JSON.stringify(obj, null, 2);
    const re = /("(?:\\.|[^"\\])*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b/g;
    let out = "", last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      if (m[1]) out += `<span class="${m[2] ? "jk" : "js"}">${esc(m[1])}</span>${m[2] ? esc(m[2]) : ""}`;
      else out += `<span class="${m[3] ? "jn" : "jb"}">${esc(m[0])}</span>`;
      last = re.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  /* ------------------------------------------------------------ boot */
  $("#tour").innerHTML = TOUR.map(([id, label, c]) => `<button type="button" data-id="${id}"><span class="dot" style="background:${c}"></span>${esc(label)}</button>`).join("");
  $("#tour").querySelectorAll("button").forEach(b => b.addEventListener("click", () => openViewer(b.dataset.id)));
  $("#xSearch").addEventListener("input", ev => { state.q = norm(ev.target.value); state.all = false; render(); });
  render();
  const fromHash = () => {
    const m = /^#ex-(\w+)$/.exec(location.hash);
    if (m && byId[m[1]] && m[1] !== cur) openViewer(m[1], false);
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
})();
