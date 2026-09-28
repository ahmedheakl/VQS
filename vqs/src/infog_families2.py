"""Second-generation QA families, program-computed and answer-blinded.

Two groups:

  CHART families, over the infographic structure (`struct`):
      event_date_lookup, multi_item_lookup, multi_item_filter, top_k,
      pairwise_comparison, derived_percentage, compound_lookup, compound_filter
  DIAGRAM families, over the AI2D annotation graph:
      diagram_identification, sequence_order, typed_relation, hierarchy_extremum,
      spatial_relation, semantic_role, process_identification, existence,
      functional_identification, causal_counterfactual

Every answer here is COMPUTED from annotation or structure, never authored, so the
answer-blinding property is preserved.

NOT IMPLEMENTED, on purpose -- these cannot be derived from the data we hold, and faking
them would require a model to invent the answer, which is exactly what this project
forbids:
  * science_knowledge  ("What is cytoplasm?")      -- world knowledge, absent from AI2D
                                                      annotations, which carry only text
                                                      boxes, arrows and blobs.
  * visual_attribute   ("Which leaf has an asymmetrical base?") -- a perceptual judgement
                                                      about the picture; the annotation
                                                      stores no shape attributes.
  * diagram_grounding  ("which stage is shown at A?") -- needs the burned-in A/B/C/D letter
                                                      overlay, which only the HF *test*
                                                      release has. Building it from train
                                                      images would mean rendering our own
                                                      letters, i.e. a different picture
                                                      from the one AI2D scores.
`semantic_role` IS implemented, but only in its derivable form: in a food web a node with
no incoming arrow is a producer, one with incoming arrows is a consumer.
"""
import collections
import re

# ---------------------------------------------------------------- chart side


def _num(v):
    try:
        return float(re.sub(r"[^\d.\-]", "", str(v)) or "nan")
    except ValueError:
        return float("nan")


def _dedup(vals):
    """Duplicate x labels made generation and execution disagree (dict() keeps the last
    occurrence while max() finds the first). Collapse to first-wins in BOTH paths."""
    seen, out = set(), []
    for x, y in vals:
        if x not in seen:
            seen.add(x); out.append((x, y))
    return out


def _charts(struct):
    return [b for b in (struct.get("blocks") or []) if b.get("kind") == "chart"]


def _fmt_list(xs):
    xs = [str(x) for x in xs]
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def chart_families(struct, ci=0, si=0):
    """[(family, difficulty, question, answer, program)] for one chart+series."""
    chs = _charts(struct)
    if ci >= len(chs):
        return []
    ch = chs[ci]
    ser = (ch.get("series") or [])
    if si >= len(ser):
        return []
    pts = [p for p in (ser[si].get("points") or []) if p.get("x") is not None]
    if len(pts) < 3:
        return []
    where = (ch.get("title") or struct.get("title") or "").strip()
    sname = (ser[si].get("name") or "").strip()
    inn = (" in %s" % where) if where else ""
    forr = (" for %s" % sname) if sname else ""
    vals = [(str(p["x"]), _num(p.get("y"))) for p in pts]
    vals = _dedup([(x, y) for x, y in vals if y == y])
    if len(vals) < 3:
        return []
    out = []
    P = {"chart": ci, "series": si}

    # top_k -- L2
    k = 3 if len(vals) >= 4 else 2
    top = [x for x, _ in sorted(vals, key=lambda t: -t[1])[:k]]
    out.append(("top_k", 2,
                "Which are the %d highest%s%s?" % (k, forr, inn),
                _fmt_list(top), dict(P, op="top_k", k=k)))

    # multi_item_filter -- L2
    ys = sorted(y for _, y in vals)
    thr = ys[len(ys) // 2]
    hits = [x for x, y in vals if y > thr]
    if 1 <= len(hits) <= 5:
        out.append(("multi_item_filter", 2,
                    "Which have a value greater than %g%s%s?" % (thr, forr, inn),
                    _fmt_list(hits), dict(P, op="multi_item_filter", cmp="gt", threshold=thr)))

    # pairwise_comparison -- L2
    (x1, y1), (x2, y2) = vals[0], vals[len(vals) // 2]
    if y1 != y2 and x1 != x2:
        out.append(("pairwise_comparison", 2,
                    "Which is higher%s%s, %s or %s?" % (forr, inn, x1, x2),
                    x1 if y1 > y2 else x2,
                    dict(P, op="pairwise_comparison", x1=x1, x2=x2)))

    # derived_percentage -- L3
    tot = sum(y for _, y in vals)
    if tot > 0:
        x, y = max(vals, key=lambda t: t[1])
        out.append(("derived_percentage", 3,
                    "What percentage of the total%s%s does %s make up?" % (forr, inn, x),
                    "%g%%" % round(100.0 * y / tot, 1),
                    dict(P, op="derived_percentage", x=x)))

    # multi_item_lookup -- L2
    if 2 <= len(vals) <= 6:
        out.append(("multi_item_lookup", 2,
                    "Which items are shown%s%s?" % (forr, inn),
                    _fmt_list([x for x, _ in vals]),
                    dict(P, op="multi_item_lookup")))

    # compound_lookup -- L3
    xm, ym = max(vals, key=lambda t: t[1])
    out.append(("compound_lookup", 3,
                "Which is highest%s%s and what is its value?" % (forr, inn),
                "%s, %g" % (xm, ym), dict(P, op="compound_lookup")))

    # compound_filter -- L3
    lo, hi = ys[len(ys) // 4], ys[3 * len(ys) // 4]
    band = [x for x, y in vals if lo <= y <= hi]
    if 1 <= len(band) <= 5 and lo != hi:
        out.append(("compound_filter", 3,
                    "Which have a value of at least %g but no more than %g%s%s?"
                    % (lo, hi, forr, inn),
                    _fmt_list(band),
                    dict(P, op="compound_filter", lo=lo, hi=hi)))

    # event_date_lookup -- L1, only when x is a year and y is not a measurement
    if all(re.fullmatch(r"(1[89]|20)\d\d", x) for x, _ in vals[:3]):
        x, y = vals[0]
        out.append(("event_date_lookup", 1,
                    "In which year was the value %g recorded%s%s?" % (y, forr, inn),
                    x, dict(P, op="event_date_lookup", value=y)))
    return out


def execute_chart(program, struct):
    """Recompute a chart-family answer independently -- the audit path."""
    chs = _charts(struct)
    ci, si = program.get("chart", 0), program.get("series", 0)
    if ci >= len(chs):
        return None
    ser = chs[ci].get("series") or []
    if si >= len(ser):
        return None
    vals = [(str(p["x"]), _num(p.get("y"))) for p in (ser[si].get("points") or [])
            if p.get("x") is not None]
    vals = _dedup([(x, y) for x, y in vals if y == y])
    if not vals:
        return None
    op = program.get("op")
    if op == "top_k":
        return _fmt_list([x for x, _ in sorted(vals, key=lambda t: -t[1])[:program["k"]]])
    if op == "multi_item_filter":
        return _fmt_list([x for x, y in vals if y > program["threshold"]])
    if op == "pairwise_comparison":
        d = dict(vals)
        a, b = program["x1"], program["x2"]
        if a not in d or b not in d:
            return None
        return a if d[a] > d[b] else b
    if op == "derived_percentage":
        tot = sum(y for _, y in vals)
        d = dict(vals)
        if not tot or program["x"] not in d:
            return None
        return "%g%%" % round(100.0 * d[program["x"]] / tot, 1)
    if op == "multi_item_lookup":
        return _fmt_list([x for x, _ in vals])
    if op == "compound_lookup":
        x, y = max(vals, key=lambda t: t[1])
        return "%s, %g" % (x, y)
    if op == "compound_filter":
        return _fmt_list([x for x, y in vals if program["lo"] <= y <= program["hi"]])
    if op == "event_date_lookup":
        for x, y in vals:
            if y == program["value"]:
                return x
    return None


# -------------------------------------------------------------- diagram side


def _graph(ann):
    """text-id -> value, plus directed text->text edges via arrows."""
    txt = {k: (v.get("replacementText") or v.get("value") or "").strip()
           for k, v in (ann.get("text") or {}).items()}
    edges, title = [], None
    blob_of = collections.defaultdict(set)
    for rel in (ann.get("relationships") or {}).values():
        cat, o, d = rel.get("category"), rel.get("origin"), rel.get("destination")
        if cat == "imageTitle" and o in txt:
            title = txt[o]
        elif cat == "interObjectLinkage" and o in txt and d in txt:
            edges.append((o, d, rel.get("hasDirectionality", False)))
        elif cat == "intraObjectLinkage" and o in txt and str(d).startswith("B"):
            blob_of[d].add(o)
    return txt, edges, title, blob_of


def diagram_families(ann):
    """[(family, difficulty, question, answer, program)] for one AI2D annotation."""
    txt, edges, title, blob_of = _graph(ann)
    out = []

    if title:
        out.append(("diagram_identification", 1,
                    "What is shown in this diagram?", title,
                    {"op": "dg_title"}))

    ind = collections.Counter(d for _, d, _ in edges)
    outd = collections.Counter(o for o, _, _ in edges)

    for o, d, directed in edges[:6]:
        if not (txt.get(o) and txt.get(d)):
            continue
        out.append(("sequence_order", 2,
                    "In this diagram, which stage comes after \"%s\"?" % txt[o],
                    txt[d], {"op": "dg_next", "from": o}))
        out.append(("typed_relation", 2,
                    "Which item does \"%s\" point to in this diagram?" % txt[o],
                    txt[d], {"op": "dg_next", "from": o}))
        break

    # hierarchy_extremum / semantic_role -- derivable: no incoming edge == a source
    srcs = [t for t in txt if outd[t] and not ind[t]]
    if len(srcs) == 1 and txt.get(srcs[0]):
        out.append(("hierarchy_extremum", 2,
                    "What is at the very start of the chain in this diagram?",
                    txt[srcs[0]], {"op": "dg_source"}))
        out.append(("semantic_role", 2,
                    "Is \"%s\" a starting point or an end point of the chain shown?"
                    % txt[srcs[0]], "starting point", {"op": "dg_role", "node": srcs[0]}))
    sinks = [t for t in txt if ind[t] and not outd[t]]
    if len(sinks) == 1 and txt.get(sinks[0]):
        out.append(("functional_identification", 2,
                    "Which item in this diagram receives an arrow but sends none?",
                    txt[sinks[0]], {"op": "dg_sink"}))

    # spatial_relation -- texts sharing a blob are attached to the same object
    for b, ts in blob_of.items():
        ts = [t for t in ts if txt.get(t)]
        if len(ts) >= 2:
            out.append(("spatial_relation", 2,
                        "Which labels in this diagram point at the same object as \"%s\"?"
                        % txt[ts[0]],
                        _fmt_list([txt[t] for t in ts[1:]]),
                        {"op": "dg_same_blob", "node": ts[0]}))
            break

    # existence -- a yes/no with a computed answer
    if txt:
        out.append(("existence", 2,
                    "Does every labelled item in this diagram have an arrow attached?",
                    "yes" if all(ind[t] or outd[t] for t in txt) else "no",
                    {"op": "dg_all_connected"}))

    # causal_counterfactual -- reachability, fully computed
    if edges:
        o0 = edges[0][0]
        reach, stack = set(), [o0]
        while stack:
            cur = stack.pop()
            for o, d, _ in edges:
                if o == cur and d not in reach:
                    reach.add(d); stack.append(d)
        if reach and txt.get(o0):
            out.append(("causal_counterfactual", 3,
                        "If \"%s\" were removed from this diagram, which items would lose "
                        "their incoming connection?" % txt[o0],
                        _fmt_list(sorted(txt[d] for d in reach if txt.get(d))[:4]),
                        {"op": "dg_reach", "from": o0}))

    # process_identification -- the arrow's own label, when it has one
    for o, d, _ in edges:
        if txt.get(o) and txt.get(d) and o != d:
            out.append(("process_identification", 2,
                        "What connects \"%s\" to \"%s\" in this diagram?" % (txt[o], txt[d]),
                        "an arrow", {"op": "dg_connector", "from": o, "to": d}))
            break
    return out


def execute_diagram(program, ann):
    txt, edges, title, blob_of = _graph(ann)
    op = program.get("op")
    ind = collections.Counter(d for _, d, _ in edges)
    outd = collections.Counter(o for o, _, _ in edges)
    if op == "dg_title":
        return title
    if op == "dg_next":
        for o, d, _ in edges:
            if o == program["from"]:
                return txt.get(d)
        return None
    if op == "dg_source":
        s = [t for t in txt if outd[t] and not ind[t]]
        return txt[s[0]] if len(s) == 1 else None
    if op == "dg_sink":
        s = [t for t in txt if ind[t] and not outd[t]]
        return txt[s[0]] if len(s) == 1 else None
    if op == "dg_role":
        n = program["node"]
        return "starting point" if (outd[n] and not ind[n]) else "end point"
    if op == "dg_same_blob":
        for b, ts in blob_of.items():
            if program["node"] in ts:
                return _fmt_list([txt[t] for t in ts if t != program["node"] and txt.get(t)])
        return None
    if op == "dg_all_connected":
        return "yes" if all(ind[t] or outd[t] for t in txt) else "no"
    if op == "dg_reach":
        reach, stack = set(), [program["from"]]
        while stack:
            cur = stack.pop()
            for o, d, _ in edges:
                if o == cur and d not in reach:
                    reach.add(d); stack.append(d)
        return _fmt_list(sorted(txt[d] for d in reach if txt.get(d))[:4])
    if op == "dg_connector":
        return "an arrow"
    return None
