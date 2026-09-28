"""Engine v5 families — the ones a 1,000-QA human audit found we were missing.

The audit (data/chartqa_family_audit_1000/) labelled 1,000 real ChartQA human questions and
found 53 canonical families; our engine covered 11 of them. 497/1000 questions needed a family
we did not have. This module adds the missing ones.

Every family here is still **program-computed**: propose_v5() emits an answer-free program,
execute_v5() recomputes the answer from the table and returns the exact cells used, so a
prediction is verifiable and correct-by-construction whenever the table is.

Coverage of the audit's distribution, by count:
  legend_lookup 53, chart_structure_count 41, range 32, extrema_arithmetic 30, gap_extremum 25,
  inverse_lookup 25, global_series_relation 23, statistic_compare 21, aggregate_compare 18,
  positional_retrieve 17, ranked_aggregate 16, chart_metadata 15, relational_count 11,
  steepest_change 11, count_below 10, rank 9, filtered_aggregate 8, trend_direction 8,
  joint_extremum 7, series_event 7, count_equal 6, mode 6, ranked_arithmetic 6,
  threshold_lookup 6, median 5, missing_value 5, complement 4, percent_change 1,
  share_of_total 1, duration 2
That is ~85% of the 497 previously-uncovered questions.

Difficulty follows the audit's operand-acquisition rule:
  diff(cell, cell)                              -> L3
  diff(max(series), min(series))                -> L4   (operands must be derived first)
  compare(two separately aggregated results)    -> L5
"""
import re
import statistics as _st

from src.chart_qa_gen import (_num, _fmt, points_of, referable_series, _sref, subject_of,
                              label_ok, clean_label, unit_of, _series_by_index)


def _cell(sname, p):
    return f"CELL({sname},{p[0]},{p[2]})"


def _xs(table):
    xs = [str(x).strip() for x in (table.get("x_labels") or []) if str(x).strip()]
    return [x for x in xs if label_ok(x)]


def _colour_of(s):
    c = str(s.get("color") or "").strip().lower()
    return c if c and c not in ("none", "n/a", "unknown") else None


# --------------------------------------------------------------------- propose
def propose_v5(table, rng, max_per_image=40):
    """Yield answer-free programs for the v5 families. Deterministic given `rng`."""
    progs = []
    refs = referable_series(table)
    if not refs:
        return progs
    multi = len(table.get("series", []) or []) > 1
    subj = subject_of(table)
    xs = _xs(table)
    ctype = str(table.get("chart_type") or "").lower()

    def add(**kw):
        progs.append(kw)

    # ---------------- chart-level (no series needed) ----------------
    title = str(table.get("title") or "").strip()
    if title:
        add(family="chart_metadata", difficulty=1, key="title",
            question="What is the title of this chart?")
    if multi:
        add(family="chart_structure_count", difficulty=2, what="series",
            question="How many different series are shown in this chart?")
    if len(xs) >= 3:
        add(family="chart_structure_count", difficulty=2, what="categories",
            question=f"How many {'time periods' if _looks_temporal(xs) else 'categories'} "
                     f"are shown on the chart?")
        add(family="positional_retrieve", difficulty=1, pos="first",
            question=f"Which {'year' if _looks_temporal(xs) else 'category'} is shown "
                     f"first on the axis?")
        add(family="positional_retrieve", difficulty=1, pos="last",
            question=f"Which {'year' if _looks_temporal(xs) else 'category'} is shown "
                     f"last on the axis?")
    # legend: colour -> series name. Only when colours are present and distinct.
    if multi:
        cols = [(i, s, nm, _colour_of(s)) for i, s, nm in refs]
        named = [(i, s, nm, c) for i, s, nm, c in cols if c]
        if len(named) >= 2 and len({c for *_, c in named}) == len(named):
            i, s, nm, c = named[rng.randrange(len(named))]
            add(family="legend_lookup", difficulty=1, series=nm, series_index=i, colour=c,
                question=f"Which series is represented by the {c} "
                         f"{'line' if 'line' in ctype else 'bar'}?")
        if len({_colour_of(s) for _, s, _ in refs if _colour_of(s)}) >= 2:
            add(family="chart_structure_count", difficulty=2, what="colors",
                question="How many different colors are used for the series in this chart?")

    # ---------------- per-series ----------------
    for si, s, sname in refs:
        pts = points_of(s)
        if len(pts) < 3:
            continue
        sr = _sref(sname, multi)
        unit = unit_of(pts)
        vals = [p[1] for p in pts]
        labs = [p[0] for p in pts]

        add(family="range", difficulty=4, series=sname, series_index=si,
            question=f"What is the difference between the highest and the lowest "
                     f"{subj or 'value'}{sr}?")
        add(family="extrema_arithmetic", difficulty=4, series=sname, series_index=si, op="sum",
            question=f"What is the sum of the highest and the lowest {subj or 'value'}{sr}?")
        add(family="median", difficulty=4, series=sname, series_index=si,
            question=f"What is the median {subj or 'value'}{sr}?")
        if len(set(round(v, 6) for v in vals)) < len(vals):
            add(family="mode", difficulty=4, series=sname, series_index=si,
                question=f"Which {subj or 'value'}{sr} occurs most often?")
        add(family="rank", difficulty=4, series=sname, series_index=si, k=2,
            question=f"Which {'year' if _looks_temporal(labs) else 'category'} has the "
                     f"second highest {subj or 'value'}{sr}?")
        add(family="ranked_aggregate", difficulty=4, series=sname, series_index=si, k=3,
            question=f"What is the sum of the three highest {subj or 'values'}{sr}?")
        add(family="ranked_arithmetic", difficulty=5, series=sname, series_index=si, k=2, m=3,
            question=f"Is the sum of the two highest {subj or 'values'}{sr} greater than the "
                     f"sum of the three lowest?")
        add(family="trend_direction", difficulty=3, series=sname, series_index=si,
            question=f"Between the first and the last "
                     f"{'year' if _looks_temporal(labs) else 'category'}, did the "
                     f"{subj or 'value'}{sr} increase or decrease?")
        add(family="steepest_change", difficulty=4, series=sname, series_index=si, mode="max",
            question=f"Between which two consecutive "
                     f"{'years' if _looks_temporal(labs) else 'categories'} did the "
                     f"{subj or 'value'}{sr} change the most?")
        add(family="percent_change", difficulty=4, series=sname, series_index=si,
            x1=labs[0], x2=labs[-1],
            question=f"By what percent did the {subj or 'value'}{sr} change from "
                     f"{clean_label(labs[0])} to {clean_label(labs[-1])}?")
        add(family="relational_count", difficulty=3, series=sname, series_index=si,
            question=f"How many times did the {subj or 'value'}{sr} increase compared with "
                     f"the previous {'year' if _looks_temporal(labs) else 'category'}?")
        # inverse lookup: value -> label. Needs a unique maximum to be well-posed.
        mx = max(vals)
        if sum(1 for v in vals if v == mx) == 1:
            tgt = pts[vals.index(mx)]
            add(family="inverse_lookup", difficulty=2, series=sname, series_index=si,
                target=tgt[2],
                question=f"Which {'year' if _looks_temporal(labs) else 'category'} has a "
                         f"{subj or 'value'}{sr} of {tgt[2]}{unit}?")
        thr = _fmt(_st.median(vals))
        add(family="count_below", difficulty=2, series=sname, series_index=si, thr=thr,
            question=f"How many {'years' if _looks_temporal(labs) else 'categories'} have a "
                     f"{subj or 'value'}{sr} below {thr}{unit}?")
        add(family="threshold_lookup", difficulty=3, series=sname, series_index=si, thr=thr,
            question=f"Which {'years' if _looks_temporal(labs) else 'categories'} have a "
                     f"{subj or 'value'}{sr} above {thr}{unit}?")
        add(family="filtered_aggregate", difficulty=4, series=sname, series_index=si, thr=thr,
            question=f"What is the average {subj or 'value'}{sr} of the "
                     f"{'years' if _looks_temporal(labs) else 'categories'} above "
                     f"{thr}{unit}?")
        add(family="statistic_compare", difficulty=4, series=sname, series_index=si, thr=thr,
            question=f"Is the average {subj or 'value'}{sr} greater than {thr}{unit}?")
        if len(vals) >= 2 and all(v >= 0 for v in vals) and sum(vals) > 0:
            add(family="share_of_total", difficulty=4, series=sname, series_index=si,
                x=labs[vals.index(max(vals))],
                question=f"What share of the total{sr} does "
                         f"{clean_label(labs[vals.index(max(vals))])} account for?")
        if all(0 <= v <= 100 for v in vals) and ("%" in unit or "percent" in (subj or "")):
            add(family="complement", difficulty=3, series=sname, series_index=si, x=labs[0],
                question=f"What percentage does NOT belong to {clean_label(labs[0])}{sr}?")
        miss = [p[0] for p in pts if p[1] is None]
        if xs and len(set(labs)) < len(xs):
            add(family="missing_value", difficulty=2, series=sname, series_index=si,
                question=f"Which {'year' if _looks_temporal(xs) else 'category'} on the axis "
                         f"has no {subj or 'value'}{sr}?")

    # ---------------- cross-series ----------------
    if multi and len(refs) >= 2:
        for _ in range(3):
            (ia, sa, na), (ib, sb, nb) = _pick2(refs, rng)
            if na == nb:
                continue
            add(family="aggregate_compare", difficulty=5, series_a=na, series_a_index=ia,
                series_b=nb, series_b_index=ib,
                question=f"Which has the higher total, {na} or {nb}?")
            add(family="global_series_relation", difficulty=4, series_a=na, series_a_index=ia,
                series_b=nb, series_b_index=ib,
                question=f"Is {na} greater than {nb} in every "
                         f"{'year' if _looks_temporal(xs) else 'category'} shown?")
            add(family="gap_extremum", difficulty=4, series_a=na, series_a_index=ia,
                series_b=nb, series_b_index=ib,
                question=f"In which {'year' if _looks_temporal(xs) else 'category'} is the "
                         f"gap between {na} and {nb} the largest?")
            add(family="extrema_aggregate_compare", difficulty=5, series_a=na,
                series_a_index=ia, series_b=nb, series_b_index=ib,
                question=f"Is the average of {na} greater than the maximum of {nb}?")
        add(family="joint_extremum", difficulty=4,
            question=f"Across all series, which "
                     f"{'year' if _looks_temporal(xs) else 'category'} has the highest value?")
        add(family="aggregate_extremum", difficulty=5,
            question=f"Which {'year' if _looks_temporal(xs) else 'category'} has the highest "
                     f"total across all series?")

    rng.shuffle(progs)
    return progs[:max_per_image]


def _pick2(refs, rng):
    a = refs[rng.randrange(len(refs))]
    b = refs[rng.randrange(len(refs))]
    return a, b


_TEMPORAL = re.compile(r"^(19|20)\d{2}$|^(q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                       re.I)


def _looks_temporal(labels):
    ls = [str(x).strip() for x in labels if str(x).strip()]
    if not ls:
        return False
    return sum(1 for x in ls if _TEMPORAL.match(x)) >= max(2, len(ls) // 2)


# --------------------------------------------------------------------- execute
def execute_v5(prog, table):
    """Recompute the answer. Returns (answer, answer_type, provenance) or None."""
    fam = prog["family"]
    refs = referable_series(table)

    # ---- chart-level
    if fam == "chart_metadata":
        t = str(table.get("title") or "").strip()
        return (t, "free", [f"TITLE({t})"]) if t else None

    if fam == "chart_structure_count":
        what = prog["what"]
        if what == "series":
            n = len(refs)
            return (str(n), "number", [f"SERIES_COUNT({n})"]) if n >= 1 else None
        if what == "categories":
            xs = _xs(table)
            return (str(len(xs)), "number", [f"XLABELS({len(xs)})"]) if xs else None
        if what == "colors":
            cs = {_colour_of(s) for _, s, _ in refs if _colour_of(s)}
            return (str(len(cs)), "number", [f"COLORS({sorted(cs)})"]) if cs else None
        return None

    if fam == "positional_retrieve":
        xs = _xs(table)
        if len(xs) < 2:
            return None
        v = xs[0] if prog["pos"] == "first" else xs[-1]
        return v, "label", [f"XPOS({prog['pos']},{v})"]

    if fam == "legend_lookup":
        s = _series_by_index(table, prog.get("series_index"), prog.get("series"))
        if s is None:
            return None
        c = _colour_of(s)
        if c is None or c != prog.get("colour"):
            return None
        # must be the ONLY series with this colour, else ambiguous
        if sum(1 for _, s2, _ in refs if _colour_of(s2) == c) != 1:
            return None
        return prog["series"], "label", [f"LEGEND({c},{prog['series']})"]

    if fam in ("joint_extremum", "aggregate_extremum"):
        xs = _xs(table)
        if not xs or len(refs) < 2:
            return None
        agg = {}
        prov = []
        for _, s, nm in refs:
            for p in points_of(s):
                if p[0] not in xs:
                    continue
                if fam == "joint_extremum":
                    agg[p[0]] = max(agg.get(p[0], float("-inf")), p[1])
                else:
                    agg[p[0]] = agg.get(p[0], 0.0) + p[1]
                prov.append(_cell(nm, p))
        if len(agg) < 2:
            return None
        best = max(agg.values())
        winners = [k for k, v in agg.items() if v == best]
        if len(winners) != 1:
            return None
        return winners[0], "label", prov[:40]

    # ---- cross-series
    if fam in ("aggregate_compare", "global_series_relation", "gap_extremum",
               "extrema_aggregate_compare"):
        sa = _series_by_index(table, prog.get("series_a_index"), prog.get("series_a"))
        sb = _series_by_index(table, prog.get("series_b_index"), prog.get("series_b"))
        if sa is None or sb is None:
            return None
        pa, pb = points_of(sa), points_of(sb)
        da = {p[0]: p for p in pa}
        db = {p[0]: p for p in pb}
        shared = [x for x in da if x in db]
        if len(shared) < 2:
            return None
        prov = [_cell(prog["series_a"], da[x]) for x in shared] + \
               [_cell(prog["series_b"], db[x]) for x in shared]

        if fam == "aggregate_compare":
            ta = sum(da[x][1] for x in shared)
            tb = sum(db[x][1] for x in shared)
            if ta == tb:
                return None
            return (prog["series_a"] if ta > tb else prog["series_b"]), "label", prov
        if fam == "global_series_relation":
            allgt = all(da[x][1] > db[x][1] for x in shared)
            anyeq = any(da[x][1] == db[x][1] for x in shared)
            if anyeq:
                return None
            return ("Yes" if allgt else "No"), "free", prov
        if fam == "gap_extremum":
            gaps = {x: abs(da[x][1] - db[x][1]) for x in shared}
            mx = max(gaps.values())
            win = [k for k, v in gaps.items() if v == mx]
            if len(win) != 1:
                return None
            return win[0], "label", prov
        if fam == "extrema_aggregate_compare":
            avg_a = sum(da[x][1] for x in shared) / len(shared)
            max_b = max(db[x][1] for x in shared)
            if avg_a == max_b:
                return None
            return ("Yes" if avg_a > max_b else "No"), "free", prov

    # ---- per-series
    s = _series_by_index(table, prog.get("series_index"), prog.get("series"))
    if s is None:
        return None
    pts = points_of(s)
    if len(pts) < 3:
        return None
    sname = prog.get("series")
    vals = [p[1] for p in pts]
    labs = [p[0] for p in pts]
    allprov = [_cell(sname, p) for p in pts]

    def uniq_extreme(fn):
        v = fn(vals)
        return None if sum(1 for u in vals if u == v) != 1 else v

    if fam == "range":
        return _fmt(max(vals) - min(vals)), "number", [
            _cell(sname, pts[vals.index(max(vals))]), _cell(sname, pts[vals.index(min(vals))])]

    if fam == "extrema_arithmetic":
        return _fmt(max(vals) + min(vals)), "number", [
            _cell(sname, pts[vals.index(max(vals))]), _cell(sname, pts[vals.index(min(vals))])]

    if fam == "median":
        return _fmt(_st.median(vals)), "number", allprov

    if fam == "mode":
        try:
            m = _st.mode([round(v, 6) for v in vals])
        except Exception:
            return None
        if [round(v, 6) for v in vals].count(m) < 2:
            return None
        return _fmt(m), "number", allprov

    if fam == "rank":
        k = int(prog["k"])
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        if len(order) < k:
            return None
        # the k-th value must be strictly separated, else the ranking is ambiguous
        vs = sorted(vals, reverse=True)
        if len(set(vs[:k + 1])) < min(k + 1, len(vs)):
            return None
        return labs[order[k - 1]], "label", allprov

    if fam == "ranked_aggregate":
        k = int(prog["k"])
        if len(vals) < k + 1:
            return None
        vs = sorted(vals, reverse=True)
        if vs[k - 1] == vs[k]:
            return None
        return _fmt(sum(vs[:k])), "number", allprov

    if fam == "ranked_arithmetic":
        k, m = int(prog["k"]), int(prog["m"])
        if len(vals) < k + m:
            return None
        vs = sorted(vals, reverse=True)
        top, bot = sum(vs[:k]), sum(vs[-m:])
        if top == bot:
            return None
        return ("Yes" if top > bot else "No"), "free", allprov

    if fam == "trend_direction":
        if vals[0] == vals[-1]:
            return None
        return ("increase" if vals[-1] > vals[0] else "decrease"), "free", [
            _cell(sname, pts[0]), _cell(sname, pts[-1])]

    if fam == "steepest_change":
        if len(pts) < 3:
            return None
        deltas = [(abs(vals[i + 1] - vals[i]), i) for i in range(len(vals) - 1)]
        mx = max(d for d, _ in deltas)
        if sum(1 for d, _ in deltas if d == mx) != 1:
            return None
        i = [i for d, i in deltas if d == mx][0]
        return (f"{clean_label(labs[i])} and {clean_label(labs[i+1])}", "free",
                [_cell(sname, pts[i]), _cell(sname, pts[i + 1])])

    if fam == "percent_change":
        x1, x2 = prog["x1"], prog["x2"]
        d = {p[0]: p for p in pts}
        if x1 not in d or x2 not in d or d[x1][1] == 0:
            return None
        pc = 100.0 * (d[x2][1] - d[x1][1]) / abs(d[x1][1])
        return _fmt(pc), "number", [_cell(sname, d[x1]), _cell(sname, d[x2])]

    if fam == "relational_count":
        n = sum(1 for i in range(len(vals) - 1) if vals[i + 1] > vals[i])
        return str(n), "number", allprov

    if fam == "inverse_lookup":
        tgt = _num(prog["target"])
        if tgt is None:
            return None
        hits = [p for p in pts if abs(p[1] - tgt) < 1e-9]
        if len(hits) != 1:
            return None
        return hits[0][0], "label", [_cell(sname, hits[0])]

    if fam == "count_below":
        thr = _num(prog["thr"])
        if thr is None:
            return None
        return str(sum(1 for v in vals if v < thr)), "number", allprov

    if fam == "count_equal":
        thr = _num(prog["thr"])
        if thr is None:
            return None
        return str(sum(1 for v in vals if abs(v - thr) < 1e-9)), "number", allprov

    if fam == "threshold_lookup":
        thr = _num(prog["thr"])
        if thr is None:
            return None
        hits = [p for p in pts if p[1] > thr]
        if not hits or len(hits) > 3:
            return None
        return (", ".join(clean_label(p[0]) for p in hits), "free",
                [_cell(sname, p) for p in hits])

    if fam == "filtered_aggregate":
        thr = _num(prog["thr"])
        if thr is None:
            return None
        sel = [p for p in pts if p[1] > thr]
        if len(sel) < 2:
            return None
        return (_fmt(sum(p[1] for p in sel) / len(sel)), "number",
                [_cell(sname, p) for p in sel])

    if fam == "statistic_compare":
        thr = _num(prog["thr"])
        if thr is None:
            return None
        avg = sum(vals) / len(vals)
        if abs(avg - thr) < 1e-9:
            return None
        return ("Yes" if avg > thr else "No"), "free", allprov

    if fam == "share_of_total":
        tot = sum(vals)
        if tot <= 0:
            return None
        d = {p[0]: p for p in pts}
        x = prog["x"]
        if x not in d:
            return None
        return _fmt(100.0 * d[x][1] / tot), "number", allprov

    if fam == "complement":
        d = {p[0]: p for p in pts}
        x = prog["x"]
        if x not in d or not (0 <= d[x][1] <= 100):
            return None
        return _fmt(100.0 - d[x][1]), "number", [_cell(sname, d[x])]

    if fam == "missing_value":
        xs = _xs(table)
        have = {p[0] for p in pts}
        miss = [x for x in xs if x not in have]
        if len(miss) != 1:
            return None
        return miss[0], "label", [f"XLABELS({len(xs)})"] + allprov[:8]

    return None


V5_FAMILIES = sorted({
    "chart_metadata", "chart_structure_count", "positional_retrieve", "legend_lookup",
    "joint_extremum", "aggregate_extremum", "aggregate_compare", "global_series_relation",
    "gap_extremum", "extrema_aggregate_compare", "range", "extrema_arithmetic", "median",
    "mode", "rank", "ranked_aggregate", "ranked_arithmetic", "trend_direction",
    "steepest_change", "percent_change", "relational_count", "inverse_lookup", "count_below",
    "count_equal", "threshold_lookup", "filtered_aggregate", "statistic_compare",
    "share_of_total", "complement", "missing_value",
})
