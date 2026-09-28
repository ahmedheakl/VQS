"""High-yield, difficulty-graded QA generator over a canonical chart table.

This is the charts analog of scripts/gen_hard_qa.py (which is scene-graph specific).
`src/vqa_engine.gen_charts` only emits ~4 programs per chart (retrieve / extremum /
compare / diff, capped per family); that is far too few to build a large training arm.
Here every readable relation in the table becomes a candidate question.

Two independent halves, on purpose:

  propose(table, rng)   -> programs WITHOUT answers (question + operands only)
  execute(program, table) -> the answer, recomputed from the table

`build()` calls both, so every shipped answer is produced by `execute`, and
`selftest()` re-runs `execute` on the shipped programs to prove reproducibility.
The question text never contains the answer (answer-blinding is structural).

Well-posedness guards carried over from the natural/chart human audits:
  * x-labels must stay distinct under the SCORING normalizer, else two options
    collapse at scoring time;
  * comparisons/extrema need a visible margin (> SEP x range) over the runner-up;
  * a retrieved value must be unique within the numeric tolerance;
  * in a multi-series chart, only series with a unique non-empty name are referable.
"""
import itertools
import re

from .match import normalize

SEP = 0.05          # required visible margin, as a fraction of the series range
TOL = 0.05          # relaxed numeric equality (ChartQA convention)
MAX_PAIRS = 40      # cap on pairwise questions per series (charts can have 30+ points)


# --------------------------------------------------------------------------- helpers
def _num(v):
    """'1,234.5%' -> 1234.5 ; None when not numeric."""
    s = str(v).strip().replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"[^\d.\-+eE]", "", s)
    if not s or s in ("-", "+", ".", "-.", "+."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(v, nd=2):
    """Compact numeric answer. Derived quantities are rounded to `nd` decimals so the
    answer looks like something a person would write off a chart, not 0.6471."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    r = round(v, nd)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:g}"


# ChartQA's own answers are BARE numbers: no "%", no currency, no footnote marker.
# The ChartQA JSON annotation, however, stores plotted values with their unit ("43.54%"),
# and passing that through verbatim taught the student to answer "43.54%" where the
# benchmark expects "43.54" -- the relaxed-accuracy scorer reads "43.54%" as 0.4354 and
# marks it wrong.  Measured cost of that one character: ~10 points of ChartQA.
# Category LABELS are left alone: ChartQA gold keeps their footnote markers
# ("Facebook Messenger*"), so stripping there would break exact match instead.
# Thousand separators too: ChartQA gold writes 5700, never "5 700" or "5,700" (measured:
# zero space- or comma-separated numbers in the 2,500 test answers).
_UNIT_STRIP = re.compile(r"^[\s$€£¥]*(-?[\d,\u00a0 ]*\.?\d+(?:[eE][-+]?\d+)?)\s*%?[\s*†‡]*$")


def normalize_numeric_answer(a):
    """'43.54%' -> '43.54' ; '$1,234' -> '1234' ; '5 700' -> '5700' ;
    'Nigeria*' -> unchanged (labels keep their footnote marker, as ChartQA gold does)."""
    m = _UNIT_STRIP.match(str(a))
    if not m:
        return str(a).strip()
    return re.sub(r"[,\u00a0 ]", "", m.group(1))


def points_of(series):
    """[(x_label, float_value, raw_string)] for the numeric, uniquely-labelled points."""
    pts = []
    for x, y in series.get("points", []):
        v = _num(y)
        if v is None or not label_ok(x):
            continue
        pts.append((str(x), v, str(y)))
    xs = [p[0] for p in pts]
    if len(set(xs)) != len(xs):
        return []
    if len({normalize(x) for x in xs} - {""}) != len(xs):    # collapse under scoring
        return []
    return pts


def referable_series(table):
    """[(index, series, name)] the question text can unambiguously point at."""
    ss = table.get("series", []) or []
    multi = len(ss) > 1
    names = [normalize(str(s.get("name", ""))) for s in ss]
    out = []
    for i, s in enumerate(ss):
        if multi and (not names[i] or names.count(names[i]) != 1):
            continue
        out.append((i, s, str(s.get("name", "") or "")))
    return out


def _sref(name, multi):
    return f" for {name}" if (multi and name) else ""


# ChartQA phrases questions from the chart's SUBJECT, not from the word "value":
#   ChartQA : "What was the population of Romania in 2015?"
#   ours v1 : "What is the value of Romania?"
# A student trained only on the second form never learns to answer the first.  When the
# chart has a title we can recover the subject from it, which covers 44% of ChartQA-train.
# The rest keep the generic wording (and the optional paraphrase pass fixes those).
_TITLE_TAIL = re.compile(r"\s*,\s*(\d{4}(\s*(to|-|–)\s*\d{2,4})?|\d{4}/\d{2,4})\s*\**\s*$")
_BAD_TITLE = re.compile(r"^\s*(figure|table|chart|source|note)\b", re.I)
# A title that is a SENTENCE ("...ratings for the U.S. drop") cannot be slotted into
# "What is the {subject} for X?" -- audit item: '#44 "What is the in Germany, ratings ... drop"'.
# Reject anything with a finite verb, a clause comma, or sentence punctuation.
_TITLE_CLAUSE = re.compile(
    r"\b(is|are|was|were|has|have|had|do|does|did|drop|drops|rise|rises|fell|fall|falls|grew|grow|"
    r"grows|say|says|said|think|thinks|see|sees|remain|remains|continue|continues|expect|expects)\b",
    re.I)


def subject_of(table):
    """A noun phrase naming what the chart measures, or "" when it cannot be trusted."""
    t = str(table.get("title", "") or "").strip()
    for _ in range(3):                     # "…, World, 1958" -> "…"
        t2 = _TITLE_TAIL.sub("", t).strip().rstrip(",;:.")
        t2 = re.sub(r",\s*[A-Z][\w .'-]{1,24}\s*$", "", t2).strip().rstrip(",;:.")
        if t2 == t:
            break
        t = t2
    if not (3 <= len(t) <= 70) or _BAD_TITLE.match(t):
        return ""
    if "\n" in t or "," in t or ";" in t or ":" in t:
        return ""
    if _TITLE_CLAUSE.search(t) or re.search(r"[.!?]", t):
        return ""
    if len(t.split()) > 12:
        return ""
    return t[0].lower() + t[1:] if t[:1].isupper() and not t[:3].isupper() else t


# ---------------------------------------------------------------- label hygiene (audit items)
# "#8 Babies, Kids and" -- a truncated category; "#71 a category named 36" -- an axis tick that
# is really a value.  Both make the question unanswerable, so such points are dropped.
_TRUNC_TAIL = re.compile(r"(\b(and|or|of|the|for|in|to|with|from|by|a|an)\s*|[,&/+-]\s*|\.\.\.|…)$", re.I)
# "#58 Total population chosen as the largest age bracket" -- aggregate rows are not peers of
# the categories they aggregate, so they must never win an extremum/rank/compare.
_AGGREGATE = re.compile(
    r"^\s*(total|all|overall|average|mean|sum|net|combined|aggregate|any|everyone|"
    r"grand total|world|global)\b", re.I)
# "#70 La Senza**", "#85 2024*" -- footnote markers are noise in the QUESTION text.
_FOOTNOTE = re.compile(r"[\s]*[\*†‡]+\s*$")


def label_ok(x):
    """False for truncated or otherwise unusable category labels."""
    x = str(x).strip()
    if len(x) < 1 or len(x) > 60:
        return False
    if _TRUNC_TAIL.search(x):
        return False
    return True


def is_aggregate(x):
    return bool(_AGGREGATE.match(str(x).strip()))


def clean_label(x):
    """Label as it should appear in question TEXT: footnote markers dropped."""
    return _FOOTNOTE.sub("", str(x)).strip() or str(x).strip()


# ------------------------------------------------------------- unit handling (audit item)
# "#5 answers 27.3 instead of 27.3%" / "#63 2.99 instead of $2.99 billion".  The ANSWER must
# stay a bare number -- that is what ChartQA grades against -- so the unit moves into the
# QUESTION instead, which is also how ChartQA itself phrases these.
def unit_of(points):
    raws = [p[2] for p in points]
    if raws and sum(r.strip().endswith("%") for r in raws) > 0.6 * len(raws):
        return "percent"
    if raws and sum(r.strip()[:1] in "$€£¥" for r in raws) > 0.6 * len(raws):
        return "currency"
    return ""


# ------------------------------------------------------- non-additive charts (audit item)
# "#15 adds revenue, EBIT and profit"; "#20 adds quarter-to-quarter % changes as if they
# formed a valid total".  Summing or averaging is only meaningful across mutually exclusive
# parts of one quantity -- never across a time axis, and never over rates/changes.
_TIMEISH = re.compile(r"^\s*((19|20)\d{2}([/-]\d{2,4})?\**|q[1-4]\s*'?\d{0,4}|"
                      r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{0,4})\s*$", re.I)
_RATEISH = re.compile(r"\b(change|growth|rate|yoy|y-o-y|qoq|margin|inflation|per capita|index|"
                      r"share|ratio|percentage point)\b", re.I)


def additive_ok(table, series_name, pts):
    """True when summing/averaging this series is semantically meaningful."""
    xs = [p[0] for p in pts]
    if sum(bool(_TIMEISH.match(x)) for x in xs) > 0.5 * len(xs):
        return False                       # a time axis: totals over years are meaningless
    blob = f"{table.get('title','')} {series_name}"
    if _RATEISH.search(blob):
        return False                       # rates and changes do not add up
    return True


# --------------------------------------------------------------------------- propose
def propose(table, rng, max_per_image=60):
    """Yield answer-free programs. Deterministic given `rng`."""
    progs = []
    refs = referable_series(table)
    multi = len(table.get("series", []) or []) > 1
    subj = subject_of(table)

    for si, s, sname in refs:
        pts = points_of(s)
        if len(pts) < 2:
            continue
        sr = _sref(sname, multi)
        unit = unit_of(pts)
        # aggregate rows ("Total", "All", "World") are not peers of the categories they sum,
        # so they are excluded from every superlative / comparison (audit item #58)
        cpts = [p for p in pts if not is_aggregate(p[0])]
        vals = [p[1] for p in pts]
        vrange = max(vals) - min(vals)
        # The gap two categories must show before a comparison is answerable from pixels.
        # Using only the series RANGE is vacuous on a 2-point series (the range IS the gap),
        # which let through "which is higher, 12.6 or 12.57?".  Scale by the largest plotted
        # magnitude too, since bar charts are drawn from zero.
        margin = SEP * max(vrange, max(abs(v) for v in vals))

        def base(family, difficulty, question, **kw):
            d = {"family": family, "difficulty": difficulty, "question": question,
                 "series": sname, "series_index": si}
            d.update(kw)
            return d

        # ---- L1 retrieve: one per point whose value is unique within tolerance ----
        for x, v, raw in pts:
            if any(p[0] != x and abs(v - p[1]) <= TOL * max(abs(v), 1e-9) for p in pts):
                continue
            xl = clean_label(x)
            subj_is_pct = bool(re.search(r"\b(share|percentage|percent|rate|proportion)\b", subj))
            if subj and unit == "percent" and not subj_is_pct:
                q = f"What percentage of {subj} is {xl}{sr}?"
            elif subj:
                q = f"What is the {subj} for {xl}{sr}?"
            elif unit == "percent":
                q = f"What percentage is shown for {xl}{sr}?"
            elif unit == "currency":
                q = f"What is the amount shown for {xl}{sr}?"
            else:
                q = f"What is the value of {xl}{sr}?"
            progs.append(base("retrieve", 1, q, x=x))

        # ---- L2 extremum + L3 rank-2, only with a visible gap ----
        if len(cpts) >= 3:
            desc = sorted(cpts, key=lambda p: -p[1])
            asc = sorted(cpts, key=lambda p: p[1])
            hi_q = (f"Which category has the highest {subj}{sr}?" if subj
                    else f"Which category has the highest value{sr}?")
            lo_q = (f"Which category has the lowest {subj}{sr}?" if subj
                    else f"Which category has the lowest value{sr}?")
            if desc[0][1] - desc[1][1] > margin:
                progs.append(base("extremum", 2, hi_q, extremum="max"))
            if asc[1][1] - asc[0][1] > margin:
                progs.append(base("extremum", 2, lo_q, extremum="min"))
            if len(desc) >= 3 and desc[1][1] - desc[2][1] > margin and desc[0][1] - desc[1][1] > margin:
                progs.append(base("rank", 3, f"Which category has the second highest value{sr}?",
                                  rank=2, order="desc"))

        # ---- pairwise families ----
        additive = additive_ok(table, sname, pts)
        pairs = list(itertools.combinations(range(len(pts)), 2))
        rng.shuffle(pairs)
        for i, j in pairs[:MAX_PAIRS]:
            a, b = pts[i], pts[j]
            if is_aggregate(a[0]) or is_aggregate(b[0]):
                continue
            if abs(a[1] - b[1]) > margin:
                progs.append(base("compare", 2,
                                  f"Which has a higher value, {clean_label(a[0])} or "
                                  f"{clean_label(b[0])}{sr}?",
                                  x=a[0], x2=b[0]))
                progs.append(base("yesno_gt", 2,
                                  f"Is {clean_label(a[0])} greater than {clean_label(b[0])}{sr}?",
                                  x=a[0], x2=b[0]))
            if a[1] != b[1]:
                hi, lo = (a, b) if a[1] > b[1] else (b, a)
                progs.append(base("diff", 3,
                                  f"How much higher is {clean_label(hi[0])} than "
                                  f"{clean_label(lo[0])}{sr}?",
                                  x=hi[0], x2=lo[0]))
                if additive:
                    progs.append(base("sum", 3,
                                      f"What is the combined value of {clean_label(a[0])} and "
                                      f"{clean_label(b[0])}{sr}?",
                                      x=a[0], x2=b[0]))
            # ratio only when the denominator is safely away from zero
            if abs(b[1]) > 1e-6 and abs(b[1]) > 0.01 * max(abs(v) for v in vals):
                progs.append(base("ratio", 3,
                                  f"What is the ratio of {clean_label(a[0])} to "
                                  f"{clean_label(b[0])}{sr}?",
                                  x=a[0], x2=b[0]))

        # ---- whole-series aggregates ----
        if additive and 2 <= len(pts) <= 12:
            progs.append(base("total", 3, f"What is the sum of all the values{sr}?"))
            progs.append(base("average", 3, f"What is the average of all the values{sr}?"))
        if len(pts) >= 3:
            mid = sorted(vals)[len(vals) // 2]
            if any(v > mid for v in vals) and any(v <= mid for v in vals):
                progs.append(base("count_above", 2,
                                  f"How many categories have a value greater than {_fmt(mid)}{sr}?",
                                  thr=mid))

        # ---- MCQ over the extremum (4 options, distinct labels) ----
        if len(cpts) >= 4:
            desc = sorted(cpts, key=lambda p: -p[1])
            if desc[0][1] - desc[1][1] > margin:
                distract = [p[0] for p in desc[1:]]
                rng.shuffle(distract)
                opts = [desc[0][0]] + distract[:3]
                if len({normalize(o) for o in opts}) == 4:
                    rng.shuffle(opts)
                    txt = "  ".join(f"({chr(97 + k)}) {o}" for k, o in enumerate(opts))
                    progs.append(base("mcq_extremum", 2,
                                      f"Which category has the highest value{sr}?  {txt}",
                                      extremum="max", options=opts))

    # ---- cross-series questions (multi-series charts only) ----
    if multi and len(refs) >= 2:
        common = None
        for _, s, _ in refs:
            xs = {p[0] for p in points_of(s)}
            common = xs if common is None else (common & xs)
        for x in sorted(common or [])[:10]:
            for (i1, s1, n1), (i2, s2, n2) in itertools.combinations(refs, 2):
                v1 = dict((p[0], p[1]) for p in points_of(s1)).get(x)
                v2 = dict((p[0], p[1]) for p in points_of(s2)).get(x)
                if v1 is None or v2 is None or v1 == v2:
                    continue
                scale = max(abs(v1), abs(v2), 1e-9)
                if abs(v1 - v2) <= SEP * scale:          # not visibly different
                    continue
                progs.append({"family": "series_compare", "difficulty": 3,
                              "question": f"In {x}, which is higher, {n1} or {n2}?",
                              "x": x, "series_a": n1, "series_a_index": i1,
                              "series_b": n2, "series_b_index": i2})

    rng.shuffle(progs)
    return progs[:max_per_image]


# --------------------------------------------------------------------------- execute
def _series_by_index(table, idx, name):
    ss = table.get("series", []) or []
    if idx is None or not (0 <= idx < len(ss)):
        return None
    s = ss[idx]
    if str(s.get("name", "") or "") != str(name):
        return None
    return s


def execute(prog, table):
    """Recompute the answer from the table. Returns (answer_str, answer_type, provenance)
    or None when the program is not well-posed on this table."""
    fam = prog["family"]

    if fam == "series_compare":
        sa = _series_by_index(table, prog.get("series_a_index"), prog.get("series_a"))
        sb = _series_by_index(table, prog.get("series_b_index"), prog.get("series_b"))
        if sa is None or sb is None:
            return None
        da = dict((p[0], p) for p in points_of(sa))
        db = dict((p[0], p) for p in points_of(sb))
        x = prog["x"]
        if x not in da or x not in db or da[x][1] == db[x][1]:
            return None
        hi = prog["series_a"] if da[x][1] > db[x][1] else prog["series_b"]
        return hi, "label", [f"CELL({prog['series_a']},{x},{da[x][2]})",
                             f"CELL({prog['series_b']},{x},{db[x][2]})"]

    s = _series_by_index(table, prog.get("series_index"), prog.get("series"))
    if s is None:
        return None
    pts = points_of(s)
    if len(pts) < 2:
        return None
    xmap = {p[0]: p for p in pts}
    vals = [p[1] for p in pts]
    sname = prog.get("series", "")

    def cell(p):
        return f"CELL({sname},{p[0]},{p[2]})"

    if fam == "retrieve":
        p = xmap.get(prog["x"])
        return (normalize_numeric_answer(p[2]), "number", [cell(p)]) if p else None

    if fam in ("extremum", "mcq_extremum"):
        if len(pts) < 3:
            return None
        sel = max if prog.get("extremum") == "max" else min
        t = sel(pts, key=lambda p: p[1])
        if sum(1 for p in pts if p[1] == t[1]) != 1:
            return None
        if fam == "mcq_extremum":
            opts = prog.get("options") or []
            if t[0] not in opts:
                return None
            return chr(97 + opts.index(t[0])), "letter", [cell(t)]
        return t[0], "label", [cell(t)]

    if fam == "rank":
        order = sorted(pts, key=lambda p: -p[1] if prog.get("order") == "desc" else p[1])
        k = int(prog.get("rank", 2)) - 1
        if not (0 <= k < len(order)):
            return None
        if sum(1 for p in pts if p[1] == order[k][1]) != 1:
            return None
        return order[k][0], "label", [cell(order[k])]

    if fam in ("compare", "yesno_gt", "diff", "sum", "ratio"):
        a, b = xmap.get(prog["x"]), xmap.get(prog["x2"])
        if not (a and b):
            return None
        prov = [cell(a), cell(b)]
        if fam == "compare":
            if a[1] == b[1]:
                return None
            return (a[0] if a[1] > b[1] else b[0]), "label", prov
        if fam == "yesno_gt":
            if a[1] == b[1]:
                return None
            return ("yes" if a[1] > b[1] else "no"), "yesno", prov
        if fam == "diff":
            if a[1] == b[1]:
                return None
            return _fmt(abs(a[1] - b[1])), "number", prov
        if fam == "sum":
            return _fmt(a[1] + b[1]), "number", prov
        if fam == "ratio":
            if abs(b[1]) < 1e-6:
                return None
            return _fmt(a[1] / b[1]), "number", prov

    if fam == "total":
        return _fmt(sum(vals)), "number", [cell(p) for p in pts]
    if fam == "average":
        return _fmt(sum(vals) / len(vals)), "number", [cell(p) for p in pts]
    if fam == "count_above":
        thr = float(prog["thr"])
        return str(sum(1 for v in vals if v > thr)), "number", [cell(p) for p in pts]

    return None


# --------------------------------------------------------------------------- build
BANNED = re.compile(r"how big|what size|how large|how small|how tall|how long", re.I)


def build(table, rng, max_per_image=60):
    """[(program, question, answer, answer_type, difficulty, family, provenance)] as dicts."""
    rows = []
    seen = set()
    for prog in propose(table, rng, max_per_image=max_per_image):
        if BANNED.search(prog["question"]):
            continue
        res = execute(prog, table)
        if res is None:
            continue
        ans, atype, prov = res
        if ans is None or str(ans).strip() == "":
            continue
        q = prog["question"]
        if q in seen:
            continue
        seen.add(q)
        rows.append({"question": q, "answer": str(ans), "answer_type": atype,
                     "difficulty": prog["difficulty"], "family": prog["family"],
                     "program": prog, "provenance": prov})
    return rows


def selftest(table, rows):
    """Re-derive every shipped answer; returns (n_ok, n_total, mismatches)."""
    bad = []
    for r in rows:
        res = execute(r["program"], table)
        if res is None or str(res[0]) != str(r["answer"]):
            bad.append((r["question"], r["answer"], None if res is None else res[0]))
    return len(rows) - len(bad), len(rows), bad
