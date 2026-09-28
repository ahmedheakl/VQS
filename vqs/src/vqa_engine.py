"""Verifiable VQA engine (Regime A: gold graphs -> answers correct by construction).

This is the executable QA engine the project was missing. A *trusted* program sampler draws
functional programs over a GOLD scene graph (GQA object graph / ChartQA table); the engine
executes each program to COMPUTE the answer. The answer is therefore correct by construction
and the generator (model) never picks it -- "answer-blinding" is automatic in this regime.

Every generated item carries a serialisable `program` dict; `recompute(program, graph)`
re-derives the answer from the graph independently, so `scripts/gen_vqa.py --selftest` can
prove 100% of stored answers are reproducible (the correct-by-construction audit).

Design choices that keep answers airtight despite GQA's sparse/incomplete annotation:
  * NATURAL questions only query facts that are *present and unique* in the gold graph
    (a uniquely-named object's single typed attribute, or a uniquely-referenced relation
    target). We never ask counts or negative-existence (GQA may miss instances).
  * CHART questions are computed from the gold data table (296/298 charts are all-numeric).

Difficulty ladder (validated downstream by a monotone accuracy drop, CLEVR/GQA-style):
  natural: 1 typed-attribute | 2 relation-query | 3 nested-reference relation-query
  charts : 1 value-retrieve  | 2 extremum/compare | 3 arithmetic difference

Everything here is pure Python over already-loaded gold graphs -> microseconds per image.
"""
import random
from .match import parse_num, value_eq
from .gt_natural import SPATIAL_PREDS

# ---------------- attribute type vocabularies (GQA attributes are a flat list) ----------------
COLORS = {"white", "black", "green", "blue", "brown", "gray", "grey", "red", "yellow",
          "orange", "silver", "gold", "pink", "purple", "tan", "beige", "maroon", "navy",
          "cream", "turquoise", "violet"}
MATERIALS = {"wood", "wooden", "metal", "metallic", "plastic", "glass", "leather", "brick",
             "concrete", "stone", "ceramic", "paper", "cardboard", "cloth", "fabric",
             "rubber", "steel", "porcelain", "marble", "granite", "denim"}
SIZES = {"large", "small", "big", "little", "tall", "short", "long", "huge", "tiny", "giant"}
ATTR_TYPES = {"color": COLORS, "material": MATERIALS, "size": SIZES}

# relation predicates that read naturally as "What is the {subj} {pred}?" (answer = object).
# FUNCTIONAL / interaction verbs ONLY: these take a specific object that GQA annotates fairly
# completely, so a unique gold target is genuinely the answer. Locative prepositions
# (on/in/near/under/above/below/behind/next to/...) are DELIBERATELY EXCLUDED -- they have many
# true targets that GQA annotates sparsely, so they are not well-posed and cause false negatives
# (design-workflow audit, high-severity finding #3).
REL_PREDS = {
    "wearing", "holding", "riding", "eating", "carrying", "using", "watching", "catching",
    "throwing", "pushing", "pulling", "playing with", "covering", "hitting", "kicking",
    "flying", "driving", "reading", "drinking", "cutting", "petting", "feeding",
}
# a distinguishing attribute for nested reference is a single, unambiguous descriptor
NESTED_ATTR_OK = COLORS | MATERIALS | SIZES


# ---------------- referent well-posedness (driven by the HUMAN AUDIT, which found the two bugs
# the programmatic checks could not see) ----------------
# (a) PLURAL names break the uniqueness guarantee: GQA stores "women"/"umbrellas" as ONE node, so
#     name_count==1 passes while the image actually shows several -> "the women" is ambiguous.
# (b) TINY referents are unreadable ("what colour is the tennis ball" when the ball is a few pixels).
IRREGULAR_PLURALS = {"women", "men", "people", "children", "teeth", "feet", "mice", "geese"}
MIN_REF_AREA = 0.01          # a referenced object must cover >=1% of the image to be readable


def _is_plural(name):
    from .match import _singular
    n = str(name).strip().lower()
    if n in IRREGULAR_PLURALS:
        return True
    last = n.split()[-1] if n.split() else n
    return _singular(last) != last          # 'umbrellas'->'umbrella'; 'grass'->'grass' (safe)


def _area_frac(o, W, H):
    """Fraction of the image the object covers, or None when geometry is UNKNOWN (predicted
    graphs carry no boxes: w=h=0 and width/height=None -> we must not reject on size)."""
    try:
        w, h = int(o.get("w", 0)), int(o.get("h", 0))
        Wi, Hi = int(W or 0), int(H or 0)
        if w * h <= 0 or Wi * Hi <= 1:
            return None
        return (w * h) / (Wi * Hi)
    except Exception:
        return None


def _good_referent(o, name, W, H):
    """A referent we may point at with 'the {name}': singular AND (when geometry is known)
    big enough to read. Size test skipped when the graph has no boxes (predicted graphs)."""
    if not name or _is_plural(name):
        return False
    af = _area_frac(o, W, H)
    return True if af is None else af >= MIN_REF_AREA


# ============================================================ NATURAL ============================================================
def _norm(s):
    return str(s).strip().lower()


def _typed_attr(attrs, atype):
    """Return the single attribute of `attrs` whose value is of type `atype`, or None if
    zero or >1 (ambiguous -> not well-posed)."""
    vocab = ATTR_TYPES[atype]
    hits = [a for a in attrs if _norm(a) in vocab]
    # size class 'grey'/'gray' both count as one color; dedupe by normalized value
    hits = list(dict.fromkeys(_norm(a) for a in hits))
    return hits[0] if len(hits) == 1 else None


def _natural_index(entry):
    """Build lookup structures over a raw GQA entry."""
    objs = entry.get("objects", {}) or {}
    id2name = {oid: o.get("name", "") for oid, o in objs.items()}
    from collections import Counter
    name_count = Counter(n for n in id2name.values() if n)
    return objs, id2name, name_count


def _sem_rels(o, id2name):
    """Semantic (non-spatial, non-part-of) relations of an object -> list of (pred, target_name)."""
    out = []
    for r in o.get("relations", []) or []:
        pred = _norm(r.get("name", ""))
        tgt = id2name.get(r.get("object", ""), "")
        if pred and tgt and pred not in SPATIAL_PREDS and pred != "of" and pred in REL_PREDS:
            out.append((pred, tgt))
    return out


def _sem_rels_ids(o, id2name):
    """Same as _sem_rels but keeps the target ID -> (pred, target_id, target_name). Needed for the
    2-hop rel_attr family, which must read the TARGET INSTANCE's own attributes."""
    out = []
    for r in o.get("relations", []) or []:
        pred = _norm(r.get("name", ""))
        tid = r.get("object", "")
        tgt = id2name.get(tid, "")
        if pred and tgt and pred not in SPATIAL_PREDS and pred != "of" and pred in REL_PREDS:
            out.append((pred, tid, tgt))
    return out


def gen_natural(entry, rng, per_family=2):
    """Yield candidate programs for one GQA entry. Each is a dict ready to serialise.

    Every referent must be SINGULAR and large enough to read (_good_referent) -- the human audit
    showed plural GQA nodes ('the women') and few-pixel targets ('the tennis ball') produce
    ambiguous / unanswerable questions that the uniqueness check alone does not catch."""
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    out = []

    # ---- L1: typed-attribute of a UNIQUELY-named object with exactly one attr of that type ----
    attr_cands = []
    for oid, o in objs.items():
        name = o.get("name", "")
        if not name or name_count[name] != 1 or not _good_referent(o, name, W, H):
            continue
        attrs = o.get("attributes", []) or []
        for atype in ATTR_TYPES:
            val = _typed_attr(attrs, atype)
            if val:
                attr_cands.append({
                    "domain": "natural", "family": "attr", "difficulty": 1,
                    "subj": name, "attr_type": atype, "answer": val, "answer_type": "word",
                    "acceptable": [val],
                    "question": f"What {atype} is the {name}?",
                    "provenance": [f"ATTR({name},{atype},{val})"],
                })
    rng.shuffle(attr_cands)
    out += attr_cands[:per_family]

    # ---- L2: relation-query on a UNIQUELY-named subject ----
    rel_cands = []
    for oid, o in objs.items():
        name = o.get("name", "")
        if not name or name_count[name] != 1 or not _good_referent(o, name, W, H):
            continue
        # group targets by predicate; accept ANY target under that predicate (well-posed as "name one").
        # The TARGET is the answer -> it too must be singular + readable.
        by_pred = {}
        for pred, tid, tgt in _sem_rels_ids(o, id2name):
            if _norm(tgt) != _norm(name) and _good_referent(objs.get(tid, {}) or {}, tgt, W, H):
                by_pred.setdefault(pred, []).append(tgt)
        for pred, tgts in by_pred.items():
            acc = list(dict.fromkeys(tgts))
            # functional predicate (locatives already excluded); ACCEPT ANY gold target -> well-posed
            # as "name one thing the {subj} is {pred}". All acc entries are scored as correct.
            rel_cands.append({
                "domain": "natural", "family": "rel", "difficulty": 2,
                "subj": name, "pred": pred, "answer": acc[0], "answer_type": "word",
                "acceptable": acc, "unique": len(acc) == 1,
                "question": f"What is the {name} {pred}?",
                "provenance": [f"REL({name},{pred},{acc[0]})"],
            })
    rng.shuffle(rel_cands)
    out += rel_cands[:per_family]

    # ---- L3: nested reference -- subject NOT unique, disambiguated by a distinguishing attribute ----
    nested_cands = []
    for oid, o in objs.items():
        name = o.get("name", "")
        if not name or name_count[name] < 2:          # need ambiguity to justify disambiguation
            continue
        af = _area_frac(o, W, H)
        if _is_plural(name) or (af is not None and af < MIN_REF_AREA):
            continue
        # a distinguishing attribute that makes (attr, name) unique among objects of this name
        attrs = [_norm(a) for a in (o.get("attributes", []) or []) if _norm(a) in NESTED_ATTR_OK]
        rels = _sem_rels_ids(o, id2name)
        if not attrs or not rels:
            continue
        for attr in attrs:
            # is (attr, name) unique in the image?
            n_match = sum(1 for oid2, o2 in objs.items()
                          if o2.get("name", "") == name
                          and attr in {_norm(a) for a in (o2.get("attributes", []) or [])})
            if n_match != 1:
                continue
            by_pred = {}
            for pred, tid, tgt in rels:
                # The relation target is the answer, so apply the same singular/readability
                # requirement used by L2 relations and by _recompute_natural.  Older scale
                # generation omitted this check, producing rel_nested rows that immediately
                # failed independent re-execution when the target was plural or too small.
                if _norm(tgt) != _norm(name) \
                        and _good_referent(objs.get(tid, {}) or {}, tgt, W, H):
                    by_pred.setdefault(pred, []).append(tgt)
            for pred, tgts in by_pred.items():
                acc = list(dict.fromkeys(tgts))
                if len(acc) != 1:                      # require a UNIQUE gold target -> well-posed
                    continue
                nested_cands.append({
                    "domain": "natural", "family": "rel_nested", "difficulty": 3,
                    "subj": name, "disambig_attr": attr, "pred": pred,
                    "answer": acc[0], "answer_type": "word", "acceptable": acc,
                    "unique": True,
                    "question": f"What is the {attr} {name} {pred}?",
                    "provenance": [f"REL(({attr} {name}),{pred},{acc[0]})"],
                })
            break                                      # one distinguishing attr per object is enough
    rng.shuffle(nested_cands)
    out += nested_cands[:per_family]

    # ---- L3: 2-hop relation->attribute ("What color is the frisbee that the man is holding?") ----
    # hop 1: resolve the target via a UNIQUE functional relation from a uniquely-named subject
    # hop 2: read that TARGET INSTANCE's single typed attribute. Genuinely compositional and far
    # more common than rel_nested (which needs an ambiguous subject + a distinguishing attribute).
    relattr_cands = []
    for oid, o in objs.items():
        name = o.get("name", "")
        if not name or name_count[name] != 1 or not _good_referent(o, name, W, H):
            continue
        by_pred = {}
        for pred, tid, tname in _sem_rels_ids(o, id2name):
            # the TARGET's attribute is the answer -> the target must be singular + readable
            if _norm(tname) != _norm(name) and _good_referent(objs.get(tid, {}) or {}, tname, W, H):
                by_pred.setdefault(pred, []).append((tid, tname))
        for pred, tgts in by_pred.items():
            if len({t[0] for t in tgts}) != 1:          # unique target -> the reference is unambiguous
                continue
            tid, tname = tgts[0]
            tattrs = (objs.get(tid, {}) or {}).get("attributes", []) or []
            for atype in ATTR_TYPES:
                val = _typed_attr(tattrs, atype)
                if val:
                    relattr_cands.append({
                        "domain": "natural", "family": "rel_attr", "difficulty": 3,
                        "subj": name, "pred": pred, "tgt": tname, "attr_type": atype,
                        "answer": val, "answer_type": "word", "acceptable": [val],
                        "question": f"What {atype} is the {tname} that the {name} is {pred}?",
                        "provenance": [f"REL({name},{pred},{tname})", f"ATTR({tname},{atype},{val})"],
                    })
    rng.shuffle(relattr_cands)
    out += relattr_cands[:per_family]
    return out


def _recompute_natural(prog, entry):
    """Independently re-derive the acceptable answer set from the gold entry.
    Mirrors gen_natural's referent filters (singular + readable) so the audit compares like-for-like."""
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    fam = prog["family"]
    if fam == "attr":
        name = prog["subj"]
        if name_count.get(name) != 1:
            return None
        for oid, o in objs.items():
            if o.get("name") == name:
                if not _good_referent(o, name, W, H):
                    return None
                value = _typed_attr(o.get("attributes", []) or [], prog["attr_type"])
                return [value] if value else None
        return None
    if fam in ("rel", "rel_nested"):
        name = prog["subj"]
        want_attr = prog.get("disambig_attr")
        if fam == "rel" and name_count.get(name) != 1:
            return None
        if fam == "rel_nested" and (name_count.get(name, 0) < 2 or not want_attr):
            return None
        candidates = []
        for oid, o in objs.items():
            if o.get("name") != name:
                continue
            if want_attr and want_attr not in {_norm(a) for a in (o.get("attributes", []) or [])}:
                continue
            candidates.append((oid, o))
        # A plain relation must have a unique named subject.  A nested relation must have
        # exactly one subject after applying its distinguishing attribute.
        if len(candidates) != 1:
            return None
        _, subject = candidates[0]
        if not _good_referent(subject, name, W, H):
            return None
        acc = []
        for pred, tid, tgt in _sem_rels_ids(subject, id2name):
            if pred == prog["pred"] and _norm(tgt) != _norm(name) \
                    and _good_referent(objs.get(tid, {}) or {}, tgt, W, H):
                acc.append(tgt)
        acc = list(dict.fromkeys(acc))
        if fam == "rel_nested" and len(acc) != 1:
            return None
        return acc or None
    if fam == "rel_attr":
        name = prog["subj"]
        if name_count.get(name) != 1:
            return None
        for oid, o in objs.items():
            if o.get("name") != name:
                continue
            if not _good_referent(o, name, W, H):
                return None
            tgts = [(tid, tn) for p, tid, tn in _sem_rels_ids(o, id2name)
                    if p == prog["pred"] and _norm(tn) != _norm(name)
                    and _good_referent(objs.get(tid, {}) or {}, tn, W, H)]
            if len({t[0] for t in tgts}) != 1:
                return None
            tid, tn = tgts[0]
            if _norm(tn) != _norm(prog["tgt"]):
                return None
            v = _typed_attr((objs.get(tid, {}) or {}).get("attributes", []) or [], prog["attr_type"])
            return [v] if v else None
        return None
    return None


# ============================================================ CHARTS ============================================================
import re as _re
_NUM_SPAN = _re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _numeric_points(series):
    """[(x, y_float, y_raw)] for a canon series, numeric only.

    parse_num matches only the FIRST numeric span, so a cell like '990 372' (space thousands)
    would silently become 990.0 -- a corrupt gold value. We therefore REJECT any cell that still
    has digits left after removing the first matched number (audit finding #5)."""
    out = []
    for x, y in series["points"]:
        yr = str(y)
        v = parse_num(yr)
        if v is None:
            continue
        leftover = _NUM_SPAN.sub("", yr, count=1)      # drop the span parse_num actually used
        if _re.search(r"\d", leftover):                 # residual digits -> ambiguous/corrupt -> skip
            continue
        out.append((str(x), v, yr))
    return out


CHART_TOL = 0.05        # ChartQA numeric convention (also the scoring tolerance)
SEP = 0.03              # min separation (fraction of value range) for a distinguishable extremum/compare


def gen_charts(canon, rng, per_family=2):
    """Yield candidate programs for one canonical chart. Well-posedness guards (audit #2/#6):
      * x-labels must be distinct under the SCORING normalizer, so '2016' vs '2016*' never become
        two options that collapse at scoring time;
      * retrieve: the target value must be unique within the numeric tolerance (else a mis-read of
        a near-equal bar would score correct);
      * extremum/compare: the winner must beat the runner-up by a visible margin (>SEP*range), else
        the ranking is not actually readable from the pixels."""
    from .match import normalize
    out = []
    multi = len(canon["series"]) > 1

    # A series name is part of the natural-language referent.  Duplicate (or blank) names in a
    # multi-series chart cannot identify which trace is queried.  Keep only series whose names
    # are unique under the same normalizer used by scoring.  The stable index is serialized as
    # an execution identity as well; names remain the human-facing identity.
    series_refs = []
    normalized_names = [normalize(str(s.get("name", ""))) for s in canon["series"]]
    for series_index, s in enumerate(canon["series"]):
        normalized_name = normalized_names[series_index]
        if multi and (not normalized_name or normalized_names.count(normalized_name) != 1):
            continue
        series_refs.append((series_index, s))

    def sref(sname):
        return f" for {sname}" if (multi and sname) else ""

    for series_index, s in series_refs:
        pts = _numeric_points(s)
        if len(pts) < 2:
            continue
        sname = s.get("name", "")
        xs = [p[0] for p in pts]
        if len(set(xs)) != len(xs):                        # duplicate x-labels (raw)
            continue
        if len({normalize(x) for x in xs}) != len(xs):     # collapse under the scoring normalizer
            continue
        vals = [p[1] for p in pts]
        vrange = max(vals) - min(vals)
        margin = SEP * vrange if vrange > 0 else 0.0

        # ---- L1: value retrieve (target value unique within tolerance) ----
        x, v, yraw = rng.choice(pts)
        if not any(p2[0] != x and abs(v - p2[1]) <= CHART_TOL * max(abs(v), 1e-9) for p2 in pts):
            out.append({
                "domain": "charts", "family": "retrieve", "difficulty": 1,
                "series": sname, "series_index": series_index,
                "x": x, "answer": yraw, "answer_type": "number",
                "acceptable": [yraw],
                "question": f"What is the value of {x}{sref(sname)}?",
                "provenance": [f"CELL({sname},{x},{yraw})"],
            })

        # ---- L2: extremum (unique argmax/argmin with a visible runner-up margin) ----
        if len(pts) >= 3:
            sdesc, sasc = sorted(vals, reverse=True), sorted(vals)
            for kind, sel, gap in (("highest", max, sdesc[0] - sdesc[1]),
                                   ("lowest", min, sasc[1] - sasc[0])):
                target = sel(pts, key=lambda p: p[1])
                if gap > margin and sum(1 for p in pts if p[1] == target[1]) == 1:
                    out.append({
                        "domain": "charts", "family": "extremum", "difficulty": 2,
                        "series": sname, "series_index": series_index,
                        "extremum": ("max" if kind == "highest" else "min"),
                        "answer": target[0], "answer_type": "label", "acceptable": [target[0]],
                        "question": f"Which category has the {kind} value{sref(sname)}?",
                        "provenance": [f"CELL({sname},{target[0]},{target[2]})"],
                    })

        # ---- L2: compare two points with a visible margin ----
        if len(pts) >= 2:
            a, b = rng.sample(pts, 2)
            if abs(a[1] - b[1]) > margin:
                hi = a if a[1] > b[1] else b
                out.append({
                    "domain": "charts", "family": "compare", "difficulty": 2,
                    "series": sname, "series_index": series_index, "x": a[0], "x2": b[0],
                    "answer": hi[0], "answer_type": "label", "acceptable": [hi[0]],
                    "question": f"Which has a higher value, {a[0]} or {b[0]}{sref(sname)}?",
                    "provenance": [f"CELL({sname},{a[0]},{a[2]})", f"CELL({sname},{b[0]},{b[2]})"],
                })

        # ---- L3: arithmetic difference (distinct values) ----
        if len(pts) >= 2:
            a, b = rng.sample(pts, 2)
            if a[1] != b[1]:
                dstr = _fmt_num(round(abs(a[1] - b[1]), 4))
                out.append({
                    "domain": "charts", "family": "diff", "difficulty": 3,
                    "series": sname, "series_index": series_index, "x": a[0], "x2": b[0],
                    "answer": dstr, "answer_type": "number", "acceptable": [dstr],
                    "question": f"What is the difference between the values of {a[0]} and {b[0]}{sref(sname)}?",
                    "provenance": [f"CELL({sname},{a[0]},{a[2]})", f"CELL({sname},{b[0]},{b[2]})"],
                })

    # cap per family per chart for balance
    by_fam = {}
    rng.shuffle(out)
    kept = []
    for p in out:
        k = p["family"]
        if by_fam.get(k, 0) < per_family:
            by_fam[k] = by_fam.get(k, 0) + 1
            kept.append(p)
    return kept


def _fmt_num(v):
    """Compact numeric string: drop trailing .0 for integers."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _recompute_charts(prog, canon):
    from .match import normalize
    sname = prog.get("series", "")
    all_series = canon.get("series", []) or []
    multi = len(all_series) > 1
    normalized_names = [normalize(str(s.get("name", ""))) for s in all_series]

    # Re-execution must never resolve an ambiguous natural-language series reference by silently
    # taking the first match.  New programs also carry a stable index, which is validated against
    # the stored human-facing name.  Legacy programs remain executable only when their name is
    # unique (a single-series chart is unique by construction).
    valid_refs = [
        (i, s) for i, s in enumerate(all_series)
        if not multi or (normalized_names[i] and normalized_names.count(normalized_names[i]) == 1)
    ]
    series_index = prog.get("series_index")
    if series_index is not None:
        try:
            series_index = int(series_index)
        except (TypeError, ValueError):
            return None
        matches = [(i, s) for i, s in valid_refs if i == series_index]
        if len(matches) != 1 or str(matches[0][1].get("name", "")) != str(sname):
            return None
        series = matches[0][1]
    else:
        matches = [(i, s) for i, s in valid_refs if str(s.get("name", "")) == str(sname)]
        if not matches and len(valid_refs) == 1 and not str(sname):
            matches = valid_refs
        if len(matches) != 1:
            return None
        series = matches[0][1]

    if series is None:
        return None
    pts = _numeric_points(series)
    xmap = {p[0]: p for p in pts}
    fam = prog["family"]
    if fam == "retrieve":
        p = xmap.get(prog["x"])
        return [p[2]] if p else None
    if fam == "extremum":
        if len(pts) < 3:
            return None
        sel = max if prog["extremum"] == "max" else min
        target = sel(pts, key=lambda p: p[1])
        ties = [p for p in pts if p[1] == target[1]]
        return [target[0]] if len(ties) == 1 else None
    if fam == "compare":
        a, b = xmap.get(prog["x"]), xmap.get(prog["x2"])
        if not (a and b) or a[1] == b[1]:
            return None
        return [a[0] if a[1] > b[1] else b[0]]
    if fam == "diff":
        a, b = xmap.get(prog["x"]), xmap.get(prog["x2"])
        if not (a and b):
            return None
        return [_fmt_num(round(abs(a[1] - b[1]), 4))]
    return None


# ============================================================ DIAGRAMS ============================================================
# Synthetic diagrams (vqa-eval Phase 1) ship EXACT sidecar graphs:
#   {diagram_type, entities:[{name, attributes:{label:{value},color:{value},shape:{value},...}}],
#    relations:[{subject,predicate,object,...}]}
# Graphs are correct BY CONSTRUCTION (rendered from the graph), so counts/degree ARE well-posed
# here (unlike GQA, whose annotation is incomplete).
NOUN_BY_TYPE = {"flowchart": "node", "network": "node", "mind_map": "node", "tree": "node",
                "er": "entity", "venn": "set", "circle_packing": "circle"}


def _dg_attr(e, key):
    v = (e.get("attributes", {}) or {}).get(key, {})
    return str(v.get("value")) if isinstance(v, dict) and v.get("value") is not None else None


def _dg_index(graph):
    # Synthetic-diagram sidecars append a graph-level metadata record named
    # ``graph`` to the entity array.  It carries num_nodes/num_edges but is not a
    # rendered node.  Counting the raw array therefore made every total-node QA
    # off by one while the old self-test repeated the same mistake.  Remove only
    # the recognizable metadata record; a legitimately rendered entity called
    # "graph" (with a label and no graph-level counters) remains valid.
    raw_ents = graph.get("entities", []) or []
    def is_graph_metadata(entity):
        attrs = (entity.get("attributes", {}) or {}) if isinstance(entity, dict) else {}
        return (entity.get("name") == "graph" and not _dg_attr(entity, "label") and
                ("num_nodes" in attrs or "num_edges" in attrs))
    ents = [e for e in raw_ents if not is_graph_metadata(e)]
    rels = graph.get("relations", []) or []
    label_of = {e["name"]: _dg_attr(e, "label") for e in ents}
    labels = [l for l in label_of.values() if l]
    from collections import Counter
    lab_count = Counter(labels)
    return ents, rels, label_of, lab_count


def gen_diagrams(graph, rng, per_family=2):
    dtype = graph.get("diagram_type", "diagram")
    noun = NOUN_BY_TYPE.get(dtype, "node")
    ents, rels, label_of, lab_count = _dg_index(graph)
    out = []

    # ---- L1 d_count: total elements, or by colour (exact graph -> counts are safe) ----
    cands = [{"domain": "diagrams", "family": "d_count", "difficulty": 1, "what": "total",
              "answer": _fmt_num(len(ents)), "answer_type": "number",
              "acceptable": [_fmt_num(len(ents))],
              "question": f"How many {noun}s are in this {dtype.replace('_', ' ')}?",
              "provenance": [f"COUNT(*,{len(ents)})"]}]
    from collections import Counter
    col_count = Counter(c for c in (_dg_attr(e, "color") for e in ents) if c)
    for col, k in col_count.items():
        cands.append({"domain": "diagrams", "family": "d_count", "difficulty": 1, "what": f"color:{col}",
                      "answer": _fmt_num(k), "answer_type": "number", "acceptable": [_fmt_num(k)],
                      "question": f"How many {col} {noun}s are in the diagram?",
                      "provenance": [f"COUNT(color={col},{k})"]})
    rng.shuffle(cands)
    out += cands[:per_family]

    # ---- L1 d_attr: colour/shape of a uniquely-labeled element ----
    attr_cands = []
    for e in ents:
        lab = _dg_attr(e, "label")
        if not lab or lab_count[lab] != 1:
            continue
        for key in ("color", "shape"):
            val = _dg_attr(e, key)
            if val:
                attr_cands.append({"domain": "diagrams", "family": "d_attr", "difficulty": 1,
                                   "subj": lab, "attr_type": key, "answer": val,
                                   "answer_type": "word", "acceptable": [val],
                                   "question": f'What {key} is the {noun} labeled "{lab}"?',
                                   "provenance": [f"ATTR({lab},{key},{val})"]})
    rng.shuffle(attr_cands)
    out += attr_cands[:per_family]

    # ---- L2 d_edge: which element does X point/connect to? (accept any true neighbour) ----
    edge_cands = []
    by_src = {}
    for r in rels:
        s, o = r.get("subject"), r.get("object")
        if s and o:
            by_src.setdefault(s, []).append(o)
    for src, objs in by_src.items():
        slab = label_of.get(src)
        if not slab or lab_count[slab] != 1:
            continue
        tlabs = sorted({label_of.get(o) for o in objs if label_of.get(o)
                        and lab_count[label_of.get(o)] == 1})
        if tlabs:
            verb = "point to" if any(r.get("predicate") == "arrow_to" for r in rels) else "connect to"
            edge_cands.append({"domain": "diagrams", "family": "d_edge", "difficulty": 2,
                               "subj": slab, "answer": tlabs[0], "answer_type": "word",
                               "acceptable": tlabs,
                               "question": f'Which {noun} does "{slab}" {verb}? Name one.',
                               "provenance": [f"EDGE({slab}->{t})" for t in tlabs]})
    rng.shuffle(edge_cands)
    out += edge_cands[:per_family]

    # ---- L3 d_degree: how many DISTINCT other elements is X directly connected to? ----
    # (distinct neighbours, not edge multiplicity: a flowchart Yes/No both entering the same node
    # counts once — the human audit flagged the multiplicity version as wrong)
    deg_cands = []
    nbrs = {}
    for r in rels:
        s, o = r.get("subject"), r.get("object")
        if s and o:
            nbrs.setdefault(s, set()).add(o)
            nbrs.setdefault(o, set()).add(s)
    deg = {k: len(v - {k}) for k, v in nbrs.items()}
    for e in ents:
        lab = _dg_attr(e, "label")
        if not lab or lab_count[lab] != 1 or deg.get(e["name"], 0) == 0:
            continue
        k = deg[e["name"]]
        deg_cands.append({"domain": "diagrams", "family": "d_degree", "difficulty": 3,
                          "subj": lab, "answer": _fmt_num(k), "answer_type": "number",
                          "acceptable": [_fmt_num(k)],
                          "question": f'How many other {noun}s is "{lab}" directly connected to?',
                          "provenance": [f"DEGREE({lab},{k})"]})
    rng.shuffle(deg_cands)
    out += deg_cands[:per_family]
    return out


def _recompute_diagrams(prog, graph):
    ents, rels, label_of, lab_count = _dg_index(graph)
    fam = prog["family"]
    from collections import Counter
    if fam == "d_count":
        what = prog.get("what", "total")
        if what == "total":
            return [_fmt_num(len(ents))]
        col = what.split(":", 1)[1]
        k = sum(1 for e in ents if _dg_attr(e, "color") == col)
        return [_fmt_num(k)] if k else None
    lab = prog.get("subj")
    if fam == "d_attr":
        if lab_count.get(lab) != 1:
            return None
        for e in ents:
            if _dg_attr(e, "label") == lab:
                v = _dg_attr(e, prog["attr_type"])
                return [v] if v else None
        return None
    if fam == "d_edge":
        if lab_count.get(lab) != 1:
            return None
        src = next((e["name"] for e in ents if _dg_attr(e, "label") == lab), None)
        tlabs = sorted({label_of.get(r.get("object")) for r in rels if r.get("subject") == src
                        and label_of.get(r.get("object"))
                        and lab_count[label_of.get(r.get("object"))] == 1})
        return tlabs or None
    if fam == "d_degree":
        if lab_count.get(lab) != 1:
            return None
        name = next((e["name"] for e in ents if _dg_attr(e, "label") == lab), None)
        nbrs = {}
        for r in rels:
            s, o = r.get("subject"), r.get("object")
            if s and o:
                nbrs.setdefault(s, set()).add(o)
                nbrs.setdefault(o, set()).add(s)
        k = len(nbrs.get(name, set()) - {name})
        return [_fmt_num(k)] if k else None
    return None


# ============================================================ public API ============================================================
def recompute(prog, graph):
    """Independently execute a program over its graph -> acceptable answer set (or None).
    `graph` is a raw GQA entry (natural), a canon_charts dict (charts), or a diagram sidecar."""
    dom = prog["domain"]
    if dom == "natural":
        return _recompute_natural(prog, graph)
    if dom == "diagrams":
        return _recompute_diagrams(prog, graph)
    return _recompute_charts(prog, graph)


# ============================================================ REGIME B (predicted graph) ============================================================
def pred_to_gqa_entry(parsed):
    """Convert a predicted NATIVE-schema natural graph (from a model) into a GQA-entry-shaped
    dict so the SAME audited gen_natural / _recompute_natural run over it UNCHANGED.

    Predicted JSON: {objects:[{name, attributes:[..], relations:[{object:<NAME>, predicate}]}]}.
    We assign synthetic ids and resolve relation targets (given by NAME) to an id, mirroring the
    raw GQA shape {objects:{id:{name, attributes, relations:[{object:<id>, name:<pred>}], x,y,w,h}}}.
    """
    objs, name_to_ids = {}, {}
    plist = parsed.get("objects", []) or [] if isinstance(parsed, dict) else []
    for i, o in enumerate(plist):
        if not isinstance(o, dict):
            continue
        name = _norm(o.get("name", ""))
        if not name:
            continue
        oid = f"p{i}"
        a = o.get("attributes", [])
        attrs = list(a) if isinstance(a, list) else ([str(v) for v in a.values()] if isinstance(a, dict) else [])
        objs[oid] = {"name": name, "attributes": attrs, "relations": [], "x": 0, "y": 0, "w": 0, "h": 0}
        name_to_ids.setdefault(name, []).append(oid)
    for i, o in enumerate(plist):
        oid = f"p{i}"
        if not isinstance(o, dict) or oid not in objs:
            continue
        for r in (o.get("relations", []) or []):
            if not isinstance(r, dict):
                continue
            pred = _norm(r.get("predicate") or r.get("name") or "")
            tgt = _norm(r.get("object") or r.get("target") or "")
            ids = name_to_ids.get(tgt)
            if pred and ids:
                objs[oid]["relations"].append({"object": ids[0], "name": pred})
    return {"objects": objs, "width": None, "height": None}


def _ans_in_set(candidate, acc_set, answer_type, tol=0.05):
    """Does `candidate` (a gold-graph-derived answer) match ANY answer in acc_set, under the
    SAME strict rules as answer_correct? Used to cross-check a predicted-graph answer vs gold."""
    from .match import normalize, tokens
    if answer_type == "number":
        return any(value_eq(candidate, g, tol) for g in acc_set)
    ctoks = set(tokens(candidate))
    for g in acc_set:
        if normalize(candidate) == normalize(g):
            return True
        if answer_type == "word":
            gs = set(tokens(g))
            if gs and ctoks and (gs <= ctoks or ctoks <= gs) and abs(len(ctoks) - len(gs)) <= 1:
                return True
    return False


def regimeb_outcome(prog, gold_graph, tol=0.05):
    """Cross-check a program GENERATED OVER A PREDICTED GRAPH against the GOLD graph.
      correct      : the program is answerable on gold AND the predicted answer == the gold answer
      wrong        : answerable on gold but the predicted answer disagrees (extraction produced a
                     wrong-but-internally-consistent fact)
      unverifiable : not answerable on gold -- the predicted entity/fact is absent (hallucination)
                     or not well-posed in gold (e.g. the name is not unique there)
    gold_graph is a raw GQA entry (natural) or a canon_charts dict (charts)."""
    gold_acc = recompute(prog, gold_graph)
    if not gold_acc:
        return "unverifiable"
    return "correct" if _ans_in_set(prog["answer"], gold_acc, prog["answer_type"], tol) else "wrong"


def answer_correct(pred_text, prog, tol=0.05):
    """Is the model's free-text answer correct for this program?

    Scoring is deliberately STRICT-lexical (deterministic, no embeddings) to avoid false
    positives -- a semantic (MiniLM) matcher rates all colours mutually similar and would
    accept 'white' for gold 'pink'. Rules:
      number     : relaxed numeric match to the gold value (parse_num handles %/$/commas/K/M/B)
      word/label : normalized-equal OR token-subset to ANY acceptable gold answer, so
                   'a red shirt' matches gold 'shirt' and gold '2016*' matches pred '2016',
                   but distinct colours/labels never cross-match.
    """
    from .match import normalize, tokens
    pred = str(pred_text).strip()
    acc = prog.get("acceptable") or [prog.get("answer")]
    at = prog["answer_type"]
    if at == "number":
        return any(value_eq(pred, g, tol) for g in acc)
    ptoks = set(tokens(pred))
    for g in acc:
        if normalize(pred) == normalize(g):
            return True
        if at == "word":
            # accept only when the answer IS the gold word give-or-take ONE modifier
            # ('red shirt' ~ 'shirt', 'shoe' ~ 'tennis shoe'); a verbose sentence that merely
            # CONTAINS the gold token is NOT accepted (audit finding #1).
            gs = set(tokens(g))
            if gs and ptoks and (gs <= ptoks or ptoks <= gs) and abs(len(ptoks) - len(gs)) <= 1:
                return True
        # answer_type == 'label': exact-under-normalize only (labels are distinct by construction)
    return False


# ==================================================== EXPERIMENT 2: "LLM PROPOSES, PROGRAM DISPOSES" ====================================================
# A small VLM rewrites the templated `question` into natural language WITHOUT ever seeing the
# answer. The program (and therefore the by-construction gold) is NEVER changed -- only the
# surface text. `paraphrase_gate` is a DETERMINISTIC faithfulness check: a PASS guarantees the
# paraphrase still references the same program entities, keeps the same question *type*, adds no
# new constraint, and leaks no answer -- so the unchanged program still computes the same gold.
# Rejects fall back to the templated question (which is faithful by construction).
from .match import tokens as _mtokens, parse_num as _pnum

# Closed-class glue the paraphrase may use freely. INVARIANT: no content nouns / no attribute
# VALUE words / no object names here -- only function words + referent-neutral domain glue.
FUNCTION_WHITELIST = {
    "what", "which", "who", "whose", "how", "where", "when",
    "is", "are", "was", "were", "be", "being", "been", "s", "re",
    "do", "does", "did", "has", "have", "had", "can", "could", "would", "will",
    "the", "a", "an", "of", "to", "for", "with", "and", "or", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "his", "her", "he", "she", "you", "i", "me", "we",
    "there", "here", "one", "ones", "some", "any", "each", "both",
    "shown", "show", "shows", "showing", "displayed", "depicted", "labeled", "listed", "given",
    "picture", "image", "photo", "photograph", "scene", "see", "tell", "us",
    # referent-neutral chart / scene glue (never changes WHICH entity is meant)
    "value", "values", "category", "categories", "series", "chart", "graph", "bar", "line",
    "plot", "data", "amount", "number", "point", "label", "group", "between", "among",
    "compared", "compare", "than", "versus", "vs", "at", "in", "on", "by", "as",
    "much", "many", "about", "regarding", "if", "represent", "represents", "corresponds",
}

# attribute-type cues (a TYPE, not a specific value)
ATTR_TYPE_CUES = {
    "color": {"color", "colour", "colored", "coloured", "colors", "colours", "shade", "shades", "hue"},
    "material": {"material", "materials", "made"},          # 'made of' / 'made from'
    "size": {"size", "sized", "dimension", "dimensions"},   # (size attr is not paraphrased -- see below)
}
# chart operation cues (open-class synonym families)
MAX_CUES = {"highest", "largest", "greatest", "maximum", "max", "biggest", "top", "peak", "most", "tallest"}
MIN_CUES = {"lowest", "smallest", "least", "minimum", "min", "fewest", "bottom", "shortest"}
COMPARE_CUES = {"higher", "greater", "larger", "bigger", "more", "taller", "exceeds", "exceed"}
DIFF_CUES = {"difference", "differ", "differs", "gap", "minus", "subtract", "subtracted"}
RETRIEVE_CUES = {"value", "much", "many", "reach", "reached", "record", "recorded",
                 "report", "reported", "equal", "equals", "read", "reads"}

ATTR_VALUE_WORDS = COLORS | MATERIALS | SIZES     # concrete attribute VALUES -> never added/leaked


def _tokset(s):
    return set(_mtokens(s))


def _phrase_present(phrase, hay_tokens):
    """All content tokens of `phrase` appear in `hay_tokens` (SPICE-style token subset)."""
    pt = set(_mtokens(phrase))
    return bool(pt) and pt <= hay_tokens


def program_entities(prog):
    """Canonical spec of what a faithful paraphrase MUST preserve and MUST NOT introduce.

    This is the *only* place the family->obligation mapping lives; both the paraphrase prompt
    and the gate consume it, so they can never drift apart.

    Returns:
      required   : entity phrases that must ALL survive (object names / x-labels / series / predicate)
      type_cue   : token set, >=1 must appear (attribute-type or chart operation)
      forbid_cues: token set that must NOT appear (opposite polarity / wrong operation -> inverts answer)
      leak_check : reject if the WITHHELD gold answer surfaces (False when the answer IS a given entity)
      supports_paraphrase: False -> always fall back to template (answer vocab == cue vocab, unsafe)
    """
    fam, dom = prog["family"], prog["domain"]
    required, type_cue, forbid, leak, supports = [], set(), set(), True, True
    if dom == "natural":
        if fam == "attr":
            required = [prog["subj"]]
            at = prog["attr_type"]
            type_cue = set(ATTR_TYPE_CUES[at])
            forbid = set().union(*[v for k, v in ATTR_TYPE_CUES.items() if k != at]) - type_cue
            if at == "size":
                supports = False        # size ANSWERS are size words -> cue and answer share vocab -> unsafe
        elif fam in ("rel", "rel_nested"):
            # predicate is a REQUIRED phrase (must survive verbatim-ish); a swapped predicate is
            # caught as a missing-entity + added-content, so no answer-inverting drift slips through.
            required = [prog["subj"], prog["pred"]] + (
                [prog["disambig_attr"]] if prog.get("disambig_attr") else [])
        elif fam == "rel_attr":
            # 2-hop: both the subject, the relation, and the referenced target must survive, and the
            # attribute TYPE cue must be present (else the paraphrase could ask for a different attr).
            required = [prog["subj"], prog["pred"], prog["tgt"]]
            at = prog["attr_type"]
            type_cue = set(ATTR_TYPE_CUES[at])
            forbid = set().union(*[v for k, v in ATTR_TYPE_CUES.items() if k != at]) - type_cue
            if at == "size":
                supports = False        # size answers share vocab with size cues -> unsafe
    else:  # charts
        sreq = [prog["series"]] if prog.get("series") else []
        if fam == "retrieve":
            required = [prog["x"]] + sreq
            type_cue = set(RETRIEVE_CUES)
            forbid = MAX_CUES | MIN_CUES | DIFF_CUES | COMPARE_CUES
        elif fam == "extremum":
            required = sreq
            type_cue = MAX_CUES if prog["extremum"] == "max" else MIN_CUES
            forbid = (MIN_CUES if prog["extremum"] == "max" else MAX_CUES) | DIFF_CUES
        elif fam == "compare":
            required = [prog["x"], prog["x2"]] + sreq
            type_cue = COMPARE_CUES | MAX_CUES
            forbid = MIN_CUES | DIFF_CUES          # 'which is lower' would invert the gold label
            leak = False                            # answer IS one of x / x2 (a required entity)
        elif fam == "diff":
            required = [prog["x"], prog["x2"]] + sreq
            type_cue = set(DIFF_CUES)
            forbid = COMPARE_CUES | MAX_CUES | MIN_CUES  # abs-difference, not a directional comparison
    return {"required": [r for r in required if r], "type_cue": type_cue,
            "forbid_cues": forbid, "leak_check": leak, "supports_paraphrase": supports,
            "family": fam, "domain": dom}


def paraphrase_gate(text, prog, template):
    """Deterministic faithfulness gate. Returns (ok: bool, reasons: list[str]).

    PASS requires ALL of: single question; every program entity preserved; the type cue present
    and no answer-inverting / wrong-family cue; no new number, no new attribute value, no new
    content token beyond the closed-class whitelist; and (where applicable) no leaked gold answer.
    Any failure -> REJECT -> caller falls back to the templated question.
    """
    spec = program_entities(prog)
    if not spec["supports_paraphrase"]:
        return False, ["family-not-paraphrasable"]

    t = str(text).strip()
    ptoks = _tokset(t)
    tmpl_toks = _tokset(template)
    reasons = []

    # 1. exactly one question, sane length
    if t.count("?") != 1 or not t.endswith("?"):
        reasons.append("not-single-question")
    nwords = len([w for w in t.replace("?", " ").split() if w])
    if nwords < 3 or nwords > 40:
        reasons.append("length-out-of-range")

    # 2. entity preservation. Only enforce entities the TEMPLATE itself states -- a
    # single-series chart omits the series name, so the paraphrase needn't invent it.
    entity_toks = set()
    for ent in spec["required"]:
        if not _phrase_present(ent, tmpl_toks):
            continue                                      # template didn't ask it -> not required
        entity_toks |= _tokset(ent)
        if not _phrase_present(ent, ptoks):
            reasons.append("missing-entity:" + str(ent))

    # 3. question-type cue present; answer-inverting / wrong-family cue absent. A forbidden cue
    # already in the template (an x-label literally named "Top"/"Least") is not drift -- only a
    # NEWLY INTRODUCED inverting cue is rejected.
    if spec["type_cue"] and not (spec["type_cue"] & ptoks):
        reasons.append("missing-type-cue")
    bad = (spec["forbid_cues"] & ptoks) - tmpl_toks - entity_toks
    if bad:
        reasons.append("forbidden-cue:" + ",".join(sorted(bad)))

    # 4a. no new number (leaked value or a new numeric constraint)
    tmpl_nums = {round(_pnum(w), 6) for w in _tokset(template) if _pnum(w) is not None}
    for w in ptoks:
        v = _pnum(w)
        if v is not None and round(v, 6) not in tmpl_nums:
            reasons.append("new-number:" + w)

    # 4b. no new attribute value / no new content token (added constraint or leaked answer)
    allowed = set(FUNCTION_WHITELIST) | spec["type_cue"] | spec["forbid_cues"] | _tokset(template)
    for ent in spec["required"]:
        allowed |= _tokset(ent)
    for w in ptoks:
        if w in allowed or _pnum(w) is not None:      # numbers already adjudicated in 4a
            continue
        reasons.append(("new-attr-value:" + w) if w in ATTR_VALUE_WORDS else ("added-content:" + w))

    # 4c. explicit answer-leak: the WITHHELD gold surfaced verbatim
    if spec["leak_check"]:
        for g in (prog.get("acceptable") or [prog.get("answer")]):
            if prog["answer_type"] == "number":
                gv = _pnum(g)
                if gv is not None and round(gv, 6) not in tmpl_nums and \
                        any(_pnum(w) is not None and abs(_pnum(w) - gv) <= 1e-6 for w in ptoks):
                    reasons.append("answer-leak")
            elif g and _phrase_present(g, ptoks):
                reasons.append("answer-leak")

    return (len(reasons) == 0), reasons


def ask_type_text(prog):
    """Human-readable description of WHAT to ask -- never reveals the answer. Feeds the prompt."""
    fam = prog["family"]
    s = prog.get("series")
    fors = f' in the "{s}" series' if s else ""
    if fam == "attr":
        at = prog["attr_type"]
        verb = {"color": "what color", "material": "what material", "size": "what size"}[at]
        made = " (what it is made of)" if at == "material" else ""
        return f'Ask {verb} the "{prog["subj"]}" is{made}.'
    if fam in ("rel", "rel_nested"):
        subj = f'the {prog["disambig_attr"]} "{prog["subj"]}"' if prog.get("disambig_attr") else f'the "{prog["subj"]}"'
        return f'Ask what {subj} is {prog["pred"]} (name the thing it is {prog["pred"]}).'
    if fam == "rel_attr":
        at = prog["attr_type"]
        verb = {"color": "what color", "material": "what material", "size": "what size"}[at]
        made = " (what it is made of)" if at == "material" else ""
        return (f'Ask {verb} the "{prog["tgt"]}" that the "{prog["subj"]}" is {prog["pred"]} is{made} '
                f'-- identify it via that relation, not by naming it directly.')
    if fam == "retrieve":
        return f'Ask for the value at "{prog["x"]}"{fors}.'
    if fam == "extremum":
        pol = "highest" if prog["extremum"] == "max" else "lowest"
        return f'Ask which category has the {pol} value{fors}.'
    if fam == "compare":
        return f'Ask which of "{prog["x"]}" or "{prog["x2"]}" has the higher value{fors}.'
    if fam == "diff":
        return f'Ask for the difference between the values at "{prog["x"]}" and "{prog["x2"]}"{fors}.'
    return prog["question"]
