"""Enriched NATURAL QA generator over GOLD GQA graphs (adds the families found in the 1,000-QA
manual audit that the 4-family engine was missing).

Why gold-only: the audit flags existence / or_existence / and_existence / *_filtered_existence /
attribute_comparison as "closed-world unsafe" on a sparse PREDICTED graph — absence in a sparse
graph does not imply absence in the image. Over the GOLD graph the closed-world assumption is
exactly the one GQA itself used to author these questions, so the engine's answers are
correct-by-construction and answer-blind (the engine computes them; no model picks them).

Every program stores only the SLOTS; `recompute(prog, entry)` re-derives the answer from the graph
independently, so `--selftest` proves 100% reproducibility (same bar as vqa_engine).

Families added (difficulty per the audit's calibrated rule):
  existence                    L2  "Are there mushrooms in this photo?"            (T/F)
  or_existence                 L3  "Are there fences or cars in the photo?"        (T/F)
  and_existence                L3  "Do you see both windows and doors there?"      (T/F)
  attribute_filtered_existence L3  "Is there a wooden chair in the picture?"       (T/F)
  relation_existence           L2  "Are there women to the left of the brown bag?" (T/F, boxes)
  attribute_comparison         L3  "Are the tree and the bush the same color?"     (T/F)
  attribute_conjunction        L3  "Is the grass green and tall?"                  (T/F)
  frame_position               L2  "On which side is the trash can, left or right?"(choice, boxes)
The 4 original families (attr/rel/rel_nested/rel_attr) come from vqa_engine.gen_natural unchanged.
"""
import random
from collections import Counter
from . import vqa_engine as ve
from .vqa_engine import (_natural_index, _good_referent, _typed_attr, _norm, _is_plural,
                         _area_frac, ATTR_TYPES, COLORS, MATERIALS, SIZES, MIN_REF_AREA)

_IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children",
              "foot": "feet", "tooth": "teeth", "mouse": "mice", "goose": "geese"}
_ALREADY_PLURAL = {"people", "men", "women", "children", "feet", "teeth", "mice", "geese",
                   "glasses", "scissors", "pants", "jeans", "shorts"}


def _plural(n):
    if n in _ALREADY_PLURAL:
        return n
    if n in _IRREGULAR:
        return _IRREGULAR[n]
    if n.endswith(("s", "x", "ch", "sh", "z")):
        return n + "es"
    if len(n) > 1 and n.endswith("y") and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def _center(o):
    try:
        return o["x"] + o["w"] / 2.0, o["y"] + o["h"] / 2.0
    except Exception:
        return None


def build_vocab(gqa, min_images=60):
    """Common singular object names seen across the corpus, for closed-world 'no' sampling."""
    c = Counter()
    for e in gqa.values():
        seen = {_norm(o.get("name", "")) for o in (e.get("objects", {}) or {}).values()}
        for n in seen:
            if n:
                c[n] += 1
    return [n for n, k in c.most_common() if k >= min_images
            and not _is_plural(n) and n not in _ALREADY_PLURAL and " " not in n]


def _typed_val(o, atype):
    return _typed_attr(o.get("attributes", []) or [], atype)


def _present(entry):
    """normalized name -> list of (oid, obj)."""
    d = {}
    for oid, o in (entry.get("objects", {}) or {}).items():
        n = _norm(o.get("name", ""))
        if n:
            d.setdefault(n, []).append((oid, o))
    return d


# ------------------------------------------------------------------ generation
def gen_enriched(entry, rng, vocab, per_family=1):
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    pres = _present(entry)
    present_names = set(pres)
    # singular, readable, uniquely-named referents good for a question
    good_singular = [n for n in present_names
                     if name_count.get(n) == 1 and not _is_plural(n)
                     and _good_referent(pres[n][0][1], n, W, H)]
    out = []

    def add(fam, diff, q, ans, fmt, slots):
        p = {"domain": "natural", "family": fam, "difficulty": diff, "question": q,
             "answer": str(ans).lower(), "answer_type": "word", "format": fmt}
        p.update(slots)
        out.append(p)

    # ---- existence (L2): a present readable class -> yes; a vocab-absent class -> no ----
    ex = []
    for n in good_singular:
        ex.append((n, "yes"))
    absent = [v for v in vocab if v not in present_names]
    rng.shuffle(absent)
    for v in absent[:max(1, per_family)]:
        ex.append((v, "no"))
    rng.shuffle(ex)
    for n, a in ex[:per_family * 2]:
        add("existence", 2, f"Are there any {_plural(n)} in the image?", a, "T" if a == "yes" else "F",
            {"q_name": n})

    # ---- or_existence (L3) / and_existence (L3) ----
    cand_names = good_singular + [v for v in absent[:20]]
    if len(cand_names) >= 2:
        for _ in range(per_family):
            a, b = rng.sample(cand_names, 2)
            pa, pb = a in present_names, b in present_names
            add("or_existence", 3, f"Are there any {_plural(a)} or {_plural(b)} in the photo?",
                "yes" if (pa or pb) else "no", "T" if (pa or pb) else "F", {"q_a": a, "q_b": b})
            a, b = rng.sample(cand_names, 2)
            pa, pb = a in present_names, b in present_names
            add("and_existence", 3, f"Do you see both {_plural(a)} and {_plural(b)} in the picture?",
                "yes" if (pa and pb) else "no", "T" if (pa and pb) else "F", {"q_a": a, "q_b": b})

    # ---- attribute_filtered_existence (L3): "Is there a {attr} {class}?" ----
    af = []
    for n in good_singular:
        o = pres[n][0][1]
        for atype in ATTR_TYPES:
            v = _typed_val(o, atype)
            if v:
                af.append((v, n, "yes"))                       # true attribute of a present object
                # a false-but-plausible attribute of the SAME type (not held by this object) -> no
                oattrs = {_norm(a) for a in (o.get("attributes", []) or [])}
                pool = list(ATTR_TYPES[atype] - oattrs)
                if pool:
                    af.append((rng.choice(pool), n, "no"))
                break
    rng.shuffle(af)
    for v, n, a in af[:per_family * 2]:
        add("attribute_filtered_existence", 3, f"Is there a {v} {n} in the picture?", a,
            "T" if a == "yes" else "F", {"q_attr": _norm(v), "q_name": n})

    # ---- relation_existence (L2): direction from gold boxes ----
    DIRS = [("to the left of", "left"), ("to the right of", "right"),
            ("above", "above"), ("below", "below")]
    anchors = [n for n in good_singular if _center(pres[n][0][1])]
    re_cands = []
    for anchor in anchors:
        bc = _center(pres[anchor][0][1])
        for qn in present_names:
            if qn == anchor:
                continue
            cens = [_center(o) for _, o in pres[qn] if _center(o)]
            if not cens:
                continue
            for phrase, key in DIRS:
                truth = any(_dir_true(key, ac, bc, W, H) for ac in cens)
                re_cands.append((qn, phrase, anchor, "yes" if truth else "no"))
    rng.shuffle(re_cands)
    # keep a balance of yes/no
    picked = []
    yes = [c for c in re_cands if c[3] == "yes"][:per_family]
    no = [c for c in re_cands if c[3] == "no"][:per_family]
    for qn, phrase, anchor, a in (yes + no):
        add("relation_existence", 2, f"Are there any {_plural(qn)} {phrase} the {anchor}?", a,
            "T" if a == "yes" else "F", {"q_name": qn, "dir": phrase, "anchor": anchor})

    # ---- attribute_comparison (L3): same {atype}? ----
    for atype in ATTR_TYPES:
        withattr = [(n, _norm(_typed_val(pres[n][0][1], atype))) for n in good_singular
                    if _typed_val(pres[n][0][1], atype)]
        if len(withattr) >= 2:
            for _ in range(per_family):
                (a, va), (b, vb) = rng.sample(withattr, 2)
                add("attribute_comparison", 3,
                    f"Are the {a} and the {b} the same {atype}?", "yes" if va == vb else "no",
                    "T" if va == vb else "F", {"q_a": a, "q_b": b, "atype": atype})

    # ---- attribute_conjunction (L3): "Is the {obj} {a1} and {a2}?" ----
    for n in good_singular:
        o = pres[n][0][1]
        typed = [(t, _norm(_typed_val(o, t))) for t in ATTR_TYPES if _typed_val(o, t)]
        if len(typed) >= 2:                                    # yes: two real attributes
            (t1, v1), (t2, v2) = typed[0], typed[1]
            add("attribute_conjunction", 3, f"Is the {n} {v1} and {v2}?", "yes", "T",
                {"subj": n, "a1": v1, "a2": v2})
        if len(typed) >= 1:                                    # no: one real + one false (unheld) attr
            t1, v1 = typed[0]
            oattrs = {_norm(a) for a in (o.get("attributes", []) or [])}
            other = list(ATTR_TYPES[t1] - oattrs)
            if other:
                v2 = rng.choice(other)
                add("attribute_conjunction", 3,
                    f"Is the {n} {v1} and {v2}?", "no", "F",
                    {"subj": n, "a1": v1, "a2": v2})
        if len([x for x in out if x["family"] == "attribute_conjunction"]) >= per_family * 2:
            break

    # ---- frame_position (L2): left/right from box center ----
    fp = []
    for n in good_singular:
        c = _center(pres[n][0][1])
        if not c:
            continue
        cx = c[0]
        if cx < W * 0.44:
            fp.append((n, "left"))
        elif cx > W * 0.56:
            fp.append((n, "right"))
    rng.shuffle(fp)
    for n, side in fp[:per_family]:
        add("frame_position", 2, f"On which side of the picture is the {n}, the left or the right?",
            side, "C", {"subj": n})

    return out


def _dir_true(key, ac, bc, W, H):
    mx, my = 0.02 * W, 0.02 * H
    if key == "left":
        return ac[0] < bc[0] - mx
    if key == "right":
        return ac[0] > bc[0] + mx
    if key == "above":
        return ac[1] < bc[1] - my
    if key == "below":
        return ac[1] > bc[1] + my
    return False


# ------------------------------------------------------------------ independent recompute (audit)
def recompute(prog, entry):
    fam = prog["family"]
    objs, id2name, name_count = _natural_index(entry)
    W, H = entry.get("width") or 1, entry.get("height") or 1
    pres = _present(entry)
    present = set(pres)
    if fam == "existence":
        return "yes" if prog["q_name"] in present else "no"
    if fam == "or_existence":
        return "yes" if (prog["q_a"] in present or prog["q_b"] in present) else "no"
    if fam == "and_existence":
        return "yes" if (prog["q_a"] in present and prog["q_b"] in present) else "no"
    if fam == "attribute_filtered_existence":
        n, want = prog["q_name"], prog["q_attr"]
        for _, o in pres.get(n, []):
            if want in {_norm(a) for a in (o.get("attributes", []) or [])}:
                return "yes"
        return "no"
    if fam == "relation_existence":
        n, anchor = prog["q_name"], prog["anchor"]
        if name_count.get(anchor) != 1 or anchor not in pres or n not in pres:
            return None
        bc = _center(pres[anchor][0][1])
        key = {"to the left of": "left", "to the right of": "right",
               "above": "above", "below": "below"}[prog["dir"]]
        cens = [_center(o) for _, o in pres[n] if _center(o)]
        if not bc or not cens:
            return None
        return "yes" if any(_dir_true(key, ac, bc, W, H) for ac in cens) else "no"
    if fam == "attribute_comparison":
        a, b, at = prog["q_a"], prog["q_b"], prog["atype"]
        if name_count.get(a) != 1 or name_count.get(b) != 1:
            return None
        va, vb = _typed_val(pres[a][0][1], at), _typed_val(pres[b][0][1], at)
        if not va or not vb:
            return None
        return "yes" if _norm(va) == _norm(vb) else "no"
    if fam == "attribute_conjunction":
        n = prog["subj"]
        if name_count.get(n) != 1:
            return None
        attrs = {_norm(a) for a in (pres[n][0][1].get("attributes", []) or [])}
        return "yes" if (prog["a1"] in attrs and prog["a2"] in attrs) else "no"
    if fam == "frame_position":
        n = prog["subj"]
        if name_count.get(n) != 1:
            return None
        c = _center(pres[n][0][1])
        if not c:
            return None
        return "left" if c[0] < W * 0.5 else "right"
    return None
