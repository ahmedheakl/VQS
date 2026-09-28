"""Generate questions with the AUDITED sg_gen family modules. Nothing here invents a template.

Modules called (family counts as measured in sg_gen):
  natural  gen_natural_enriched2.gen        16 families
  natural  gen_natural_enriched.gen_enriched 8 families
  natural  vqa_engine.gen_natural            4 families (audited core)
  charts   chart_qa_families_v5.propose_v5  30 families
  charts   chart_qa_gen  (via vqa_engine.gen_charts core) 4 families
  infog    infog_families2.chart_families    8 families
  diagram  infog_families2.diagram_families 10 families

Our parser emits its own JSON; each module wants a different shape, so this file is ONLY
converters + calls. Every answer is recomputed by the module's own executor, so a converter
bug shows up as a self-test failure rather than as silently wrong training data.

Usage:  python scripts/gen_questions_v2.py --selftest
        python scripts/gen_questions_v2.py --parses runs/parse/all.jsonl --out runs/qa4/raw.jsonl
"""
import argparse
import collections
import json
import pathlib
import ast
import math
import random
import re
import sys

# the template-family modules are vendored under vqs/src/ so this repo runs standalone
SG = str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, SG)
from src import chart_qa_families_v5 as v5          # noqa: E402
from src import chart_qa_gen as cqg                 # noqa: E402
from src import gen_natural_enriched as ge1         # noqa: E402
from src import gen_natural_enriched2 as ge2        # noqa: E402
from src import infog_families2 as inf2            # noqa: E402
from src import vqa_engine as ve                    # noqa: E402


# ---------------------------------------------------------------- converters
BAD_VALUES = collections.Counter()


def is_tuple_repr(x):
    """True only for an actual Python tuple repr like "('Yes', 'free', ['CELL(..)'])".
    Must NOT fire on a real answer such as "(ii)" or "(a)"."""
    s = str(x).strip()
    if not (s.startswith("(") and s.endswith(")")):
        return False
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return False
    return isinstance(v, tuple) and len(v) >= 2


def clean_val(v):
    """Reject values the templates cannot compute with. A parse that emitted 1e400 made
    chart_qa_gen raise OverflowError on round(); clean the INPUT, count what is dropped."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.findall(r"[-+]?\d[\d,]*\.?\d*", s.replace(" ", ""))
    if not m:
        return s                      # non-numeric label, fine
    try:
        f = float(m[0].replace(",", ""))
    except ValueError:
        BAD_VALUES["unparseable"] += 1
        return None
    if not math.isfinite(f) or abs(f) > 1e12:
        BAD_VALUES["non_finite_or_huge"] += 1
        return None
    return s

def to_gqa_entry(parse):
    """our natural parse -> GQA entry shape, keeping boxes so frame/position families work."""
    objs, name_ids = {}, {}
    for o in parse.get("objects", []):
        n = ve._norm(o.get("object_name", ""))
        if not n:
            continue
        oid = f"p{o.get('object_id')}"
        b = o.get("box_0_1000") or [0, 0, 0, 0]
        objs[oid] = {"name": n, "attributes": list(o.get("attributes") or []),
                     "relations": [], "x": b[0], "y": b[1],
                     "w": max(b[2] - b[0], 0), "h": max(b[3] - b[1], 0)}
        name_ids.setdefault(n, []).append(oid)
    ids = {o.get("object_id"): f"p{o.get('object_id')}" for o in parse.get("objects", [])}
    for r in parse.get("relations", []):
        s, t = ids.get(r.get("subject_id")), ids.get(r.get("object_id"))
        p = ve._norm(r.get("predicate", ""))
        if s in objs and t in objs and p:
            objs[s]["relations"].append({"object": t, "name": p})
    return {"objects": objs, "width": 1000, "height": 1000}


def to_table(parse):
    """our chart parse -> chart_qa_gen / v5 `table` shape (points are (x, y) pairs)."""
    series = []
    for s in parse.get("series", []):
        pts = [(p.get("category"), clean_val(p.get("printed_value")))
               for p in s.get("points", []) if p.get("category")]
        pts = [(x, y) for x, y in pts if y is not None]
        if pts:
            series.append({"name": s.get("series_name", ""), "points": pts})
    return {"title": parse.get("title", ""), "series": series, "chart_type": "bar",
            "x_title": parse.get("x_axis_title", ""), "y_title": parse.get("y_axis_title", "")}


def to_infog_struct(parse):
    """our infographic parse -> infog_families2 `struct` (blocks style)."""
    pts = [{"x": e.get("entry_label"), "y": clean_val(e.get("printed_value"))}
           for e in parse.get("entries", []) if e.get("entry_label")]
    pts = [p for p in pts if p["y"] is not None]
    return {"title": parse.get("title", ""),
            "blocks": [{"kind": "chart", "title": parse.get("title", ""),
                        "series": [{"name": "", "points": pts}]}]}


def to_ai2d_ann(parse):
    """our diagram parse -> AI2D annotation shape (text ids + relationships)."""
    text, rels, i = {}, {}, 0
    for n in parse.get("nodes", []):
        if n.get("node_label"):
            text[f"T{n['node_id']}"] = {"value": n["node_label"]}
    if parse.get("title"):
        text["T_title"] = {"value": parse["title"]}
        rels["r_title"] = {"category": "imageTitle", "origin": "T_title", "destination": None}
    if parse.get("has_arrows"):
        for e in parse.get("edges", []):
            o, d = f"T{e.get('from_node_id')}", f"T{e.get('to_node_id')}"
            if o in text and d in text:
                rels[f"r{i}"] = {"category": "interObjectLinkage", "origin": o,
                                 "destination": d, "hasDirectionality": True}
                i += 1
    return {"text": text, "relationships": rels}


# ---------------------------------------------------------------- generation
def natural_progs(parse, rng, vocab, per_family):
    e = to_gqa_entry(parse)
    out = []
    for fn, args in ((ge2.gen, (e, rng, vocab, per_family)),
                     (ge1.gen_enriched, (e, rng, vocab, per_family)),
                     (ve.gen_natural, (e, rng, per_family))):
        out += list(fn(*args) or [])
    return out


def chart_progs(parse, rng, per_family):
    t = to_table(parse)
    if not t["series"]:
        return []
    out = list(ve.gen_charts(t, rng, per_family) or [])
    for q in (cqg.build(t, rng, max_per_image=60) or []):
        if q.get("question") and q.get("answer") not in (None, ""):
            out.append({"domain": "charts", "family": q.get("family"),
                        "difficulty": q.get("difficulty", 2), "question": q["question"],
                        "answer": str(q["answer"]),
                        "answer_type": q.get("answer_type", "word"),
                        "acceptable": [str(x) for x in (q.get("acceptable") or [q["answer"]])]})
    for p in (v5.propose_v5(t, rng, max_per_image=40) or []):
        res = v5.execute_v5(p, t)   # returns (answer, answer_type, provenance) -- MUST unpack
        if res is None:
            continue
        ans, atype, prov = res
        q = p.get("question") or p.get("q")
        if not q or ans is None or str(ans).strip() == "":
            continue
        out.append({"domain": "charts", "family": p.get("family"),
                    "difficulty": p.get("difficulty", 2), "question": q,
                    "answer": str(ans), "answer_type": atype or "word",
                    "acceptable": [str(ans)], "provenance": prov})
    return out


def infog_progs(parse, rng, per_family):
    st = to_infog_struct(parse)
    out = []
    # an infographic block is a chart, so the 30 v5 + 13 chart_qa_gen families apply here too
    tbl = {"title": parse.get("title", ""), "chart_type": "bar", "x_title": "", "y_title": "",
           "series": [{"name": "", "points": [(e.get("entry_label"), clean_val(e.get("printed_value")))
                                              for e in parse.get("entries", [])
                                              if e.get("entry_label")
                                              and clean_val(e.get("printed_value")) is not None]}]}
    if tbl["series"][0]["points"]:
        for q in (cqg.build(tbl, rng, max_per_image=40) or []):
            if q.get("question") and q.get("answer") not in (None, ""):
                out.append({"domain": "infographics", "family": "ig_" + str(q.get("family")),
                            "difficulty": q.get("difficulty", 2), "question": q["question"],
                            "answer": str(q["answer"]), "answer_type": q.get("answer_type", "word"),
                            "acceptable": [str(x) for x in (q.get("acceptable") or [q["answer"]])]})
        for pr in (v5.propose_v5(tbl, rng, max_per_image=30) or []):
            res = v5.execute_v5(pr, tbl)
            qq = pr.get("question") or pr.get("q")
            if res is None or not qq:
                continue
            ans, atype, prov = res
            if ans is None or str(ans).strip() == "":
                continue
            out.append({"domain": "infographics", "family": "ig_" + str(pr.get("family")),
                        "difficulty": pr.get("difficulty", 2), "question": qq,
                        "answer": str(ans), "answer_type": atype or "word",
                        "acceptable": [str(ans)], "provenance": prov})
    for fam, diff, q, ans, prog in (inf2.chart_families(st, 0, 0) or []):
        out.append({"domain": "infographics", "family": fam, "difficulty": diff,
                    "question": q, "answer": str(ans), "answer_type": "word",
                    "acceptable": [str(ans)], "prog": prog})
    return out


def diagram_progs(parse, rng, per_family):
    ann = to_ai2d_ann(parse)
    out = []
    for fam, diff, q, ans, prog in (inf2.diagram_families(ann) or []):
        out.append({"domain": "diagrams", "family": fam, "difficulty": diff,
                    "question": q, "answer": str(ans), "answer_type": "word",
                    "acceptable": [str(ans)], "prog": prog})
    out += list(ve.gen_diagrams({"entities": [], "relations": []}, rng, per_family) or [])
    return out


GEN = {"natural": natural_progs, "charts": chart_progs,
       "infographics": infog_progs, "diagrams": diagram_progs}


def build_vocab(parses):
    """Common object names FROM OUR OWN PARSES -- never from GQA scene graphs, so the loop
    stays label-free. Used only for closed-world 'no' sampling in existence families."""
    c = collections.Counter()
    for r in parses:
        if r["domain"] != "natural":
            continue
        for o in r["parse"].get("objects", []):
            n = ve._norm(o.get("object_name", ""))
            if n and " " not in n:
                c[n] += 1
    return [n for n, k in c.most_common() if k >= 20][:400]


def selftest():
    rng = random.Random(0)
    nat = {"objects": [{"object_id": 0, "object_name": "dog", "attributes": ["brown"],
                        "box_0_1000": [10, 10, 400, 400]},
                       {"object_id": 1, "object_name": "ball", "attributes": ["red"],
                        "box_0_1000": [600, 600, 800, 800]}],
           "relations": [{"subject_id": 0, "predicate": "chasing", "object_id": 1}]}
    ch = {"title": "Population of Romania", "x_axis_title": "year", "y_axis_title": "millions",
          "series": [{"series_name": "Romania", "points": [
              {"category": "2015", "printed_value": "19.8", "numeric_value": 19.8},
              {"category": "2016", "printed_value": "19.6", "numeric_value": 19.6},
              {"category": "2017", "printed_value": "19.5", "numeric_value": 19.5},
              {"category": "2018", "printed_value": "19.4", "numeric_value": 19.4}]}]}
    ig = {"title": "Energy use", "entries": [
        {"entry_label": "US", "printed_value": "4475", "numeric_value": 4475, "unit": "TWh"},
        {"entry_label": "China", "printed_value": "7500", "numeric_value": 7500, "unit": "TWh"},
        {"entry_label": "India", "printed_value": "1500", "numeric_value": 1500, "unit": "TWh"},
        {"entry_label": "Japan", "printed_value": "1000", "numeric_value": 1000, "unit": "TWh"}]}
    dg = {"title": "Life cycle of a frog", "has_arrows": True,
          "nodes": [{"node_id": 1, "node_label": "egg"}, {"node_id": 2, "node_label": "tadpole"},
                    {"node_id": 3, "node_label": "froglet"}, {"node_id": 4, "node_label": "frog"}],
          "edges": [{"from_node_id": 1, "to_node_id": 2, "edge_label": ""},
                    {"from_node_id": 2, "to_node_id": 3, "edge_label": ""},
                    {"from_node_id": 3, "to_node_id": 4, "edge_label": ""}]}
    vocab = ["cat", "tree", "car", "person", "table"]
    tot = 0
    for dom, parse in (("natural", nat), ("charts", ch), ("infographics", ig), ("diagrams", dg)):
        ps = GEN[dom](parse, rng, vocab, 3) if dom == "natural" else GEN[dom](parse, rng, 3)
        fams = sorted({p.get("family") for p in ps})
        print(f"{dom:14s} {len(ps):4d} questions, {len(fams):2d} families: {fams}")
        for p in ps[:3]:
            print(f"     [{p.get('family')}/d{p.get('difficulty')}] {p.get('question')}"
                  f"  -> {p.get('answer')!r}")
        tot += len(ps)
    assert tot > 0, "generated nothing"
    print(f"SELFTEST produced {tot} questions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parses")
    ap.add_argument("--out")
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0,
                    help="a later cycle passes a new seed to draw new questions from the same parses")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rows = [json.loads(l) for l in open(a.parses)]
    vocab = build_vocab(rows)
    print(f"vocab from our own parses: {len(vocab)} names", flush=True)
    rng = random.Random(a.seed)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fam_c, dom_c, diff_c, n = collections.Counter(), collections.Counter(), collections.Counter(), 0
    with open(a.out, "w") as f:
        for r in rows:
            d = r["domain"]
            try:
                ps = (natural_progs(r["parse"], rng, vocab, a.per_family) if d == "natural"
                      else GEN[d](r["parse"], rng, a.per_family))
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError,
                    OverflowError, AttributeError) as e:
                BAD_VALUES["gen_error/" + type(e).__name__] += 1
                continue
            for p in ps:
                if not p.get("question") or p.get("answer") in (None, ""):
                    continue
                # 46% of one pool shipped with gold = "('Yes', 'free', [...])" and the correct
                # answer scored 0. Never again. Must detect a real tuple repr, not a legitimate
                # option label like "(ii)" from sequence_order.
                assert not is_tuple_repr(p["answer"]), \
                    f"tuple gold leaked from {p.get('family')}: {str(p['answer'])[:80]}"
                f.write(json.dumps({**p, "image": r["image"], "domain": d,
                                    "image_id": r["id"]}) + "\n")
                fam_c[p.get("family")] += 1
                dom_c[d] += 1
                diff_c[p.get("difficulty")] += 1
                n += 1
    print(f"images={len(rows)} questions={n} ({n/max(len(rows),1):.2f}/image)")
    print("by domain:", dict(dom_c))
    print("by difficulty:", dict(sorted(diff_c.items(), key=lambda kv: str(kv[0]))))
    if BAD_VALUES:
        print("dropped inputs (counted, not hidden):", dict(BAD_VALUES))
    print(f"{len(fam_c)} families used:")
    for k, v in fam_c.most_common():
        print(f"   {k:32s} {v}")


if __name__ == "__main__":
    main()
