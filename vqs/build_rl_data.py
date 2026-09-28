"""Package the questions that passed the filters as EasyR1 training data.

Each training item is the image, the question and a suffix that fixes the answer format; the
suffix is chosen by the program's answer type (paper A.4.4). The reward is Eq. (2), exact match
plus a format term, so the gold carries only what the matcher needs.

Training rows are written in curriculum order (paper Sec. 4 and Table 6, "within-family hops"):
each template family's questions are sorted by hop count, fewest first, and the families are
interleaved so every family moves from its easiest to its hardest questions over the whole run.
scripts/train.sh sets data.shuffle=false so EasyR1 keeps this order. The two ablation orders of
Table 6 are available as --order global_hops and --order shuffle.

Usage: python vqs/build_rl_data.py --qa data/gated.jsonl --out data/rl_pool
"""
import argparse
import json
import pathlib
import random

FMT = {"number": "Answer the question using a single number.",
       "word": "Answer the question using a single word or phrase.",
       "mc": "Answer with the option's letter from the given choices directly."}
SUFFIX = "\n{fmt}"

NUMERIC_FAMS = {"chart_count", "chart_diff", "chart_sum", "diag_count_nodes", "diag_degree",
                "nat_count", "info_count", "info_diff"}


def answer_type(row):
    """Which answer-format suffix the question gets."""
    if row.get("answer_type") == "letter":
        return "mc"
    if row["family"] in NUMERIC_FAMS:
        return "number"
    a = str(row["answer"]).replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        float(a)
        return "number"
    except ValueError:
        return "word"


def hops_of(row):
    # sg_gen families carry `difficulty`, the older ones `hops`; both count program operations
    return int(row.get("hops", row.get("difficulty", 2)))


def gold_of(row):
    """What eq(y, a) compares against -- in the reward, the blind gate and the difficulty band."""
    acc = row.get("acceptable") or [row["answer"]]
    return {"answer": str(row["answer"]), "answer_type": row.get("answer_type", answer_type(row)),
            "acceptable": [str(x) for x in acc]}


def order_rows(rows, how, rng):
    """Training order (paper Table 6): family_hops (default), global_hops or shuffle."""
    rows = list(rows)
    rng.shuffle(rows)                       # random order among equal keys
    if how == "shuffle":
        return rows
    if how == "global_hops":
        return sorted(rows, key=lambda r: r["hops"])
    assert how == "family_hops", how
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    keyed = []
    for group in fams.values():
        group.sort(key=lambda r: r["hops"])
        # place the i-th of n questions at relative position (i + u) / n, so all families run
        # side by side and each reaches its highest hop count at the end of the run
        keyed += [((i + rng.random()) / len(group), r) for i, r in enumerate(group)]
    keyed.sort(key=lambda kr: kr[0])
    return [r for _, r in keyed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="family_hops",
                    choices=["family_hops", "global_hops", "shuffle"])
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap-per-domain", type=int, default=0,
                    help="balance domains; 0 = no cap")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.qa)]
    assert rows, "no gated questions"
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    if a.cap_per_domain:
        # round 1 was 60% natural / 2% diagrams, yet AI2D was the only benchmark that
        # improved and GQA did not move. Balance so no domain dominates the gradient.
        by, out = {}, []
        for r in rows:
            g = by.setdefault(r["domain"], [])
            if len(g) < a.cap_per_domain:
                g.append(r)
                out.append(r)
        print("domain caps applied: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(by.items())))
        rows = out

    out = []
    for r in rows:
        at = answer_type(r)
        hops = hops_of(r)
        gold = {**gold_of(r), "hops": hops, "family": r["family"], "domain": r["domain"]}
        # absolute: EasyR1 runs from its own checkout, not from here
        out.append({"images": [str(pathlib.Path(r["image"]).resolve())],
                    # EasyR1 splits the prompt on a literal "<image>" to place the
                    # image; without it vLLM raises "Failed to apply prompt replacement"
                    "problem": "<image>\n" + r["question"] + SUFFIX.format(fmt=FMT[at]),
                    "answer": json.dumps(gold),
                    "hops": hops, "family": r["family"], "domain": r["domain"]})

    # De-dup and drop contradictions BEFORE splitting. Measured on rl6: 123 exact duplicate
    # rows and 8 (image, question) groups carrying two different gold answers.
    seen, by_q, dedup = set(), {}, []
    for r in out:
        k = (r["images"][0], r["problem"])
        ans = json.loads(r["answer"])["answer"]
        by_q.setdefault(k, set()).add(ans)
        if (k, ans) in seen:
            continue
        seen.add((k, ans))
        dedup.append(r)
    conflict = {k for k, v in by_q.items() if len(v) > 1}
    clean = [r for r in dedup if (r["images"][0], r["problem"]) not in conflict]
    print(f"dedup: {len(out)} -> {len(dedup)} (removed {len(out)-len(dedup)} duplicates)")
    print(f"conflicts: dropped {len(dedup)-len(clean)} rows in {len(conflict)} contradictory groups")
    out = clean

    # Split by IMAGE, not by row. Measured on rl6: 565/713 val rows used an image that was
    # also in training, so the internal val signal was inflated.
    imgs = sorted({r["images"][0] for r in out})
    rng.shuffle(imgs)
    n_val_img = max(1, int(len(imgs) * a.val_frac))
    val_imgs = set(imgs[:n_val_img])
    val = [r for r in out if r["images"][0] in val_imgs]
    train = [r for r in out if r["images"][0] not in val_imgs]
    leak = sum(1 for r in val if r["images"][0] in {x["images"][0] for x in train})
    assert leak == 0, f"{leak} val rows share an image with train"
    print(f"image-level split: {len(val_imgs)} val images, 0 leaked")
    train = order_rows(train, a.order, rng)
    print(f"training order: {a.order}")
    d = pathlib.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(d / f"{name}.jsonl", "w") as f:
            for x in part:
                f.write(json.dumps(x) + "\n")
    print(f"train={len(train)}  val={len(val)}")
    for k in ("domain", "hops"):
        c = {}
        for x in train:
            c[x[k]] = c.get(x[k], 0) + 1
        print(f"  by {k}: " + "  ".join(f"{a}={b}" for a, b in sorted(c.items(), key=str)))
    at_c = {}
    for x in train:
        t = x["problem"].rsplit("\n", 1)[-1]
        t = next(k for k, v in FMT.items() if v == t)
        at_c[t] = at_c.get(t, 0) + 1
    print(f"  answer format: {at_c}")
    print(f"WROTE -> {d}/train.jsonl, {d}/val.jsonl")


if __name__ == "__main__":
    main()
