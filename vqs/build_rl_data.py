"""Turn gated questions into EasyR1 training data.

Every question becomes TWO rows:
  * scaffold row -- prompt asks for <see> evidence then <answer>; scored on answer + evidence F1
  * plain row    -- the exact prompt style used at eval; scored on the answer alone

The smoke test measured that making the model emit evidence at eval time costs 4.84 mean
points, so evidence is a training-time auxiliary only. Keeping plain rows in the mixture means
the eval-time behaviour is directly optimised, not merely hoped for.

Usage: python scripts/build_rl_data.py --qa runs/qa/gated.jsonl --out runs/rl/ [--scaffold-frac 0.5]
"""
import argparse
import json
import pathlib
import random

SCAFFOLD_SUFFIX = ("\n{fmt}\nReply in exactly this format:\n<see>\nlabel: value\nlabel: value\n"
                   "</see>\n<answer>your answer</answer>")
PLAIN_SUFFIX = "\n{fmt}"
FMT = {"number": "Answer the question using a single number.",
       "word": "Answer the question using a single word or phrase."}

NUMERIC_FAMS = {"chart_count", "chart_diff", "chart_sum", "diag_count_nodes", "diag_degree",
                "nat_count", "info_count", "info_diff"}


def answer_type(row):
    if row["family"] in NUMERIC_FAMS:
        return "number"
    a = row["answer"].replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        float(a)
        return "number"
    except ValueError:
        return "word"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaffold-frac", type=float, default=0.5)
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
        rng.shuffle(rows)

    out = []
    for i, r in enumerate(rows):
        at = answer_type(r)
        # sg_gen families carry `difficulty` and their own `acceptable`; my older pool
        # carried `evidence`/`hops`. Accept both so one builder serves both generations.
        hops = r.get("hops", r.get("difficulty", 2))
        acc = r.get("acceptable") or [r["answer"]]
        gold = {"answer": r["answer"], "answer_type": r.get("answer_type", at),
                "acceptable": [str(x) for x in acc],
                "evidence": r.get("evidence", []), "hops": hops,
                "family": r["family"], "domain": r["domain"]}
        scaffold = (i % 100) < int(a.scaffold_frac * 100)
        suffix = (SCAFFOLD_SUFFIX if scaffold else PLAIN_SUFFIX).format(fmt=FMT[at])
        # absolute: EasyR1 runs from its own checkout, not from here
        out.append({"images": [str(pathlib.Path(r["image"]).resolve())],
                    # EasyR1 splits the prompt on a literal "<image>" to place the
                    # image; without it vLLM raises "Failed to apply prompt replacement"
                    "problem": "<image>\n" + r["question"] + suffix,
                    "answer": json.dumps({**gold, "scaffold": scaffold}),
                    "hops": hops, "family": r["family"], "domain": r["domain"],
                    "scaffold": scaffold})

    # De-dup and drop contradictions BEFORE splitting. Measured on rl6: 123 exact duplicate
    # rows and 8 (image, question) groups carrying two different gold answers.
    seen, by_q, dedup = set(), {}, []
    for r in out:
        k = (r["images"][0], r["problem"])
        ans = json.loads(r["answer"])["answer"]
        by_q.setdefault(k, set()).add(ans)
        sig = (k, ans, r["scaffold"])
        if sig in seen:
            continue
        seen.add(sig)
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
    d = pathlib.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(d / f"{name}.jsonl", "w") as f:
            for x in part:
                f.write(json.dumps(x) + "\n")
    print(f"train={len(train)}  val={len(val)}")
    print(f"  scaffold rows {sum(x['scaffold'] for x in train)} "
          f"({sum(x['scaffold'] for x in train)/len(train):.1%})")
    for k in ("domain", "hops"):
        c = {}
        for x in train:
            c[x[k]] = c.get(x[k], 0) + 1
        print(f"  by {k}: " + "  ".join(f"{a}={b}" for a, b in sorted(c.items(), key=str)))
    at_c = {}
    for x in train:
        t = json.loads(x["answer"])["answer_type"]
        at_c[t] = at_c.get(t, 0) + 1
    print(f"  answer_type: {at_c}")
    print(f"WROTE -> {d}/train.jsonl, {d}/val.jsonl")


if __name__ == "__main__":
    main()
