"""Label-free parser training data: best-of-K parses chosen by claim-level checks.

Paper Sec. 3 (Eq. 5-6) with the settings of Sec. 4. For each image the parser samples K=4 parses
under the domain schema (temperature 1.0, top-p 0.95, up to 2048 tokens). Every field of a parse
becomes one atomic claim F(z), and the checker (the same weights, greedy, four new tokens) answers
YES or NO to each claim with the image. A parse scores P(z), the fraction of its claims confirmed.
The best parse z* is kept as an SFT target only if

    P(z*) - mean_k P(z^(k)) >= delta (0.10)   and   |F(z*)| >= m (4).

A tie in P is broken toward the parse with more claims. A parse that asserts no claim scores 0.
The kept targets are written in LLaMA-Factory's sharegpt format for configs/parser_sft.yaml, with
exactly the prompt parse_images.py sends at inference.

Usage: CUDA_VISIBLE_DEVICES=7 python vqs/select_parser_targets.py \
         --manifest data/manifest.jsonl --out data/parser_sft
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from filter_by_facts import Q                            # noqa: E402  (the checker's prompt)
from parse_images import MODEL, PROMPTS, SCHEMAS         # noqa: E402
from vlm import END, IMG, generate                       # noqa: E402

LF_INFO = {"parser_sft": {"file_name": "parser_sft.json", "formatting": "sharegpt",
                          "columns": {"messages": "messages", "images": "images"},
                          "tags": {"role_tag": "role", "content_tag": "content",
                                   "user_tag": "user", "assistant_tag": "assistant",
                                   "system_tag": "system"}}}


def claims_of_parse(parse, domain):
    """F(z): one checkable statement per field the parse asserts."""
    out = []
    if domain == "charts":
        if parse.get("title"):
            out.append(f'The title of this chart is "{parse["title"]}".')
        for s in parse.get("series", []):
            if s.get("series_name"):
                out.append(f'This chart has a series labelled "{s["series_name"]}".')
            for p in s.get("points", []):
                c, v = p.get("category", ""), p.get("printed_value", "")
                if c and v:
                    out.append(f'In this chart, the value for "{c}" is {v}.')
    elif domain == "infographics":
        if parse.get("title"):
            out.append(f'The title of this infographic is "{parse["title"]}".')
        for e in parse.get("entries", []):
            l, v = e.get("entry_label", ""), e.get("printed_value", "")
            if l and v:
                out.append(f'This infographic states that "{l}" is {v}.')
    elif domain == "natural":
        for o in parse.get("objects", []):
            n = o.get("object_name", "")
            if not n:
                continue
            out.append(f"There is a {n} in this image.")
            for att in o.get("attributes") or []:
                out.append(f"The {n} in this image is {att}.")
        names = {o.get("object_id"): o.get("object_name", "") for o in parse.get("objects", [])}
        for r in parse.get("relations", []):
            s, p, ob = names.get(r.get("subject_id")), r.get("predicate"), names.get(r.get("object_id"))
            if s and p and ob:
                out.append(f"In this image, the {s} is {p} the {ob}.")
    else:  # diagrams
        for n in parse.get("nodes", []):
            if n.get("node_label"):
                out.append(f'This diagram has a part labelled "{n["node_label"]}".')
        lab = {n["node_id"]: n.get("node_label", "") for n in parse.get("nodes", [])}
        for e in parse.get("edges", []):
            s, t = lab.get(e.get("from_node_id")), lab.get(e.get("to_node_id"))
            if s and t:
                out.append(f'In this diagram, an arrow goes from "{s}" to "{t}".')
    return out


def select(scored, delta, m):
    """Eq. (6). scored: [(P, n_claims, idx)] for one image's samples -> idx of z*, or None."""
    p_best, n_best, i_best = max(scored)
    mean = sum(p for p, _, _ in scored) / len(scored)
    return i_best if p_best - mean >= delta and n_best >= m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--min-claims", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    if a.limit:
        rows = rows[: a.limit]
    assert rows, "empty manifest"
    print(f"{len(rows)} images, K={a.k}", flush=True)

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(model=a.model, max_model_len=16384, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, max_num_seqs=64,
              mm_processor_kwargs={"max_pixels": 1003520})

    # ---- K parses per image, one domain (= one schema) at a time ----
    cand = []   # (row, raw_text, claims)
    for dom in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == dom]
        sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                            max_tokens=a.max_tokens, seed=0,
                            structured_outputs=StructuredOutputsParams(json=SCHEMAS[dom]))
        outs = generate(llm, sub, lambda r: {
            "prompt": IMG + PROMPTS[dom] + END,
            "multi_modal_data": {"image": Image.open(r["image"]).convert("RGB")}}, sp)
        n_trunc = 0
        for r, o in zip(sub, outs):
            for c in o.outputs:
                # the schema guarantees a valid PREFIX; a sample cut at the token cap is not a parse
                if c.finish_reason == "length":
                    n_trunc += 1
                    continue
                cand.append((r, c.text, claims_of_parse(json.loads(c.text), dom)))
        print(f"  {dom}: {len(sub)} images, {n_trunc} of {len(sub) * a.k} samples truncated",
              flush=True)
    assert cand, "no parses survived"

    # ---- every claim of every candidate, checked one at a time ----
    items = [(ci, c) for ci, (_, _, cs) in enumerate(cand) for c in cs]
    print(f"checking {len(items)} claims", flush=True)
    outs = generate(llm, items, lambda ic: {
        "prompt": IMG + Q.format(claim=ic[1]) + END,
        "multi_modal_data": {"image": Image.open(cand[ic[0]][0]["image"]).convert("RGB")}},
        SamplingParams(temperature=0, max_tokens=4))
    yes = collections.Counter()
    for (ci, _), o in zip(items, outs):
        yes[ci] += o.outputs[0].text.strip().upper().startswith("Y")

    # ---- Eq. (5)-(6) per image ----
    by_img = collections.defaultdict(list)
    for ci, (r, _, cs) in enumerate(cand):
        by_img[r["image"]].append((yes[ci] / len(cs) if cs else 0.0, len(cs), ci))
    kept = []
    for scored in by_img.values():
        ci = select(scored, a.delta, a.min_claims)
        if ci is None:
            continue
        r, txt, cs = cand[ci]
        mean = sum(p for p, _, _ in scored) / len(scored)
        kept.append({"image": str(pathlib.Path(r["image"]).resolve()), "domain": r["domain"],
                     "prompt": PROMPTS[r["domain"]], "target": txt, "precision": yes[ci] / len(cs),
                     "sample_mean": mean, "n_claims": len(cs)})

    d = pathlib.Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "targets.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    json.dump([{"messages": [{"role": "user", "content": "<image>" + r["prompt"]},
                             {"role": "assistant", "content": r["target"]}],
                "images": [r["image"]]} for r in kept], open(d / "parser_sft.json", "w"))
    json.dump(LF_INFO, open(d / "dataset_info.json", "w"), indent=1)
    print(f"\nKEPT {len(kept)}/{len(by_img)} images as parser targets "
          f"(delta={a.delta}, m={a.min_claims})")
    for dom in sorted({r["domain"] for r in kept}):
        ks = [r for r in kept if r["domain"] == dom]
        print(f"  {dom:14s} {len(ks):6d} targets, mean precision "
              f"{sum(r['precision'] for r in ks) / len(ks):.1%}")
    print(f"WROTE {d}/targets.jsonl, {d}/parser_sft.json, {d}/dataset_info.json")


if __name__ == "__main__":
    main()
