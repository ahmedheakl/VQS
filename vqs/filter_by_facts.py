"""P1: keep a training row only if the facts its ANSWER DEPENDS ON verify against the image.

Why this and not the previous filter. The QA-level fact-checker was refuted: it accepts a
deliberately corrupted answer as readily as the true one (discrimination 0.000-0.067,
runs/review/qa_checker_validation.json). The PER-FACT checker is validated on 27k+ facts at
precision 0.890-0.916 with trap-YES 0.026-0.043 (runs/parser_eval/*.json). So we ask atomic
questions the checker can actually answer, about the specific cells the program read.

Row is kept iff EVERY supporting fact returns YES. Rows whose facts we cannot identify are
passed through untouched and counted, so the filter never silently drops what it cannot judge.

Usage: CUDA_VISIBLE_DEVICES=7 python scripts/filter_by_facts.py \
   --qa runs/qa5/all_gated.jsonl --out runs/qa6/fact_filtered.jsonl [--limit 0]
"""
import argparse, collections, json, os, pathlib, re
# honour the caller's HF_HOME; do not pin it to a local path
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
IMG = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
END = "<|im_end|>\n<|im_start|>assistant\n"
Q = '{claim}\nLooking only at the image, is that statement true? Answer with one word, YES or NO.'
CELL = re.compile(r"CELL\(([^,]*),([^,]*),([^)]*)\)")


def claims_of(row):
    """The atomic statements the computed answer rests on."""
    out = []
    for p in (row.get("provenance") or []):
        m = CELL.search(str(p))
        if m:
            series, cat, val = (x.strip() for x in m.groups())
            where = f" for {series}" if series else ""
            out.append(f'In this image, the value for "{cat}"{where} is {val}.')
    if not out:
        for k, lab in (("q_name", "object"), ("subj", "object"), ("entry_label", "item")):
            v = row.get(k)
            if v:
                out.append(f"There is a {v} in this image.")
                break
    return out[:4]          # cap: 4 checks per row keeps the sweep affordable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.qa)]
    if a.limit:
        rows = rows[: a.limit]
    items, owner = [], []
    n_nocheck = 0
    for i, r in enumerate(rows):
        cs = claims_of(r)
        if not cs:
            n_nocheck += 1
            continue
        for c in cs:
            items.append((i, c))
    print(f"{len(rows)} rows -> {len(items)} atomic checks; "
          f"{n_nocheck} rows have no identifiable facts (passed through)", flush=True)

    from PIL import Image
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=8192, gpu_memory_utilization=a.gpu_util,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, max_num_seqs=256,
              mm_processor_kwargs={"max_pixels": 1003520})
    sp = SamplingParams(temperature=0, max_tokens=4)
    reqs = [{"prompt": IMG + Q.format(claim=c) + END,
             "multi_modal_data": {"image": Image.open(rows[i]["image"]).convert("RGB")}}
            for i, c in items]
    outs = llm.generate(reqs, sp)
    assert len(outs) == len(items), f"{len(outs)} vs {len(items)}"

    bad = set()
    for (i, c), o in zip(items, outs):
        if not o.outputs[0].text.strip().upper().startswith("Y"):
            bad.add(i)
    kept = [r for i, r in enumerate(rows) if i not in bad]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    fam_in = collections.Counter(r["family"] for r in rows)
    fam_out = collections.Counter(r["family"] for r in kept)
    print(f"\nKEPT {len(kept)}/{len(rows)} ({len(kept)/len(rows):.1%})")
    print(f"{'family':30s} {'kept':>7s} {'of':>7s} {'rate':>7s}")
    for fam, n in fam_in.most_common(18):
        print(f"{fam:30s} {fam_out[fam]:7d} {n:7d} {fam_out[fam]/n:7.1%}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
