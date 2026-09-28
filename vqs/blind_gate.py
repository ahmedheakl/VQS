"""Blind gate (paper Sec. 3, Eq. 1; settings in Sec. 4; prompt in A.4.3).

Drops questions that do not need the image. The gate is the parser's own weights with the image
withheld, not a second model. It answers each (paraphrased) question J=4 times, and the question
is dropped when more than a fraction lambda=0.5 of those answers match the computed answer, i.e.
when 3 or 4 of the 4 do. Matching is the reward's eq(y, a).

Usage: CUDA_VISIBLE_DEVICES=7 python vqs/blind_gate.py --qa data/verified.jsonl \
         --out data/gated.jsonl --model <trained parser>
"""
import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_rl_data import gold_of              # noqa: E402
from reward.exact_match import match           # noqa: E402
from vlm import END, TXT, generate             # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
BLIND = ("Answer this question with your best guess. You cannot see the image, so guess from the "
         "wording alone. Reply with the answer only, no explanation.\n\nQuestion: {q}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL, help="the trained parser, run without the image")
    ap.add_argument("--j", type=int, default=4, help="blind answers per question")
    ap.add_argument("--lam", type=float, default=0.5, help="drop if more than this fraction match")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.qa)]
    if a.limit:
        rows = rows[: a.limit]
    assert rows, "no questions"
    print(f"blind-answering {len(rows)} questions, J={a.j}, lambda={a.lam}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.85,
              trust_remote_code=True, max_num_seqs=256)
    sp = SamplingParams(n=a.j, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens,
                        seed=0)
    outs = generate(llm, rows, lambda r: TXT + BLIND.format(q=r["question"]) + END, sp)

    kept, fam_in, fam_out = [], collections.Counter(), collections.Counter()
    for r, o in zip(rows, outs):
        guesses = [c.text.strip() for c in o.outputs]
        hits = sum(1 for g in guesses if g and match(g, gold_of(r)))
        fam_in[r["family"]] += 1
        if hits / len(guesses) <= a.lam:
            kept.append({**r, "blind_guesses": guesses})
            fam_out[r["family"]] += 1
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nKEPT {len(kept)}/{len(rows)} ({len(kept)/len(rows):.1%})")
    print(f"{'family':30s} {'kept':>7s} {'of':>7s} {'rate':>7s}")
    for fam, n in fam_in.most_common(18):
        print(f"{fam:30s} {fam_out[fam]:7d} {n:7d} {fam_out[fam]/n:7.1%}")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
