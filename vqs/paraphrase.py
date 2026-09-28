"""Rewrite each template question in natural wording (paper Sec. 3; prompt in A.4.2).

The parser's own weights do the rewrite, greedily, from the question text alone. The answer is
never shown and never changed. An unconstrained rewrite silently changes what the program
computed, so a rewrite is rejected, and the template wording kept, when it adds any content word
the template question did not contain, does not end in a question mark, is implausibly short or
long, or names the answer.

Usage: CUDA_VISIBLE_DEVICES=7 python vqs/paraphrase.py --qa data/raw.jsonl --out data/para.jsonl \
         --model <trained parser>
"""
import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vlm import END, TXT, generate             # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PARA = ("Rewrite this question so it sounds natural, keeping its meaning exactly the same.\n"
        "Do not answer it. Do not add or remove any condition. Use only the words and entities "
        "already in the question. Reply with only the rewritten question.\n\nQuestion: {q}")

_PUNCT, _WS = re.compile(r"[^\w\s.]"), re.compile(r"\s+")
STOP = set("a an the is are was were do does did you see in of for to on at by with and or not "
           "this that these those it its there here how what which who whom whose when where why "
           "many much more most higher greater less lower highest lowest than value values total "
           "sum ratio percent percentage between from picture image photo chart diagram shown "
           "show shows depicted category categories item items s be been being have has had".split())


def norm(s):
    s = _WS.sub(" ", _PUNCT.sub(" ", str(s).strip().lower())).strip().rstrip(".")
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def words(s):
    return {w for w in norm(s).split() if w and w not in STOP}


def clean(text):
    p = text.strip().split("\n")[0].strip().strip('"')
    return re.sub(r"^(question|rewritten question)\s*[:\-]\s*", "", p, flags=re.I)


def accept(p, row):
    """True if rewrite `p` may replace the template question of `row`."""
    q0 = row["question"]
    gold = [str(row["answer"])] + [str(x) for x in (row.get("acceptable") or [])]
    return not (not p.endswith("?") or len(p) < 8 or len(p) > 2.5 * len(q0)
                or words(p) - words(q0)
                or any(norm(g) and norm(g) in norm(p).split() for g in gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL, help="the trained parser")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.qa)]
    if a.limit:
        rows = rows[: a.limit]
    assert rows, "no questions"
    uniq = sorted({r["question"] for r in rows})     # greedy: one rewrite per distinct question
    print(f"{len(rows)} questions, {len(uniq)} distinct", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.85,
              trust_remote_code=True, max_num_seqs=256)
    outs = generate(llm, uniq, lambda q: TXT + PARA.format(q=q) + END,
                    SamplingParams(temperature=0, max_tokens=64))
    rewrite = {q: clean(o.outputs[0].text) for q, o in zip(uniq, outs)}

    kept = collections.Counter()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            p = rewrite[r["question"]]
            ok = accept(p, r)
            kept[ok] += 1
            f.write(json.dumps({**r, "templated": r["question"],
                                "question": p if ok else r["question"],
                                "paraphrased": ok}) + "\n")
    print(f"rewrite accepted for {kept[True]}/{len(rows)} ({kept[True]/len(rows):.1%}); "
          f"the rest keep the template wording")
    print(f"WROTE {a.out}")


if __name__ == "__main__":
    main()
