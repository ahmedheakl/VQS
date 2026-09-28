"""Solve-rate probes of the solver, used twice in the paper.

band      The difficulty band (paper Sec. 3, Eq. 1; settings in Sec. 4). The solver about to be
          trained answers every question n=8 times at the training settings (the training prompt,
          temperature 1.0, top-p 1.0, 256 new tokens, the same image resize), and a question is
          kept only when 0 < p < 1: a group that is all right or all wrong gives every rollout the
          same advantage and so no gradient. Row order is preserved, so the curriculum that
          build_rl_data.py wrote survives. val.jsonl is copied unchanged.

resample  Later training cycles (paper Sec. 5). The previous cycle's merged solver answers every
          question of the combined pools 8 times; s is its fraction correct. Each question gets
          weight w(s) = exp(-((s - 0.5) / 0.15)^2), and w = 0 when s is 0 or 1. 8,000 training rows
          are drawn with these weights and put in curriculum order. val.jsonl of the first pool
          is copied, and rows on its images are left out of the draw.

Usage:
  python vqs/difficulty.py band --data data/rl_pool --out data/rl_band
  python vqs/difficulty.py resample --data data/rl_band data/c2/rl_pool --out data/c2/rl_train \
      --model runs/vqs_c1/ckpt/global_step_96/actor/huggingface
"""
import argparse
import functools
import json
import math
import pathlib
import random
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_rl_data import order_rows                    # noqa: E402
from reward.exact_match import extract_answer, match    # noqa: E402
from vlm import END, IMG, generate                      # noqa: E402

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
MIN_PIXELS, MAX_PIXELS = 262144, 1003520         # scripts/train.sh data.min/max_pixels
TARGET, WIDTH = 0.5, 0.15


@functools.lru_cache(maxsize=512)
def load_image(path):
    """EasyR1's resize (verl/utils/dataset.py:process_image), so the probe sees training pixels."""
    from PIL import Image
    im = Image.open(path)
    im.load()
    for lim, grow in ((MAX_PIXELS, False), (MIN_PIXELS, True)):
        if (im.width * im.height < lim) if grow else (im.width * im.height > lim):
            f = math.sqrt(lim / (im.width * im.height))
            im = im.resize((int(im.width * f), int(im.height * f)))
    return im.convert("RGB")


def solve_rates(rows, model, n):
    """Fraction of n sampled answers that match the computed answer, per row."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, max_num_seqs=128)
    sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=256, seed=0)
    # probe the rows image by image, so each image is decoded once; rates come back in row order
    idx = sorted(range(len(rows)), key=lambda i: rows[i]["images"][0])
    got = generate(llm, [rows[i] for i in idx], lambda r: {
        "prompt": IMG + r["problem"].replace("<image>", "", 1) + END,
        "multi_modal_data": {"image": load_image(r["images"][0])}}, sp)
    outs = [None] * len(rows)
    for i, o in zip(idx, got):
        outs[i] = o
    rates = []
    for r, o in zip(rows, outs):
        gold = json.loads(r["answer"])
        hits = sum(1 for c in o.outputs
                   if (y := extract_answer(c.text)) and match(y, gold))
        rates.append(hits / len(o.outputs))
    return rates


def weight(s):
    return 0.0 if s <= 0.0 or s >= 1.0 else math.exp(-((s - TARGET) / WIDTH) ** 2)


def write(out, train, val_src):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    shutil.copy(pathlib.Path(val_src) / "val.jsonl", d / "val.jsonl")
    print(f"WROTE {len(train)} rows -> {d}/train.jsonl (val copied from {val_src})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["band", "resample"])
    ap.add_argument("--data", nargs="+", required=True, help="pool dir(s) with train/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL, help="band: the base solver; resample: the "
                                                   "previous cycle's merged solver")
    ap.add_argument("--n", type=int, default=8, help="sampled answers per question")
    ap.add_argument("--size", type=int, default=8000, help="resample: rows to draw")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    assert a.mode == "resample" or len(a.data) == 1, "band filters one pool"
    # val.jsonl comes from the first pool; a later cycle's split may have put those images in train
    val_imgs = {json.loads(l)["images"][0] for l in open(pathlib.Path(a.data[0]) / "val.jsonl")}
    rows, seen = [], set()
    for d in a.data:
        for l in open(pathlib.Path(d) / "train.jsonl"):
            r = json.loads(l)
            k = (r["images"][0], r["problem"])
            if k not in seen and r["images"][0] not in val_imgs:   # a later cycle may repeat one
                seen.add(k)
                rows.append(r)
    assert rows, "no training rows"
    print(f"probing {len(rows)} questions x {a.n} with {a.model}", flush=True)
    rates = solve_rates(rows, a.model, a.n)
    n_dead = sum(1 for s in rates if s in (0.0, 1.0))
    print(f"never solved {sum(1 for s in rates if s == 0.0)}, always solved "
          f"{sum(1 for s in rates if s == 1.0)}, in between {len(rows) - n_dead}")

    if a.mode == "band":
        write(a.out, [r for r, s in zip(rows, rates) if 0.0 < s < 1.0], a.data[0])
        return
    w = [weight(s) for s in rates]
    assert sum(w) > 0, "every question is always or never solved"
    rng = random.Random(a.seed)
    drawn = rng.choices(rows, weights=w, k=a.size)
    print(f"drew {a.size} rows ({len({id(r) for r in drawn})} distinct) with w(s)")
    write(a.out, order_rows(drawn, "family_hops", rng), a.data[0])


if __name__ == "__main__":
    main()
