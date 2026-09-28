# VQS — label-free visual question synthesis for RL

Training code for the released checkpoint. A vision-language model writes its own training data
from unlabeled images and is then improved on it with GRPO. No human question, answer or gold label
enters the loop at any stage, and **only the model's own weights are used** — the parser, the
question generator, the verifier and the solver are all the same checkpoint.

## The loop

```
unlabeled image
  │
  ├─ 1. parse        vqs/parse_images.py     image → structured JSON, under a per-domain schema
  │                                          enforced by constrained decoding
  ├─ 2. generate     vqs/gen_questions.py    graph → (question, answer) via 80+ template programs;
  │                                          the answer is COMPUTED by the program, never generated
  ├─ 3. verify       vqs/filter_by_facts.py  each fact the answer depends on is re-asked about the
  │                                          image; the row survives only if all of them check out
  └─ 4. package      vqs/build_rl_data.py    → train.jsonl / val.jsonl for GRPO
                     scripts/train.sh        GRPO with the evidence reward
```

Because step 2 computes the answer from the parse rather than sampling it, the reward has no
model-in-the-loop judge and its false-positive rate is zero by construction. Step 3 exists because
the *parse* can still be wrong.

## Quick start

```bash
pip install -r requirements.txt
git clone https://github.com/hiyouga/EasyR1 && (cd EasyR1 && pip install -e .)

export VQS=$PWD EASYR1=$PWD/EasyR1

# check the 80 template families work before spending GPU time
python vqs/gen_questions.py --selftest

# 1. parse images  → parses.jsonl
python vqs/parse_images.py --manifest data/manifest.jsonl --out data/parses.jsonl

# 2. generate QA   → raw.jsonl
python vqs/gen_questions.py --parses data/parses.jsonl --out data/raw.jsonl

# 3. verify facts  → verified.jsonl
python vqs/filter_by_facts.py --qa data/raw.jsonl --out data/verified.jsonl

# 4. package + train
python vqs/build_rl_data.py --qa data/verified.jsonl --out data/rl_pool
GPUS=0,1 bash scripts/train.sh data/rl_pool vqs_release
```

`data/manifest.jsonl` is one JSON object per line: `{"image": "...", "domain": "...", "id": "..."}`
with `domain` in `charts | infographics | natural | diagrams`.

**Training needs 2 GPUs.** With one rank, FSDP falls back to `NO_SHARD`, whose
`_use_sharded_views()` asserts every parameter view is an `nn.Parameter`; under PEFT the frozen
base weights are plain tensors and the run dies. That is a hard constraint of the stack.

## Released configuration

| | |
|---|---|
| backbone | `Qwen/Qwen3-VL-2B-Instruct` |
| adapter | LoRA r=64, α=128, `all-linear`, `exclude_modules='.*visual.*'` |
| vision tower | frozen |
| algorithm | GRPO, 8 rollouts at T=1.0, KL coef 1e-2 |
| optimiser | lr 1e-5 |
| batches | rollout 256, global 128 |
| steps | 128 |
| reward | `vqs/reward/se_evidence.py` |
| pool | 33,087 train / 728 val rows over 4 domains |

See [`docs/RESULTS.md`](docs/RESULTS.md) for the measured scores and for two findings that matter
if you plan to extend the loop.

## Layout

```
vqs/parse_images.py       stage 1: per-domain JSON schemas + prompts
vqs/gen_questions.py      stage 2: converters + calls into the family modules
vqs/src/                  the template family programs (vendored, unmodified)
vqs/filter_by_facts.py    stage 3: per-fact verification
vqs/build_rl_data.py      stage 4: answer-format suffixes, pool assembly
vqs/reward/se_evidence.py the GRPO reward
scripts/train.sh          the exact released invocation
scripts/eval.sh           lmms-eval over the 10 reported benchmarks
```

Training uses [EasyR1](https://github.com/hiyouga/EasyR1); evaluation uses
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Both are installed separately.
