# VQS — label-free visual question synthesis for RL

Code for *Program-Verified Self-Evolution for Vision-Language Models*. A vision-language model
writes its own training data from unlabeled images and is then improved on it with GRPO. No human
question, answer or gold label enters the loop at any stage, and **only the model's own weights are
used**: the parser, the paraphraser, the checker, the blind gate and the solver are all the same
checkpoint.

## The loop

One training cycle (Algorithm 1 in the paper):

```
unlabeled image
  │
  ├─ 0. train parser   vqs/select_parser_targets.py  K=4 schema-constrained parses per image; the
  │                                                  best by verified-claim fraction becomes a target
  │                    configs/parser_sft.yaml       SFT on those targets (LLaMA-Factory)
  ├─ 1. parse          vqs/parse_images.py     image → structured JSON under a per-domain schema
  │                                            enforced by constrained decoding (trained parser)
  ├─ 2. generate       vqs/gen_questions.py    parse → (question, answer, hops) via template
  │                                            programs; the answer is COMPUTED, never generated
  │                    vqs/paraphrase.py       the parser rewrites the question in natural wording
  ├─ 3. filter         vqs/filter_by_facts.py  every fact the answer depends on is re-asked about
  │                                            the image; the row survives only if all check out
  │                    vqs/blind_gate.py       drop questions answerable without the image
  │                    vqs/build_rl_data.py    answer-format suffix, curriculum order, train/val
  │                    vqs/difficulty.py band  keep questions the solver gets right sometimes
  └─ 4. train          scripts/train.sh        GRPO with the exact-match reward
```

Because step 2 computes the answer from the parse rather than sampling it, the reward has no
model-in-the-loop judge and its false-positive rate is zero by construction. Steps 0 and 3 exist
because the *parse* can still be wrong.

## Quick start

```bash
pip install -r requirements.txt
git clone https://github.com/hiyouga/EasyR1 && (cd EasyR1 && pip install -e .)
git clone https://github.com/hiyouga/LLaMA-Factory && (cd LLaMA-Factory && pip install -e .)

export VQS=$PWD EASYR1=$PWD/EasyR1

# check the template families and the reward before spending GPU time
python vqs/gen_questions.py --selftest
python vqs/reward/exact_match.py

# 0. train the parser: best-of-K targets, then SFT      → runs/parser_sft
python vqs/select_parser_targets.py --manifest data/manifest.jsonl --out data/parser_sft
llamafactory-cli train configs/parser_sft.yaml
P=runs/parser_sft

# 1. parse with the trained parser                     → parses.jsonl
python vqs/parse_images.py --manifest data/manifest.jsonl --out data/parses.jsonl --model $P

# 2. generate and paraphrase                           → raw.jsonl, para.jsonl
python vqs/gen_questions.py --parses data/parses.jsonl --out data/raw.jsonl
python vqs/paraphrase.py --qa data/raw.jsonl --out data/para.jsonl --model $P

# 3. fact-check, blind gate, package, difficulty band  → rl_band/{train,val}.jsonl
python vqs/filter_by_facts.py --qa data/para.jsonl --out data/verified.jsonl --model $P
python vqs/blind_gate.py --qa data/verified.jsonl --out data/gated.jsonl --model $P
python vqs/build_rl_data.py --qa data/gated.jsonl --out data/rl_pool
python vqs/difficulty.py band --data data/rl_pool --out data/rl_band

# 4. train, merge the adapter, evaluate
GPUS=0,1 bash scripts/train.sh data/rl_band vqs_2b
python $EASYR1/scripts/model_merger.py --local_dir runs/vqs_2b/ckpt/global_step_96/actor
bash scripts/eval.sh 0 runs/vqs_2b/ckpt/global_step_96/actor/huggingface
```

`data/manifest.jsonl` is one JSON object per line: `{"image": "...", "domain": "...", "id": "..."}`
with `domain` in `charts | infographics | natural | diagrams`.

The commands above are for Qwen3-VL-2B. For 4B or 8B, pass that base checkpoint to every `--model`
(and to `model_name_or_path` in `configs/parser_sft.yaml`) and set `MODEL` for `scripts/train.sh`;
parser and solver always start from the same pretrained weights.

**Training needs 2 GPUs.** With one rank, FSDP falls back to `NO_SHARD`, whose
`_use_sharded_views()` asserts every parameter view is an `nn.Parameter`; under PEFT the frozen
base weights are plain tensors and the run dies. That is a hard constraint of the stack. The paper's
runs used 8 AMD MI210 GPUs.

## Hyperparameters

Every setting the paper reports, and where it is set.

**Data (Sec. 3–4)**

| stage | setting | where |
|---|---|---|
| parser targets | K=4 parses per image, temperature 1.0, top-p 0.95, ≤2048 tokens, schema-constrained | `vqs/select_parser_targets.py` |
| | keep the best parse z\* only if P(z\*) − mean P ≥ δ=0.10 and z\* asserts ≥ m=4 claims | `vqs/select_parser_targets.py` |
| parser SFT | full fine-tune, vision tower frozen, per-device batch 4 × 8 accumulation steps, lr 1e-5, 2 epochs | `configs/parser_sft.yaml` |
| parsing | greedy, ≤2048 tokens, schema-constrained | `vqs/parse_images.py` |
| paraphrase | the parser, greedy; the template wording is kept if the rewrite adds a content word | `vqs/paraphrase.py` |
| fact-check | the parser answers YES/NO to one claim at a time, greedy, 4 new tokens; every claim must hold | `vqs/filter_by_facts.py` |
| blind gate | the parser without the image answers J=4 times; drop if more than λ=0.5 match | `vqs/blind_gate.py` |
| difficulty band | the base solver at training settings: 8 rollouts, temperature 1.0, 256 tokens; keep 0 < p̂ < 1 | `vqs/difficulty.py band` |
| curriculum | each template family's questions ordered by hop count, fewest first | `vqs/build_rl_data.py` |
| answer format | single word or phrase / single number / option's letter, by answer type | `vqs/build_rl_data.py` |

**Solver GRPO (Sec. A.1, Table 7)**, all in `scripts/train.sh`

| | |
|---|---|
| backbone | `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct` (`MODEL`, default 2B) |
| adapter | LoRA r=64, α=128, all linear layers, `.*visual.*` excluded (0 trainable visual parameters) |
| gradient checkpointing | on |
| optimiser | AdamW, betas (0.9, 0.999), weight decay 0.01, lr 1e-5 constant |
| KL penalty | low-variance estimator, β=0.01, frozen reference |
| clipping | ratio 0.2 (low) / 0.3 (high), dual clip 3.0 |
| PPO epochs per step | 1 |
| dynamic sampling | off |
| rollouts | 8 per prompt, temperature 1.0, top-p 1.0 |
| budgets | prompt 2048 tokens, response 256 tokens, images ≤ 1,003,520 pixels |
| batches | rollout 256 prompts, global update 128; micro-batch per device 16 at 2B, 4 at 8B |
| vLLM memory | 0.60 |
| steps | 96 |
| reward | Eq. (2): 0.9 · exact match with the computed answer + 0.1 · format, `vqs/reward/exact_match.py` |

**Evaluation (Sec. 4)**: lmms-eval on the 10 benchmarks, each with its default generation settings
(`scripts/eval.sh`).

## Later cycles

Each new cycle (Sec. 5) starts from the previous solver with its LoRA merged, writes new questions
from the templates on the same images, and adds them to the earlier cycles' questions. The merged
solver answers every question 8 times; with s its fraction correct, each question is weighted by
w(s) = exp(−((s − 0.5) / 0.15)²), with w = 0 at s = 0 and s = 1, and 8,000 training rows are drawn
with these weights. The solver then trains 96 GRPO steps from the previous cycle with the same
settings.

```bash
PREV=runs/vqs_2b/ckpt/global_step_96/actor        # merged above: $PREV/huggingface
python vqs/gen_questions.py --parses data/parses.jsonl --out data/c2/raw.jsonl --seed 2
python vqs/paraphrase.py --qa data/c2/raw.jsonl --out data/c2/para.jsonl --model $P
python vqs/filter_by_facts.py --qa data/c2/para.jsonl --out data/c2/verified.jsonl --model $P
python vqs/blind_gate.py --qa data/c2/verified.jsonl --out data/c2/gated.jsonl --model $P
python vqs/build_rl_data.py --qa data/c2/gated.jsonl --out data/c2/rl_pool
python vqs/difficulty.py resample --data data/rl_band data/c2/rl_pool --out data/c2/rl_train \
    --model $PREV/huggingface
GPUS=0,1 MODEL=$PREV/huggingface bash scripts/train.sh data/c2/rl_train vqs_2b_c2
```

A third cycle adds `data/c3/rl_pool` to `--data` and starts from the second cycle's merged solver.

See [`docs/RESULTS.md`](docs/RESULTS.md) for measured scores of an earlier configuration of this
code and for two findings that matter if you plan to extend the loop.

## Layout

```
vqs/select_parser_targets.py  stage 0: best-of-K parser targets (Eq. 5-6)
configs/parser_sft.yaml       stage 0: parser SFT with LLaMA-Factory
vqs/parse_images.py           stage 1: per-domain JSON schemas + prompts
vqs/gen_questions.py          stage 2: converters + calls into the family modules
vqs/src/                      the template family programs (vendored, unmodified)
vqs/paraphrase.py             stage 2: guarded rewrite of the template question
vqs/filter_by_facts.py        stage 3: per-fact verification
vqs/blind_gate.py             stage 3: questions answerable without the image are dropped
vqs/build_rl_data.py          stage 3: answer-format suffixes, curriculum order, pool assembly
vqs/difficulty.py             stage 3: difficulty band; resampling for later cycles
vqs/vlm.py                    shared vLLM prompt wrappers and chunked generation
vqs/reward/exact_match.py     the GRPO reward (Eq. 2)
scripts/train.sh              GRPO with the settings of Table 7
scripts/eval.sh               lmms-eval over the 10 reported benchmarks
```

Training uses [EasyR1](https://github.com/hiyouga/EasyR1), parser SFT uses
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and evaluation uses
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). All three are installed separately.
