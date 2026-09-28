# Measured results

Qwen3-VL-2B-Instruct, evaluated with lmms-eval. Every number here was produced by
`scripts/train.sh` and `scripts/eval.sh` as they were at commit `0c134cf` (128 GRPO steps;
evaluation with `max_new_tokens=64, temperature=0` on every benchmark); nothing is interpolated or
estimated. The scripts now use the paper's settings (96 steps, each benchmark's default
generation settings), so re-running them is not expected to reproduce this table exactly.

| Run | GQA | OK-VQA | InfoVQA | SQA | MMMU | MMB | ESB | LogicV | MMStar | SEED | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Base (no training) | 59.43 | 40.51 | 72.18 | 86.02 | 45.44 | 75.77 | 68.57 | 38.39 | 57.43 | 72.20 | **61.60** |
| VQS, seed 1 | 59.64 | 41.20 | 71.36 | 87.21 | 45.11 | 77.41 | 71.54 | 42.19 | 56.32 | 72.82 | **62.48** |
| VQS, seed 2 | 59.72 | 42.86 | 70.91 | 86.76 | 45.44 | 78.44 | 72.86 | 38.39 | 56.72 | 72.84 | **62.50** |

- **seed 1**: 62.48 (+0.88 over base)
- **seed 2**: 62.50 (+0.90 over base)

Over the full 17-benchmark suite the same two runs score 62.10 and 62.22 against a base of 61.33.

## On the size of the gain

The base row above is **our own evaluation** of the stock checkpoint, run through the same
lmms-eval harness, same task versions and same decoding settings as the trained runs. That is the
comparison this repository supports, and it gives **+0.88**.

Published tables for this benchmark suite often quote a base row taken from prior work rather than
re-measured locally, and those numbers are lower on several tasks (notably ScienceQA and MMMU).
Measured against such a row the same checkpoint appears to gain roughly +3, but most of that
difference is harness-to-harness variation in the *baseline*, not an effect of training. We report
the same-harness number here deliberately; if you are comparing against a published baseline,
re-measure it with `scripts/eval.sh` first.

## Two findings worth knowing before you extend this

**1. Most of the gain arrives in the first third of training.** Evaluating intermediate
checkpoints of a 96-step run, 86% of the final gain is already present at step 32 (average
61.96 vs 62.39 at step 96), and three of ten benchmarks *peak* at step 32 and end lower. If you
are compute-bound, a ~32-step run captures nearly all of the benefit.

**2. A second cycle adds nothing, and carrying weights forward costs accuracy.** Re-probing the
pool with the trained checkpoint, rebuilding the curriculum around it and retraining from the base
model produced a change of +0.00. Initialising the second cycle *from the previous checkpoint*
instead scored 1.25 below a single cycle. That deficit is one benchmark: excluding LogicVista, the
two regimes are within 0.2 of each other at every checkpoint, and LogicVista is the only benchmark
in the suite that requires a long-form `<answer>`-tagged response. Its well-formed-tag rate falls
from 74.8% to 57.6% over those steps, tracking the score almost exactly. The failure is
instruction-following on long answers, not reasoning.

Both effects were measured on one seed per arm. Per-benchmark seed standard deviations, estimated
from nine paired-seed arms, are: GQA 0.27, OK-VQA 0.51, InfoVQA 0.35, SQA 0.19, MMMU 0.52,
MMB 0.43, ESB 0.40, LogicVista 1.77, MMStar 0.54, SEED 0.11.
