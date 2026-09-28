#!/usr/bin/env bash
# Evaluation with lmms-eval on the paper's 10 benchmarks. The adapter is merged into the base model
# first and a plain HF checkpoint is served, so nothing here depends on LoRA support in the serving
# stack.
#
# Each benchmark runs with its own lmms-eval generation settings (paper Sec. 4: "the default
# hyperparameters for each benchmark"), so no --gen_kwargs override is passed. Images are capped at
# the training resolution.
#
#   python $EASYR1/scripts/model_merger.py --local_dir runs/vqs_2b/ckpt/global_step_96/actor
#   OUT=runs/eval/vqs_2b bash scripts/eval.sh 0 runs/vqs_2b/ckpt/global_step_96/actor/huggingface
#   OUT=runs/eval/base_2b bash scripts/eval.sh 1 Qwen/Qwen3-VL-2B-Instruct
#
# TASKS overrides the benchmark list, e.g. to split the suite across GPUs.
set -euo pipefail
GPU=$1; MODEL=$2
OUT=${OUT:-runs/eval}
TASKS=${TASKS:-"gqa ok_vqa_val2014 infovqa_val scienceqa_img mmmu_val mmbench_en_dev \
embspatial logicvista_reasoning mmstar seedbench"}
for t in $TASKS; do
  CUDA_VISIBLE_DEVICES=$GPU python -m lmms_eval \
    --model vllm \
    --model_args model="$MODEL",max_pixels=1003520 \
    --tasks "$t" \
    --batch_size 256 \
    --output_path "$OUT/$t" --log_samples
done
