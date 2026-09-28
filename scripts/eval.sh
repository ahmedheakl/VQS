#!/usr/bin/env bash
# Evaluation with lmms-eval. The adapter is merged into the base model first and a plain HF
# checkpoint is served, so nothing here depends on LoRA support in the serving stack.
#
#   bash scripts/eval.sh 0 runs/vqs_release/ckpt/global_step_128/actor/huggingface
set -euo pipefail
GPU=$1; MODEL=$2
TASKS="gqa ok_vqa_val2014 infovqa_val scienceqa_img mmmu_val mmbench_en_dev \
embspatial logicvista_reasoning mmstar seedbench"
for t in $TASKS; do
  CUDA_VISIBLE_DEVICES=$GPU python -m lmms_eval \
    --model vllm \
    --model_args model="$MODEL",max_pixels=1003520 \
    --tasks "$t" \
    --batch_size 256 \
    --gen_kwargs max_new_tokens=64,temperature=0 \
    --output_path "runs/eval/$t" --log_samples
done
