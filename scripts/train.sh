#!/usr/bin/env bash
# GRPO training of the solver with the settings of the paper (Sec. A.1, Table 7).
#
# Every Table 7 value is passed explicitly instead of being inherited from EasyR1's defaults, so a
# different EasyR1 version cannot change the run silently. The 2B, 4B and 8B runs share these
# settings; only the per-device micro-batch changes (Table 7: 16 at 2B, 4 at 8B), which sets the
# gradient-accumulation granularity, not the batch. Set MICRO_BATCH to override it.
#
# The training rows are consumed in the curriculum order build_rl_data.py wrote (data.shuffle=false).
# A later cycle starts from the previous cycle's merged solver: MODEL=<ckpt>/actor/huggingface.
#
# Needs 2 GPUs. Single-rank FSDP falls back to NO_SHARD, whose _use_sharded_views() asserts every
# parameter view is an nn.Parameter; under PEFT the frozen base weights are plain Tensors and it
# dies. This is a real constraint, not a preference.
#
#   GPUS=0,1 bash scripts/train.sh data/rl_band vqs_2b
#   GPUS=0,1 MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/train.sh data/rl_band vqs_8b
set -euo pipefail
DATA=$(cd "$1" && pwd); EXP=$2     # absolute: EasyR1 runs from its own checkout
: "${GPUS:?set GPUS, e.g. GPUS=0,1}"
: "${EASYR1:?set EASYR1 to your EasyR1 checkout}"
: "${VQS:?set VQS to this repository root}"
VQS=$(cd "$VQS" && pwd)
MODEL=${MODEL:-Qwen/Qwen3-VL-2B-Instruct}
# a local checkpoint must be absolute, or transformers reads it as a hub id after the cd below
[ -d "$MODEL" ] && MODEL=$(cd "$MODEL" && pwd)
case "$MODEL" in
  *8B*) MB=${MICRO_BATCH:-4} ;;
  *)    MB=${MICRO_BATCH:-16} ;;
esac
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=${OUT:-runs/$EXP}; mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export CUDA_VISIBLE_DEVICES=$GPUS
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false

cd "$EASYR1"
python -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files="$DATA/train.jsonl@train" \
  data.val_files="$DATA/val.jsonl@train" \
  data.format_prompt="$VQS/vqs/se_passthrough.jinja" \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.min_pixels=262144 \
  data.max_pixels=1003520 \
  data.rollout_batch_size=256 \
  data.shuffle=false \
  data.seed="${SEED:-1}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_loss=true \
  algorithm.kl_penalty=low_var_kl \
  algorithm.kl_coef=1.0e-2 \
  algorithm.online_filtering=false \
  worker.rollout.n=8 \
  worker.rollout.temperature=1.0 \
  worker.rollout.top_p=1.0 \
  worker.rollout.tensor_parallel_size=$NG \
  worker.rollout.gpu_memory_utilization=0.60 \
  worker.actor.model.model_path="$MODEL" \
  worker.actor.model.freeze_vision_tower=true \
  worker.actor.model.enable_gradient_checkpointing=true \
  worker.actor.model.lora.rank=64 \
  worker.actor.model.lora.alpha=128 \
  worker.actor.model.lora.target_modules=all-linear \
  worker.actor.model.lora.exclude_modules='.*visual.*' \
  worker.actor.global_batch_size=128 \
  worker.actor.micro_batch_size_per_device_for_update=$MB \
  worker.actor.micro_batch_size_per_device_for_experience=$MB \
  worker.actor.ppo_epochs=1 \
  worker.actor.clip_ratio_low=0.2 \
  worker.actor.clip_ratio_high=0.3 \
  worker.actor.clip_ratio_dual=3.0 \
  worker.actor.optim.strategy=adamw \
  worker.actor.optim.lr=1.0e-5 \
  'worker.actor.optim.betas=[0.9,0.999]' \
  worker.actor.optim.weight_decay=1.0e-2 \
  worker.actor.optim.lr_scheduler_type=constant \
  worker.actor.optim.lr_warmup_ratio=0.0 \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.ref.fsdp.enable_cpu_offload=false \
  worker.reward.reward_function="$VQS/vqs/reward/exact_match.py:compute_score" \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node=$NG \
  trainer.max_steps=96 \
  trainer.save_freq=50 \
  trainer.val_freq=25 \
  trainer.save_checkpoint_path="$OUT/ckpt" \
  2>&1 | tee -a "$OUT/train.log"
