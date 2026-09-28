#!/usr/bin/env bash
# GRPO training for the released checkpoint (arm `c5_datafix`, 128 steps).
#
# Every value below is the one the released run actually used, read back from
# runs/train/c5_datafix/ckpt/experiment_config.json -- nothing here is a guess.
#
# Needs 2 GPUs. Single-rank FSDP falls back to NO_SHARD, whose _use_sharded_views() asserts every
# parameter view is an nn.Parameter; under PEFT the frozen base weights are plain Tensors and it
# dies. This is a real constraint, not a preference.
#
#   GPUS=0,1 bash scripts/train.sh data/rl_pool vqs_release
set -euo pipefail
DATA=$1; EXP=$2
: "${GPUS:?set GPUS, e.g. GPUS=0,1}"
: "${EASYR1:?set EASYR1 to your EasyR1 checkout}"
: "${VQS:?set VQS to this repository root}"
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=${OUT:-runs/$EXP}; mkdir -p "$OUT"
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
  data.seed="${SEED:-1}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_loss=true \
  algorithm.kl_coef=1.0e-2 \
  worker.rollout.n=8 \
  worker.rollout.temperature=1.0 \
  worker.rollout.tensor_parallel_size=$NG \
  worker.rollout.gpu_memory_utilization=0.60 \
  worker.actor.model.model_path=Qwen/Qwen3-VL-2B-Instruct \
  worker.actor.model.freeze_vision_tower=true \
  worker.actor.model.lora.rank=64 \
  worker.actor.model.lora.alpha=128 \
  worker.actor.model.lora.target_modules=all-linear \
  worker.actor.model.lora.exclude_modules='.*visual.*' \
  worker.actor.global_batch_size=128 \
  worker.actor.optim.lr=1.0e-5 \
  worker.actor.micro_batch_size_per_device_for_update=16 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false \
  worker.ref.fsdp.enable_cpu_offload=false \
  worker.reward.reward_function="$VQS/vqs/reward/se_evidence.py:compute_score" \
  trainer.experiment_name="$EXP" \
  trainer.n_gpus_per_node=$NG \
  trainer.max_steps=128 \
  trainer.save_freq=50 \
  trainer.val_freq=25 \
  trainer.save_checkpoint_path="$OUT/ckpt" \
  2>&1 | tee -a "$OUT/train.log"
