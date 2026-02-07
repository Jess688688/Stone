#!/usr/bin/env bash

# ================================
# Basic setup
# ================================

MASTER_PORT=$((RANDOM % 50001 + 10000))

# ----------------
# Experiment config
# ----------------

forget_losses=(
    DPO+GD
)

task_list=(1)
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)

mask=true
use_LoRA=false
save_root=results/tofu_phi1-5_additional_dpo_1e-6

forget_coeff=0.1
regularization_coeff=1.0

save_checkpoint=true
num_epochs=5

save_steps=steps_per_epoch
eval_steps=(last)

# ================================
# GPU config (AUTO)
# ================================
# Examples:
#   GPU_ID=1        -> single GPU
#   GPU_ID=1,2      -> 2 GPUs (distributed)
#   GPU_ID=0,1,2,3  -> 4 GPUs (distributed)

GPU_ID=1

IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using GPUs: $GPU_ID"
echo "Number of GPUs: $NUM_GPUS"

# Decide launcher
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT"
    echo "[Mode] Distributed training (torchrun)"
else
    LAUNCHER="python"
    echo "[Mode] Single-GPU training (python)"
fi

# ================================
# split = forget01 (ONLY)
# ================================

split=forget01

for forget_loss in ${forget_losses[@]}; do
    for lr in ${learning_rates[@]}; do
        for task_id in ${task_list[@]}; do

            COMMON="use_LoRA=$use_LoRA \
forget_coeff=$forget_coeff \
regularization_coeff=$regularization_coeff \
lr=$lr \
split=$split \
forget_loss=$forget_loss \
num_epochs=$num_epochs \
mask=$mask \
fix_ref_model=$fix_ref_model \
save_root=$save_root \
save_checkpoint=$save_checkpoint"

            echo "=============================================="
            echo "TRAIN | split=$split | loss=$forget_loss | lr=$lr"
            echo "=============================================="

            CUDA_VISIBLE_DEVICES=$GPU_ID \
            $LAUNCHER forget.py \
            --config-name=phi1-5_tofu.yaml \
            task_id=$task_id \
            save_steps=$save_steps \
            $COMMON

            echo "----------------------------------------------"
            echo "EVAL  | split=$split"
            echo "----------------------------------------------"

            for step in ${eval_steps[@]}; do
                CUDA_VISIBLE_DEVICES=$GPU_ID \
                $LAUNCHER eval.py \
                --config-name=phi1-5_tofu.yaml \
                task_id=$task_id \
                eval_unlearn_step=$step \
                $COMMON
            done
        done
    done
done
