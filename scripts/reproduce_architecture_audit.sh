#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 IMAGENET_C_ROOT RN50_V1_CHECKPOINT TIMM_ORIG_VIT_CHECKPOINT OUTPUT_DIR [PHYSICAL_GPU=2]" >&2
  exit 2
fi
data_root=$1
resnet_checkpoint=$2
vit_checkpoint=$3
output_dir=$4
export CUDA_VISIBLE_DEVICES=${5:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=${ACTTA_PYTHON:-python}
mkdir -p "$output_dir"

run() {
  local name=$1 config=$2 method=$3 checkpoint=$4
  "$python_bin" evaluate_architecture.py --config "configs/$config" \
    --data-dir "$data_root" --checkpoint "$checkpoint" --method "$method" \
    --seed 1 --workers 4 --device cuda --output "$output_dir/${name}_seed1.json" \
    2>&1 | tee "$output_dir/${name}_seed1.log"
}

run resnet50_nesterov_tent audit_resnet50_calls25.yaml tent "$resnet_checkpoint"
run resnet50_calls25_actta audit_resnet50_calls25.yaml actta_tent "$resnet_checkpoint"
run resnet50_legacy_nesterov_actta audit_resnet50_legacy_nesterov.yaml actta_tent "$resnet_checkpoint"
run vit_b16_source audit_vit_b16_blocks6_joint_ln.yaml source "$vit_checkpoint"
run vit_b16_tent audit_vit_b16_blocks6_joint_ln.yaml tent "$vit_checkpoint"
run vit_b16_blocks6_joint_ln_actta audit_vit_b16_blocks6_joint_ln.yaml actta_tent "$vit_checkpoint"
run vit_b16_blocks6_depth_ablation_actta audit_vit_b16_blocks6_depth_ablation.yaml actta_tent "$vit_checkpoint"
run vit_b16_blocks6_joint_ln_exact_gelu_actta audit_vit_b16_blocks6_joint_ln_exact_gelu.yaml actta_tent "$vit_checkpoint"
run resnet50_calls25_main_lr_actta audit_resnet50_calls25_main_lr.yaml actta_tent "$resnet_checkpoint"
run vit_b16_plain_sgd_tent audit_vit_b16_blocks6_joint_ln_plain_sgd.yaml tent "$vit_checkpoint"
run vit_b16_blocks6_joint_ln_plain_sgd_actta audit_vit_b16_blocks6_joint_ln_plain_sgd.yaml actta_tent "$vit_checkpoint"
run vit_b16_blocks6_joint_ln_archived_lr_actta audit_vit_b16_blocks6_joint_ln_archived_lr.yaml actta_tent "$vit_checkpoint"
"$python_bin" -m tools.summarize_architecture_audit "$output_dir"/*_seed1.json \
  --output "$output_dir/architecture_audit_summary"
