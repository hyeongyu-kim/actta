#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 IMAGENET_C_ROOT RN50_CHECKPOINT VIT_CHECKPOINT OUTPUT_DIR [PHYSICAL_GPU=3]" >&2
  exit 2
fi
data_root=$1
rn_checkpoint=$2
vit_checkpoint=$3
output_dir=$4
export CUDA_VISIBLE_DEVICES=${5:-3}
if [[ ! $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]; then
  echo "Choose one physical GPU index." >&2
  exit 2
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=${ACTTA_PYTHON:-python}
read -r -a depth_points <<< "${DEPTH_POINTS:-0 25 50 75 100}"
mkdir -p "$output_dir"
results=()
for architecture in resnet50 vit_b16; do
  checkpoint=$rn_checkpoint
  if [[ $architecture == vit_b16 ]]; then checkpoint=$vit_checkpoint; fi
  for depth in "${depth_points[@]}"; do
    printf -v padded_depth '%03d' "$depth"
    name="${architecture}_d${padded_depth}"
    result="$output_dir/${name}_seed1.json"
    if [[ -e $result || -e $output_dir/${name}_seed1.log ]]; then
      echo "Existing result/log; choose a new output directory: $name" >&2
      exit 1
    fi
    "$python_bin" evaluate_depth_sweep_ordered.py --config "configs/depth_sweep_ordered/${name}.yaml" \
      --data-dir "$data_root" --checkpoint "$checkpoint" --method actta_tent \
      --seed 1 --workers 4 --device cuda --output "$result" \
      2>&1 | tee "$output_dir/${name}_seed1.log"
    results+=("$result")
  done
done
"$python_bin" -m tools.summarize_depth_sweep "${results[@]}" \
  --depths "${depth_points[@]}" --expected-gpu "$CUDA_VISIBLE_DEVICES" --input-order ordered \
  --output "$output_dir/depth_sweep_summary"
MPLCONFIGDIR="${MPLCONFIGDIR:-$output_dir/.mpl-cache}" \
  "$python_bin" -m tools.plot_depth_sweep --summary "$output_dir/depth_sweep_summary.json" \
  --output "$output_dir/depth_sweep_curve"
