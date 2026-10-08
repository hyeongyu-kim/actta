#!/usr/bin/env bash
# Reproduce the selected main and small-batch experiments reported in docs.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 DATA_ROOT CHECKPOINT_ROOT OUTPUT_DIRECTORY [PHYSICAL_GPU=2]" >&2
    exit 2
fi
data_root="$(realpath "$1")"
checkpoint_root="$(realpath "$2")"
result_dir="$(realpath -m "$3")"
gpu_index="${4:-2}"
python_bin="${ACTTA_PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$result_dir"
cd "$(dirname "$0")/.."

run_experiment() {
    local config="$1" method="$2" seed="$3" dataset checkpoint name
    name="${config}_${method}_seed${seed}"
    if [[ "$config" == cifar100* ]]; then
        dataset="$data_root/CIFAR-100-C"
        checkpoint="$checkpoint_root/cifar100/corruptions/Hendrycks2020AugMix_WRN.pt"
    else
        dataset="$data_root/CIFAR-10-C"
        checkpoint="$checkpoint_root/cifar10/corruptions/Standard.pt"
    fi
    "$python_bin" -u evaluate.py --config "configs/${config}.yaml" \
        --data-dir "$dataset" --checkpoint "$checkpoint" --device cuda \
        --method "$method" --seed "$seed" --output "$result_dir/${name}.json" \
        2>&1 | tee "$result_dir/${name}.log"
}

run_experiment cifar10_wrn28_bs128 source 1
for seed in 1 2 3; do
    for method in tent actta_tent; do
        run_experiment cifar10_wrn28_bs128 "$method" "$seed"
    done
done
for method in source tent actta_tent; do
    run_experiment cifar100_wrn40_bs128 "$method" 1
done
for method in tent actta_tent; do
    run_experiment cifar100_wrn40_bs4 "$method" 1
done
"$python_bin" tools/aggregate_results.py "$result_dir"/*_seed*.json \
    --output "$result_dir/aggregate.json"
