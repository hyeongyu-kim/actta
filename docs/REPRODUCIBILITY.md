# Reproducing the supplied configurations

Run commands from the repository root. Active evaluators implement Source,
TENT, and AcTTA_TENT. Original snapshots under `reference/methods/` are supplied
for comparison and porting of other objectives.

Use [the configuration index](../configs/README.md) to select a profile.
`python tools/run_suite.py --help` lists the common suite launcher options;
see [suite usage](DEVELOPMENT.md#run-a-suite).

## Environment

Install the **Linux x86_64 Conda environment** and pinned Python requirements:

```bash
conda create --name actta --file environment-conda-linux-64.lock.txt --yes
conda activate actta
python -m pip install -r requirements-depth-sweep.txt -r requirements-conda-extras.lock.txt
python tools/check_environment.py
```

The lock pins Python, PyTorch/CUDA, NumPy, Pillow, and native libraries.
ImageNet-C uses Pillow 10.2 with IJG JPEG 9; JPEG decoder builds affect input
pixels. Use the locked environment to keep decoding consistent. See
[environment details](FRESH_CLONE_VALIDATION.md).

`requirements.txt` suffices for CIFAR-C; `requirements-imagenet-audit.txt`
adds timm/Pillow; `requirements-depth-sweep.txt` adds matplotlib for plotting.
Measured versions are Python 3.12.2, PyTorch 2.2.1 (CUDA 12.1), torchvision
0.17.1, NumPy 1.26.4, timm 1.0.15, Pillow 10.2.0, and matplotlib 3.10.3.
Full runs used float32 without mixed precision on an RTX A6000.
`results/architecture_runtime_settings.json` records attention/dropout/TF32 defaults.

## ImageNet-C data and checkpoints

Obtain [ImageNet-C](https://github.com/hendrycks/robustness) separately. Layout:
`ImageNet-C/<corruption>/5/<class>/<image>.JPEG`. Evaluation uses the original
RobustBench 5,000-image list and class map in `reference/data/`, for all fifteen
corruptions. These are indices/labels; dataset images are not bundled.

The already 224×224 JPEGs are decoded as RGB, converted with `ToTensor`, and
normalized inside the model. There is no additional resize/crop.

| Backbone | Checkpoint | Mean | Std |
|---|---|---|---|
| torchvision ResNet-50 `IMAGENET1K_V1` | `resnet50-0676ba61.pth` | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| timm `vit_base_patch16_224.orig_in21k_ft_in1k` | `jx_vit_base_p16_224-80ecf9dd.pth` | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 |

Official sources: [torchvision checkpoint](https://download.pytorch.org/models/resnet50-0676ba61.pth),
[timm release checkpoint](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth),
and [timm model card](https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k).
The YAMLs enforce full checkpoint SHA-256:

```text
resnet50-0676ba61.pth
0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a

jx_vit_base_p16_224-80ecf9dd.pth
80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c
```

Other pretrained variants are not interchangeable with these measurements.

## Ordered depth sweep

```bash
bash scripts/reproduce_depth_sweep.sh /path/to/ImageNet-C \
  /path/to/resnet50-0676ba61.pth /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  "$PWD/output/depth_sweep" 3
```

The last argument selects a physical GPU; replace 3 with an available index.
Configurations are in `configs/depth_sweep_ordered/`.
For one point:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python evaluate_depth_sweep_ordered.py \
  --config configs/depth_sweep_ordered/vit_b16_d050.yaml \
  --data-dir /path/to/ImageNet-C \
  --checkpoint /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  --method actta_tent --seed 1 --workers 4 --device cuda \
  --output output/vit_b16_d050_seed1.json
```

Select a different YAML/checkpoint for ResNet. To include 10% depth,
set `DEPTH_POINTS="0 10 25 50 75 100"` before the
sweep command. Use a fresh output directory.

The profile fixes seed 1, batch 128, SGD LR 0.0025, momentum 0.9,
Nesterov=True, zero weight decay/dampening, frozen BN/LN affine values, and
one entropy step per batch. `shuffle=False` retains exact image-ID file order,
matching the original depth runner. The final eight-image batch is retained.
Predictions are taken before the optimizer update; model and optimizer reset
at every corruption.

At zero depth there are no activation vectors or optimizer. ResNet keeps
target-batch BN statistics; ViT keeps fixed LN and checkpoint exact GELU.
Exact selection boundaries and the joint-LN main profile are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Shuffled architecture/settings comparisons

`evaluate_architecture.py` follows the original main runner's `shuffle=True`.
Run the architecture/settings configurations:

```bash
bash scripts/reproduce_architecture_audit.sh /path/to/ImageNet-C \
  /path/to/resnet50-0676ba61.pth /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  "$PWD/output/architecture" 3
```

Individual ViT joint-LN configuration:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python evaluate_architecture.py \
  --config configs/audit_vit_b16_blocks6_joint_ln.yaml \
  --data-dir /path/to/ImageNet-C \
  --checkpoint /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  --method actta_tent --seed 1 --workers 4 --device cuda \
  --output output/vit_b16_joint_ln_seed1.json
```

This configuration adapts the first six MLP activations and all 25 LayerNorms,
using LR 0.001 and Nesterov=True. `audit_resnet50_calls25_main_lr.yaml`
selects 25/49 calls, frozen BN affine values, and LR 0.005. Separate YAMLs
expose exact GELU, standard SGD, archived LR, and residual-output selection.
Use `--method tent` or `--method source` for controls. Inactive YAML options
remain in the raw record; actual applied selection/settings are recorded too.

## CIFAR-C

Put the fifteen complete corruption arrays and `labels.npy` under
`CIFAR-10-C/` or `CIFAR-100-C/`. Obtain original
[RobustBench](https://github.com/RobustBench/robustbench) checkpoints separately.

| Dataset | Measured backbone | Checkpoint | Configs |
|---|---|---|---|
| CIFAR-10-C | WRN-28-10 (`Standard`) | `Standard.pt` | `configs/cifar10_wrn28_bs128.yaml` |
| CIFAR-100-C | WRN-40-2 (`Hendrycks2020AugMix_WRN`) | `Hendrycks2020AugMix_WRN.pt` | `configs/cifar100_wrn40_bs128.yaml`, `configs/cifar100_wrn40_bs4.yaml` |

```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
  --config configs/cifar10_wrn28_bs128.yaml \
  --data-dir /path/to/CIFAR-10-C --checkpoint /path/to/Standard.pt \
  --method actta_tent --seed 1 --output output/cifar10_actta_seed1.json
```

The CIFAR script uses seeds 1, 2, and 3 for CIFAR-10-C TENT/AcTTA and seed 1
for the other configurations. Its directory convention is:

```text
DATA_ROOT/CIFAR-10-C/
DATA_ROOT/CIFAR-100-C/
CHECKPOINT_ROOT/cifar10/corruptions/Standard.pt
CHECKPOINT_ROOT/cifar100/corruptions/Hendrycks2020AugMix_WRN.pt
```

```bash
bash scripts/reproduce_selected.sh /path/to/DATA_ROOT \
  /path/to/CHECKPOINT_ROOT "$PWD/output/cifar" 3
```

CIFAR uses Adam (beta1 0.9, beta2 0.999, default epsilon, zero weight decay).
BS128 LR is 0.001 for TENT and 0.01 for AcTTA; BS4 rates are 0.0001 and
0.001. AcTTA adapts three activation vectors and freezes BN affine values
while using target-batch statistics. TENT adapts BN affine values. Each run
covers 15 severity-5 corruptions × 10,000 images, shuffled, with shift resets.
Source uses checkpoint running statistics. Activation prefixes are in the YAMLs.

## Output checks and aggregation

Evaluators refuse existing JSON outputs. `--limit` and `--corruption` permit
smoke checks; these are marked partial and rejected by full-benchmark
summarizers.
The original logging `torch.rand(1)` draw is retained before model creation
to preserve the research random stream. Changing order, batch, checkpoint,
GELU, normalization adaptation, or LR defines a new experiment.

Recompute supplied depth results and plot without rerunning models:

```bash
python -m tools.summarize_depth_sweep results/depth_sweep_runs/*.json \
  --depths 0 25 50 75 100 --input-order ordered \
  --output output/depth_summary
MPLCONFIGDIR="$PWD/output/.mpl-cache" python -m tools.plot_depth_sweep \
  --summary output/depth_summary.json --output output/depth_curve
```

Other aggregations:

```bash
python -m tools.summarize_architecture_audit results/architecture_runs/*.json \
  --output output/architecture_summary
python tools/aggregate_results.py results/gpu2_runs/*.json \
  --output output/cifar_summary.json
```

Regenerate the benchmark overview, additional configuration results, and run index:

```bash
python -m tools.report_completed_results --output output/completed_report
```

Additional ResNet configurations are described in
[the results](RESULTS.md#additional-resnet-50-configurations).
