# Configuration index

Each YAML is a complete experiment configuration. Main settings use shuffled
input; `depth_sweep_ordered/` uses fixed image-ID order and frozen normalization
affine values.

| Experiment | Configuration | Evaluator |
|---|---|---|
| CIFAR-10-C, WRN-28-10, batch 128 | `cifar10_wrn28_bs128.yaml` | `evaluate.py` |
| CIFAR-100-C, WRN-40-2, batch 128 / 4 | `cifar100_wrn40_bs128.yaml`, `cifar100_wrn40_bs4.yaml` | `evaluate.py` |
| RN50, first 25 ReLU calls, LR 0.005 | `audit_resnet50_calls25_main_lr.yaml` | `evaluate_architecture.py` |
| ViT, first 6 MLP activations + all LN, LR 0.001 | `audit_vit_b16_blocks6_joint_ln.yaml` | `evaluate_architecture.py` |
| RN50 / ViT depth sweep | `depth_sweep_ordered/{resnet50,vit_b16}_d{000,025,050,075,100}.yaml` | `evaluate_depth_sweep_ordered.py` |

## Architecture variants

| Configuration | Setting |
|---|---|
| `audit_resnet50_calls25.yaml` | First 25/49 ReLU calls, LR 0.0025, frozen BN affine |
| `audit_resnet50_legacy_nesterov.yaml` | Stem + residual-output ReLUs, LR 0.005 |
| `audit_vit_b16_blocks6_depth_ablation.yaml` | First 6 MLP activations, LR 0.0025, frozen LN, shuffled |
| `audit_vit_b16_blocks6_joint_ln_exact_gelu.yaml` | Joint LN with exact GELU |
| `audit_vit_b16_blocks6_joint_ln_plain_sgd.yaml` | Joint LN with exact GELU, Nesterov=False |
| `audit_vit_b16_blocks6_joint_ln_archived_lr.yaml` | Joint LN with tanh GELU, LR 0.00025 |

The ordered depth directory also contains `d010` configurations. See
[layer selection](../docs/ARCHITECTURE.md) for exact paths and parameter counts.
Suite commands are in [the reproduction guide](../docs/REPRODUCIBILITY.md).
