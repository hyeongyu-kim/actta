# Activation selection and architecture-specific settings

The adaptation objective is TENT entropy minimization. These notes describe
activation selection and settings for the depth sweep and main configurations.
Raw JSONs record actual module paths, widths, trainable parameters, and settings.

## Activation parameters

For `z = x - c`, `actta/activation.py` implements the original GS2 formula:

```text
g(x) = phi(z) + [lambda_neg + (lambda_pos - lambda_neg) * sigmoid(beta * z)] * z
```

`lambda_neg`, `lambda_pos`, and `c` are channel-wise vectors initialized to
zero; beta stays at one. Parameter names retain `neg_gsL`, `pos_gsL`, and
`shift_gsL`. An activation of width C contributes 3C trainable values.
Original `ln/lp/lb` flags concern activation vectors, rather than LayerNorm;
`adapt_norm` separately controls normalization affine updates in the YAMLs.

## ResNet-50: independent ReLU call sites

The torchvision ResNet-50 has sixteen bottlenecks, each reusing one ReLU three
times: 49 calls including the stem. `split_resnet_calls` in
`actta/architecture_audit.py` gives each bottleneck independent
`relu1`, `relu2`, and `relu3` call sites without changing its forward computation.
Selection follows stem → layer1 → layer2 → layer3 → layer4, and relu1/2/3
within each bottleneck.

| Nominal depth | Selected / total | Actual ratio | Last selected path | Trainable activation values |
|---|---|---:|---|---:|
| 0% | 0 / 49 | 0% | none | 0 |
| 25% | 13 / 49 | 26.53% | `model.layer2.0.relu3` | 5,952 |
| 50% | 25 / 49 | 51.02% | `model.layer3.0.relu3` | 17,472 |
| 75% | 38 / 49 | 77.55% | `model.layer3.5.relu1` | 36,672 |
| 100% | 49 / 49 | 100% | `model.layer4.2.relu3` | 68,160 |

At 50%, select the stem, all nine calls in layer1, all twelve in layer2,
and all three in `layer3.0`. The selected counts are 13/25/38/49.
The 10% YAML selects five calls through
`model.layer1.1.relu1`. Full measured path lists are in
[the layer manifest](../results/depth_sweep_layer_manifest.csv).

AcTTA freezes BN affine values and uses target-batch statistics. At zero depth,
these statistics are retained without an optimizer. Source uses checkpoint
running statistics. `audit_resnet50_legacy_nesterov.yaml` selects the stem
plus sixteen residual-output ReLUs (17 modules, 45,504 trainable values).

## ViT-B/16: MLP GELUs

The measured backbone is timm 1.0.15's
`vit_base_patch16_224.orig_in21k_ft_in1k`, with the exact official
`jx_vit_base_p16_224-80ecf9dd.pth` checkpoint. It has twelve blocks, embedding
width 768, and MLP hidden width 3,072. Selected paths are
`model.blocks.<i>.mlp.act`; attention and projection weights stay frozen.

| Nominal depth | Selected blocks | Last selected path | Trainable activation values |
|---|---|---|---:|
| 0% | none | none | 0 |
| 25% | 0–2 | `model.blocks.2.mlp.act` | 27,648 |
| 50% | 0–5 | `model.blocks.5.mlp.act` | 55,296 |
| 75% | 0–8 | `model.blocks.8.mlp.act` | 82,944 |
| 100% | 0–11 | `model.blocks.11.mlp.act` | 110,592 |

Each replaced GELU has three vectors of width 3,072. At 50%, replace the
GELUs in **blocks 0 through 5**. The 10% YAML selects
block 0 (one of twelve blocks).

Normalization selection is independent of activation depth. The ordered sweep
sets `adapt_norm: false`. The joint-LN configuration sets `adapt_norm: true`
and updates **all 24 block LayerNorms plus the final LayerNorm**, including
norms in blocks whose GELU is not adapted. These add 38,400 affine values;
the six-block joint-LN total is **93,696**. Embeddings, attention, linear
weights, classifier, and convolutional weights stay frozen. TENT updates only
normalization affine values: 53,120 for ResNet and 38,400 for ViT.

All ViT dropout probabilities are zero; measured attention uses fused
scaled-dot-product attention. See [runtime settings](../results/architecture_runtime_settings.json).

## GELU and zero initialization

The checkpoint uses exact `nn.GELU(approximate="none")`. Configurations with
`gelu_approximation: tanh` use tanh GELU in selected blocks; zero activation
vectors recover that tanh function. Unselected blocks and zero-depth ViT use
the checkpoint's exact GELU.

`audit_vit_b16_blocks6_joint_ln_exact_gelu.yaml` uses exact GELU, including at
zero initialization. Results for both variants are in [RESULTS.md](RESULTS.md).

## Depth and main profiles

| Profile | Input order | AcTTA LR | Normalization affine | Base activation |
|---|---|---:|---|---|
| RN50 ordered depth sweep | fixed image-ID order | 0.0025 | frozen BN | ReLU |
| ViT ordered depth sweep | fixed image-ID order | 0.0025 | frozen LN | tanh GELU in selected blocks |
| RN50 25-call main configuration | shuffled | 0.005 | frozen BN | ReLU |
| ViT 6-block joint-LN configuration | shuffled | 0.001 | all LN updated | tanh GELU in selected blocks |

All four use SGD momentum 0.9, Nesterov=True, zero dampening/weight decay, and
one entropy step. Other declared variants have separate YAMLs and results.
Standard-SGD variants set Nesterov=False. The depth sweep freezes normalization
affine values; the ViT main configuration jointly adapts all LayerNorms.

Other objectives can select normalization parameters differently. Original
SAR/DeYO collectors exclude ViT blocks 9–11 and RN50 layer4; their literal
name filters leave the wrapped final `model.norm` selected. See
[the objective-specific record](../results/architecture_norm_selection_by_objective.json).
