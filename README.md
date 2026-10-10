# AcTTA

**AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation**

Hyeongyu Kim, Geonhui Han, Dosik Hwang · CVPR 2026

[Paper](https://arxiv.org/abs/2603.26096) ·
[Project page](https://hyeongyu-kim.github.io/actta/) ·
[Reproduction guide](docs/REPRODUCIBILITY.md) ·
[Architecture and layer selection](docs/ARCHITECTURE.md) ·
[Measured results](docs/RESULTS.md)

[Configuration index](configs/README.md) ·
[Repository layout and maintenance](docs/DEVELOPMENT.md)

Code, configurations, and results for **AcTTA with the TENT entropy objective**
on CIFAR-C and ImageNet-C, including Source/TENT baselines and ResNet-50 and
ViT-B/16 activation-depth sweeps. Implementations of the other adaptation
objectives are available in [reference/methods](reference/methods/).

## Install

Install the Linux x86_64 Conda environment:

```bash
conda create --name actta --file environment-conda-linux-64.lock.txt --yes
conda activate actta
python -m pip install -r requirements-depth-sweep.txt -r requirements-conda-extras.lock.txt
python tools/check_environment.py
```

The lock fixes native libraries, including the JPEG decoder used for ImageNet-C.
See [environment details](docs/FRESH_CLONE_VALIDATION.md).

The measured environment used Python 3.12.2, PyTorch 2.2.1 with CUDA 12.1,
torchvision 0.17.1, timm 1.0.15, and an NVIDIA RTX A6000. Datasets and model
weights must be obtained separately. Exact checkpoint identifiers, download
links, preprocessing, and individual-run commands are in the
[reproduction guide](docs/REPRODUCIBILITY.md).

## ImageNet-C depth sweep

From the repository root, run both backbones at 0/25/50/75/100% depth:

```bash
bash scripts/reproduce_depth_sweep.sh /path/to/ImageNet-C \
  /path/to/resnet50-0676ba61.pth /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  "$PWD/output/depth_sweep" 3
```

The final argument selects the physical GPU. The script's default is GPU 3;
set it to the GPU available on your machine. Each point uses severity 5,
15 corruptions × 5,000 images, batch 128, seed 1, fixed image-ID order,
SGD LR 0.0025, momentum 0.9, Nesterov, and frozen normalization affine values.
Only the selected activation vectors are optimized. Model and optimizer reset
at each corruption. A fresh output directory is required.

**50% depth** means the first **25 of 49 independent ReLU call sites** for
ResNet-50, and the MLP GELUs of the first **6 of 12 transformer blocks** for
ViT-B/16. ResNet's shared bottleneck ReLU is split into three call sites before
selection. Joint LayerNorm adaptation is a separate ViT main-profile option.
See [the exact boundaries and parameter counts](docs/ARCHITECTURE.md).

| Requested depth | ResNet-50 error (%) | ViT-B/16 error (%) |
|---|---:|---:|
| 0% | 68.1027 | 62.1040 |
| 25% | 67.0773 | 51.4067 |
| 50% | 65.8693 | 52.4747 |
| 75% | 64.6400 | 57.8107 |
| 100% | 64.0440 | 65.8427 |

Results use seed 1. At 0%, no optimizer is constructed; ResNet uses target-batch
BN statistics. Paper comparisons are in [the results](docs/RESULTS.md).

![Measured activation-depth curves](results/depth_sweep_measured.png)

## Other measured configurations

The following entries use the main shuffled stream. Errors are percentages;
lower is better. AcTTA uses the TENT entropy objective.

| Dataset / backbone | Batch | TENT | AcTTA-TENT | AcTTA setting |
|---|---:|---:|---:|---|
| CIFAR-10-C / WRN-28-10 | 128 | 18.4276 | 17.0322 | first WRN stage; mean of seeds 1, 2, 3 |
| CIFAR-100-C / WRN-40-2 | 128 | 35.4367 | 33.9793 | first two WRN stages; seed 1 |
| CIFAR-100-C / WRN-40-2 | 4 | 57.1500 | 54.8673 | first two WRN stages; seed 1 |
| ImageNet-C / ResNet-50 | 128 | 66.6027 | 64.3213 | first 25/49 ReLU calls, LR 0.005, frozen BN affine; seed 1 |
| ImageNet-C / ViT-B/16 | 128 | 53.4987 | 49.0547 | first 6/12 MLP GELUs + all LN, LR 0.001, tanh GELU; seed 1 |

These configurations use the LRs and normalization choices shown above;
the ordered depth sweep uses fixed LR 0.0025 and frozen normalization affine values.
The complete overview is in [benchmark_summary.csv](results/benchmark_summary.csv).

- [ImageNet architecture/settings comparisons](docs/RESULTS.md#imagenet-c-architecture-and-settings-comparisons):
  separate shuffled-stream runs, including joint LayerNorm, exact/tanh GELU,
  and declared optimizer/LR variants.
- [CIFAR-C results](docs/RESULTS.md#cifar-c): WRN-28-10 on CIFAR-10-C and
  WRN-40-2 on CIFAR-100-C, including batch-size 4.
- [Result files](results/README.md): raw JSONs, per-corruption errors,
  layer manifests, and summary tables.
- [Additional configurations](docs/RESULTS.md#additional-resnet-50-configurations):
  residual-output adaptation and shuffled depth results.

## Citation and licenses

```bibtex
@article{kim2026actta,
  title={AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation},
  author={Kim, Hyeongyu and Han, Geonhui and Hwang, Dosik},
  journal={arXiv preprint arXiv:2603.26096},
  year={2026}
}
```

Code uses the [MIT license](LICENSE), with upstream notices retained in
[THIRD_PARTY.md](THIRD_PARTY.md). The existing project website (`index.html`
and `static/`) retains its
[Creative Commons Attribution-ShareAlike 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
