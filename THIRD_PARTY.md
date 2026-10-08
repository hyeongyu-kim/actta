# Third-party code

`LICENSE` retains the original MIT notice for Mario Döbler and Robert Marsden's
[test-time-adaptation benchmark](https://github.com/mariodoebler/test-time-adaptation).
The paired research objective files are derived from this benchmark and retain
their in-file upstream links and notices, including TENT and related methods.

`vendor/wide_resnet.py` is an unchanged snapshot from the research copy of
RobustBench's `model_zoo/architectures/wide_resnet.py`. Its TRADES attribution
remains in the file. RobustBench's upstream license, including its copyright
and model-specific notices, is preserved in `vendor/ROBUSTBENCH_LICENSE`.

`vendor/resnext.py` is an unchanged snapshot of the research workspace's
module-based ResNeXt implementation. Its embedded MIT notice for Xuanyi Dong
and its AugMix attribution are retained.

Datasets and pretrained weights are external artifacts and are not bundled.
The ResNet-50 bottleneck forwards in `actta/imagenet.py` and
`actta/architecture_audit.py` follow torchvision's
BSD-3-Clause implementation. Its license is retained in
`vendor/TORCHVISION_LICENSE`. The original RobustBench ImageNet test-image list
and class mapping are included in `reference/data/`, with source hashes in
`reference/imagenet_data_manifest.json`.
The optional ViT audit uses timm1.0.15 as an external Apache-2.0 library and the
official `orig_in21k_ft_in1k` checkpoint; that checkpoint is not bundled.
The unchanged project website (`index.html` and `static/`) retains its
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).
The code license does not replace the website license.
