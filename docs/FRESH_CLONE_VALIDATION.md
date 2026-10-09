# Runtime environment

Use the Linux x86_64 Conda lock for the ImageNet-C experiments:

```bash
conda create --name actta --file environment-conda-linux-64.lock.txt --yes
conda activate actta
python -m pip install -r requirements-depth-sweep.txt -r requirements-conda-extras.lock.txt
python tools/check_environment.py
```

| Component | Version |
|---|---|
| Python | 3.12.2 |
| PyTorch / torchvision | 2.2.1 / 0.17.1 |
| CUDA / cuDNN | 12.1 / 8.9.2 |
| NumPy | 1.26.4 |
| timm | 1.0.15 |
| Pillow / JPEG decoder | 10.2.0 / IJG JPEG 9 |
| matplotlib | 3.10.3 |

ImageNet-C inputs are JPEG images. The decoder build affects decoded RGB values,
so use the pinned Conda binaries as well as the Python package versions.

[environment-conda-linux-64.lock.txt](../environment-conda-linux-64.lock.txt)
pins native package builds;
[requirements-conda-extras.lock.txt](../requirements-conda-extras.lock.txt)
pins the additional Python packages. `check_environment.py` checks runtime
versions and the JPEG decoder.
