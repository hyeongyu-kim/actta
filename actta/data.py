"""CIFAR-C severity slices; the runner controls the evaluation stream order."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]


class CIFARCorruption(Dataset):
    def __init__(self, root, corruption, severity=5, limit=None):
        if corruption not in CORRUPTIONS or severity not in range(1, 6):
            raise ValueError("Invalid corruption or severity")
        root = Path(root)
        self.images = np.load(root / (corruption + ".npy"), mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        if self.images.shape != (50000, 32, 32, 3):
            raise ValueError(f"Expected full CIFAR-C arrays, got {self.images.shape}")
        start = (severity - 1) * 10000
        self.images = self.images[start:start + 10000]
        self.labels = labels[start:start + 10000] if len(labels) == 50000 else labels
        if len(self.labels) != 10000:
            raise ValueError("CIFAR-C labels must contain 10,000 or 50,000 entries")
        if limit is not None and not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10,000")
        self.length = 10000 if limit is None else limit

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Copy the mmap slice; model operations must not modify dataset bytes.
        image = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1).float() / 255.
        return image, int(self.labels[index])


class IndexedDataset(Dataset):
    """Expose original sample indices so an evaluation order can be hashed."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index
