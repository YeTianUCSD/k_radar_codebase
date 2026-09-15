"""Descriptor loading, support-only standardization, and torch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class FeatureStandardizer:
    means: dict
    scales: dict

    @classmethod
    def fit(cls, arrays: Mapping[str, np.ndarray], mask: np.ndarray):
        means, scales = {}, {}
        for modality, values in arrays.items():
            selected = np.asarray(values[mask], dtype=np.float64)
            means[modality] = selected.mean(axis=0).astype(np.float32)
            scale = selected.std(axis=0).astype(np.float32)
            scales[modality] = np.maximum(scale, 1e-6)
        return cls(means, scales)

    def transform(self, modality: str, values):
        return (
            (np.asarray(values, dtype=np.float32) - self.means[modality])
            / self.scales[modality]
        ).astype(np.float32)

    def state_dict(self):
        return {
            "means": self.means,
            "scales": self.scales,
        }


def load_split_arrays(descriptor_bank: Path, split: str, modalities: Sequence[str], descriptor: str):
    root = Path(descriptor_bank) / split
    index = pd.read_csv(root / "index.csv")
    arrays = {
        modality: np.load(root / f"{modality}_{descriptor}.npy", mmap_mode="r")
        for modality in modalities
    }
    for modality, values in arrays.items():
        if len(values) != len(index):
            raise ValueError(f"{split}/{modality} feature-index mismatch")
    return arrays, index


class AttributeDataset(Dataset):
    def __init__(self, features, labels, positions):
        self.features = features
        self.labels = labels
        self.positions = np.asarray(positions, dtype=int)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = self.positions[item]
        return (
            {key: torch.from_numpy(value[position]) for key, value in self.features.items()},
            {key: torch.tensor(value[position], dtype=torch.long) for key, value in self.labels.items()},
            int(position),
        )
