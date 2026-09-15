from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


def modality_combinations(modalities: Sequence[str]) -> List[Tuple[str, ...]]:
    return [combo for size in range(1, len(modalities) + 1) for combo in combinations(modalities, size)]


@dataclass
class ProjectionSettings:
    pca_components: int = 64
    whiten: bool = False
    l2_normalize: bool = True
    seed: int = 20260812


class PerModalityProjector:
    """Fit scaling/PCA independently per modality, then concatenate outputs."""

    def __init__(self, modalities: Sequence[str], settings: Optional[ProjectionSettings] = None):
        self.modalities = tuple(modalities)
        self.settings = settings or ProjectionSettings()
        self.scalers: Dict[str, StandardScaler] = {}
        self.pcas: Dict[str, Optional[PCA]] = {}
        self.output_dimensions: Dict[str, int] = {}

    def fit(self, arrays: Mapping[str, np.ndarray]) -> "PerModalityProjector":
        for modality in self.modalities:
            values = np.asarray(arrays[modality], dtype=np.float32)
            scaler = StandardScaler(copy=True)
            scaled = scaler.fit_transform(values)
            requested = int(self.settings.pca_components)
            maximum = min(scaled.shape[0] - 1, scaled.shape[1])
            count = min(requested, maximum) if requested > 0 else 0
            pca: Optional[PCA]
            if count and count < scaled.shape[1]:
                pca = PCA(
                    n_components=count,
                    whiten=self.settings.whiten,
                    svd_solver="randomized",
                    random_state=self.settings.seed,
                )
                pca.fit(scaled)
                dimension = count
            else:
                pca = None
                dimension = scaled.shape[1]
            self.scalers[modality] = scaler
            self.pcas[modality] = pca
            self.output_dimensions[modality] = dimension
        return self

    def transform_modalities(self, arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for modality in self.modalities:
            values = np.asarray(arrays[modality], dtype=np.float32)
            projected = self.scalers[modality].transform(values)
            pca = self.pcas[modality]
            if pca is not None:
                projected = pca.transform(projected)
            projected = np.asarray(projected, dtype=np.float32)
            if self.settings.l2_normalize:
                projected = normalize(projected, norm="l2", axis=1, copy=False)
            result[modality] = projected
        return result

    def transform(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        parts = self.transform_modalities(arrays)
        return np.concatenate([parts[modality] for modality in self.modalities], axis=1)

    def fit_transform(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.fit(arrays).transform(arrays)

    def metadata(self) -> Dict[str, object]:
        explained = {}
        for modality, pca in self.pcas.items():
            explained[modality] = None if pca is None else float(pca.explained_variance_ratio_.sum())
        return {
            "modalities": list(self.modalities),
            "settings": vars(self.settings),
            "output_dimensions": self.output_dimensions,
            "pca_explained_variance_sum": explained,
        }

