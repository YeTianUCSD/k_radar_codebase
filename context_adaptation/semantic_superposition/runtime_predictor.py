"""Run the frozen V7 semantic classifier on live ASF encoder outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch


class EncodedAttributePredictor:
    """Adapter around ``AttributePredictor`` for NCHW encoder tensors."""

    DEFAULT_FEATURE_KEYS = {
        "camera": "cam_bev_feat",
        "lidar": "spatial_features_2d",
        "radar": "bev_feat",
    }

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "auto",
        feature_keys: Mapping[str, str] | None = None,
    ) -> None:
        # Imports stay local so the main K-Radar runner can install the two
        # research-package roots before constructing this object.
        from online_context.predictor import AttributePredictor

        self.predictor = AttributePredictor(checkpoint_path, device=device)
        self.feature_keys = dict(self.DEFAULT_FEATURE_KEYS)
        self.feature_keys.update(dict(feature_keys or {}))

    @staticmethod
    def _mean_std(value: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(value) or value.ndim != 4:
            raise ValueError("live encoder features must be NCHW tensors")
        values = value.detach().float()
        mean = values.mean(dim=(2, 3))
        std = values.std(dim=(2, 3), unbiased=False)
        return torch.cat((mean, std), dim=1).cpu().numpy().astype(
            np.float32, copy=False
        )

    @staticmethod
    def illumination_from_batch(batch: Mapping[str, Any]) -> np.ndarray:
        from factorized_mlp.illumination import extract_spatial_illumination

        metadata = batch.get("meta", ())
        rows = []
        for item in metadata:
            path = item.get("path", {}).get("front")
            if not path:
                raise KeyError("batch metadata does not contain path.front")
            image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"could not read front camera image: {path}")
            rows.append(extract_spatial_illumination(image))
        if not rows:
            raise ValueError("batch contains no metadata rows")
        return np.stack(rows).astype(np.float32, copy=False)

    def descriptors(
        self,
        encoded: Mapping[str, Any],
        illumination: np.ndarray,
    ) -> dict[str, np.ndarray]:
        output = {}
        for modality in self.predictor.sensor_modalities:
            key = self.feature_keys[modality]
            if key not in encoded:
                raise KeyError(f"encoder output is missing {key!r}")
            output[modality] = self._mean_std(encoded[key])
        if "illumination" in self.predictor.modalities:
            values = np.asarray(illumination, dtype=np.float32)
            if values.ndim != 2 or len(values) != len(next(iter(output.values()))):
                raise ValueError("illumination statistics are not batch aligned")
            output["illumination"] = values
        return output

    def predict(
        self,
        encoded: Mapping[str, Any],
        illumination: np.ndarray,
    ) -> dict[str, np.ndarray]:
        descriptors = self.descriptors(encoded, illumination)
        probabilities = self.predictor.predict_probabilities(descriptors)
        if {len(value) for value in probabilities.values()} != {1}:
            raise ValueError("semantic online routing requires batch size 1")
        return {key: value[0] for key, value in probabilities.items()}

    @property
    def label_names(self):
        return self.predictor.label_names
