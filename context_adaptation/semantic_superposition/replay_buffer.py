"""Bounded CPU replay storage for frames withheld during context decisions."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


SemanticKey = Tuple[str, str, str]


def compact_encoded_batch(encoded: Any, feature_keys: Sequence[str]) -> dict:
    """Keep only tensors consumed by fuser/head/loss during pending replay."""
    if not isinstance(encoded, dict):
        raise TypeError("encoded batch must be a dictionary")
    required = tuple(dict.fromkeys((*tuple(feature_keys), "gt_boxes", "batch_size")))
    missing = [key for key in required if key not in encoded]
    if missing:
        raise KeyError(f"encoded batch is missing replay fields: {missing}")
    return {key: encoded[key] for key in required}


def _to_cpu(value: Any, floating_dtype: Optional[torch.dtype]) -> Any:
    if torch.is_tensor(value):
        result = value.detach().to("cpu")
        if floating_dtype is not None and result.is_floating_point():
            result = result.to(floating_dtype)
        return result.contiguous()
    if isinstance(value, dict):
        return {key: _to_cpu(item, floating_dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item, floating_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item, floating_dtype) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        result = value.to(device)
        # Encoder outputs are stored as FP16 to bound CPU memory, while the
        # existing ASF fuser/head run in FP32 without autocast.
        if result.is_floating_point() and result.dtype == torch.float16:
            result = result.float()
        return result
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ReplayFrame:
    step: int
    semantic_key: SemanticKey
    encoded_cpu: Any
    sample_id: str = ""

    def materialize(self, device: torch.device) -> Any:
        return _to_device(self.encoded_cpu, device)


class PendingReplayBuffer:
    """Keep only recent, not-yet-trained encoded frames."""

    def __init__(self, max_frames: int = 16, storage_dtype: str = "float16"):
        if int(max_frames) < 1:
            raise ValueError("max_frames must be positive")
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype must be float16 or float32")
        self.max_frames = int(max_frames)
        self.storage_dtype = str(storage_dtype)
        self._frames = deque(maxlen=self.max_frames)

    @property
    def floating_dtype(self) -> torch.dtype:
        return torch.float16 if self.storage_dtype == "float16" else torch.float32

    def append(
        self,
        *,
        step: int,
        semantic_key: Sequence[str],
        encoded: Any,
        sample_id: str = "",
    ) -> None:
        key = tuple(str(value) for value in semantic_key)
        if len(key) != 3:
            raise ValueError("semantic_key must have three attributes")
        if isinstance(encoded, dict):
            # BEV maps dominate memory and tolerate FP16 storage. Detection
            # targets must retain their original precision for fair candidate
            # loss comparison and replay training.
            encoded_cpu = {
                name: _to_cpu(
                    item,
                    None if name == "gt_boxes" else self.floating_dtype,
                )
                for name, item in encoded.items()
            }
        else:
            encoded_cpu = _to_cpu(encoded, self.floating_dtype)
        self._frames.append(ReplayFrame(
            step=int(step),
            semantic_key=key,
            encoded_cpu=encoded_cpu,
            sample_id=str(sample_id),
        ))

    def matching(
        self,
        semantic_key: Sequence[str],
        *,
        steps: Optional[Iterable[int]] = None,
    ) -> tuple[ReplayFrame, ...]:
        key = tuple(str(value) for value in semantic_key)
        allowed = None if steps is None else {int(value) for value in steps}
        return tuple(
            frame for frame in self._frames
            if frame.semantic_key == key
            and (allowed is None or frame.step in allowed)
        )

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    def steps(self) -> tuple[int, ...]:
        return tuple(frame.step for frame in self._frames)

    def state_dict(self) -> dict:
        return {
            "max_frames": self.max_frames,
            "storage_dtype": self.storage_dtype,
            "frames": [
                {
                    "step": frame.step,
                    "semantic_key": list(frame.semantic_key),
                    "encoded_cpu": _clone_for_checkpoint(frame.encoded_cpu),
                    "sample_id": frame.sample_id,
                }
                for frame in self._frames
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["max_frames"]) != self.max_frames:
            raise ValueError("replay max_frames differs from checkpoint")
        if str(state["storage_dtype"]) != self.storage_dtype:
            raise ValueError("replay storage_dtype differs from checkpoint")
        self._frames.clear()
        for item in state.get("frames", ()):
            self._frames.append(ReplayFrame(
                step=int(item["step"]),
                semantic_key=tuple(str(value) for value in item["semantic_key"]),
                encoded_cpu=_clone_for_checkpoint(item["encoded_cpu"]),
                sample_id=str(item.get("sample_id", "")),
            ))


def _clone_for_checkpoint(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_for_checkpoint(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_for_checkpoint(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_for_checkpoint(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)
