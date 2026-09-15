"""Select the historical PSP context with lowest pre-update detection loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from context_adaptation.model_adapter import (
    ContextAwareModelAdapter,
    freeze_encoder_batch_stats,
)

from .replay_buffer import ReplayFrame


@dataclass(frozen=True)
class ParentScore:
    context_id: int
    context_name: str
    mean_loss: float
    frame_count: int
    rank: int


def _network_device(network: torch.nn.Module) -> torch.device:
    return next(network.parameters()).device


class HistoricalContextSelector:
    """Compare all stored best residuals on the same confirmed frames."""

    def __init__(
        self,
        network: torch.nn.Module,
        loss_fn: Callable[[object], torch.Tensor] | None = None,
    ) -> None:
        self.network = network
        self.adapter = ContextAwareModelAdapter(network)
        self.loss_fn = loss_fn or self.adapter.loss

    def score(
        self,
        frames: Sequence[ReplayFrame],
        contexts: Mapping[int, str],
    ) -> tuple[ParentScore, ...]:
        if not frames:
            raise ValueError("parent selection requires confirmed pending frames")
        if not contexts:
            raise ValueError("parent selection requires historical contexts")
        device = _network_device(self.network)
        was_training = self.network.training
        previous = getattr(self.network, "default_scene_context", None)
        rows = []
        # AnchorHead creates its target tensors only in training mode. Gradients
        # remain disabled here, so this computes the same supervised pre-update
        # loss used online without changing model parameters.
        self.network.train()
        freeze_encoder_batch_stats(self.network)
        try:
            with torch.no_grad():
                for context_id, context_name in sorted(contexts.items()):
                    losses = []
                    for frame in frames:
                        encoded = frame.materialize(device)
                        output = self.adapter.forward_with_context(encoded, context_name)
                        value = self.loss_fn(output)
                        if not torch.isfinite(value):
                            raise RuntimeError(
                                f"non-finite parent-selection loss for {context_name}"
                            )
                        losses.append(float(value.detach().cpu().item()))
                    rows.append((int(context_id), str(context_name), float(np.mean(losses))))
        finally:
            if previous is not None:
                self.network.default_scene_context = previous
            self.network.train(was_training)
        ordered = sorted(rows, key=lambda item: (item[2], item[0]))
        return tuple(
            ParentScore(context_id, name, loss, len(frames), rank)
            for rank, (context_id, name, loss) in enumerate(ordered, start=1)
        )

    def select(
        self,
        frames: Sequence[ReplayFrame],
        contexts: Mapping[int, str],
    ) -> tuple[ParentScore, tuple[ParentScore, ...]]:
        scores = self.score(frames, contexts)
        return scores[0], scores
