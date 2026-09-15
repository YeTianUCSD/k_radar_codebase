"""Fixed-epoch training that consumes no external validation labels."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import AttributeDataset
from .training import ATTRIBUTES, class_weights, seed_everything


def train_fixed_epochs(
    model,
    features,
    labels,
    train_positions,
    *,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    seed,
    device,
    class_weight_exponents=None,
):
    """Train on the complete label budget without an external validation set."""
    class_weight_exponents = {
        attribute: float((class_weight_exponents or {}).get(attribute, 1.0))
        for attribute in ATTRIBUTES
    }
    seed_everything(seed)
    model.to(device)
    dataset = AttributeDataset(features, labels, train_positions)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=0,
    )
    losses = {
        attribute: nn.CrossEntropyLoss(
            weight=class_weights(
                labels[attribute][train_positions],
                model.heads[attribute][-1].out_features,
                exponent=class_weight_exponents[attribute],
            ).to(device)
        )
        for attribute in ATTRIBUTES
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    started = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch_features, batch_labels, _ in loader:
            batch_features = {
                key: value.to(device) for key, value in batch_features.items()
            }
            batch_labels = {
                key: value.to(device) for key, value in batch_labels.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)["logits"]
            loss = sum(
                losses[key](logits[key], batch_labels[key])
                for key in ATTRIBUTES
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(next(iter(batch_labels.values())))
            total_loss += float(loss.detach().cpu()) * count
            samples += count
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
        })
    return {
        "model": model,
        "history": history,
        "epochs": int(epochs),
        "elapsed_seconds": time.time() - started,
    }
