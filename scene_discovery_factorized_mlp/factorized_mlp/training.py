"""Training and calibration utilities for the factorized MLP."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import AttributeDataset


ATTRIBUTES = ("weather", "road", "lighting")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: np.ndarray, class_count: int, exponent: float = 1.0):
    if not 0.0 <= float(exponent) <= 1.0:
        raise ValueError("class-weight exponent must be in [0, 1]")
    counts = np.bincount(labels.astype(int), minlength=class_count).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("support training is missing an attribute class")
    inverse = counts.sum() / (class_count * counts)
    weights = np.power(inverse, float(exponent))
    return torch.tensor(weights, dtype=torch.float32)


def _move_features(features, device):
    return {key: value.to(device) for key, value in features.items()}


def predict_logits(model, features, positions, batch_size, device):
    output = {attribute: [] for attribute in ATTRIBUTES}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            selected = np.asarray(positions[start:start + batch_size], dtype=int)
            batch = {
                key: torch.from_numpy(value[selected]).to(device)
                for key, value in features.items()
            }
            result = model(batch)["logits"]
            for attribute in ATTRIBUTES:
                output[attribute].append(result[attribute].cpu().numpy())
    return {
        attribute: np.concatenate(parts, axis=0) if parts else np.empty((0, 0))
        for attribute, parts in output.items()
    }


def validation_score(model, features, labels, positions, batch_size, device):
    logits = predict_logits(model, features, positions, batch_size, device)
    scores = []
    for attribute in ATTRIBUTES:
        prediction = logits[attribute].argmax(axis=1)
        scores.append(f1_score(labels[attribute][positions], prediction, average="macro"))
    joint = np.ones(len(positions), dtype=bool)
    for attribute in ATTRIBUTES:
        joint &= logits[attribute].argmax(axis=1) == labels[attribute][positions]
    return float(np.mean(scores)), float(joint.mean())


def train_model(
    model,
    features,
    labels,
    train_positions,
    validation_positions,
    *,
    learning_rate,
    weight_decay,
    batch_size,
    max_epochs,
    patience,
    seed,
    device,
):
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
                labels[attribute][train_positions], model.heads[attribute][-1].out_features
            ).to(device)
        )
        for attribute in ATTRIBUTES
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best = None
    history = []
    stale = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch_features, batch_labels, _ in loader:
            batch_features = _move_features(batch_features, device)
            batch_labels = {
                key: value.to(device) for key, value in batch_labels.items()
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)["logits"]
            loss = sum(losses[key](logits[key], batch_labels[key]) for key in ATTRIBUTES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = len(next(iter(batch_labels.values())))
            total_loss += float(loss.detach().cpu()) * batch_count
            samples += batch_count
        score, joint = validation_score(
            model, features, labels, validation_positions, batch_size, device
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "validation_macro_f1_mean": score,
            "validation_joint_accuracy": joint,
        }
        history.append(row)
        ranking = (score, joint)
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "epoch": epoch,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best["state_dict"])
    return {
        "model": model,
        "best_epoch": best["epoch"],
        "best_macro_f1_mean": best["ranking"][0],
        "best_joint_accuracy": best["ranking"][1],
        "history": history,
        "elapsed_seconds": time.time() - started,
    }


def fit_temperatures(logits, labels, positions):
    """Fit one scalar temperature per head by validation NLL grid search."""
    temperatures = {}
    candidates = np.geomspace(0.25, 4.0, 81)
    for attribute in ATTRIBUTES:
        values = torch.tensor(logits[attribute], dtype=torch.float32)
        truth = torch.tensor(labels[attribute][positions], dtype=torch.long)
        nll = [
            float(nn.functional.cross_entropy(values / float(temp), truth))
            for temp in candidates
        ]
        temperatures[attribute] = float(candidates[int(np.argmin(nll))])
    return temperatures
