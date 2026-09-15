"""Leakage-resistant stream construction for open-world weather discovery."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .attributes import normalize_attribute, sequence_attribute_table
from .online_protocol import normalize_sequence


def sequence_weather_mapping(index: pd.DataFrame) -> dict:
    """Return one normalized weather label per sequence for scoring only."""
    table = sequence_attribute_table(index)
    return {
        normalize_sequence(row.sequence): str(row.weather)
        for row in table.itertuples(index=False)
    }


def build_weather_orders(
    base_order: Sequence[object],
    initial_scene: object,
    weather_by_sequence: Mapping[str, str],
    random_orders: int,
    seed: int,
):
    """Build first-visit train and disjoint test-revisit weather streams.

    Novelty is defined by the first occurrence of a weather value, not by the
    first occurrence of a sequence.  Sequence and weather fields are retained
    only in the replay/evaluation metadata and never enter the manager.
    """
    if random_orders < 0:
        raise ValueError("random_orders must be non-negative")
    base = [normalize_sequence(value) for value in base_order]
    initial = normalize_sequence(initial_scene)
    if not base or len(base) != len(set(base)):
        raise ValueError("base_order must contain unique scenes")
    if initial not in base:
        raise ValueError("initial_scene must occur in base_order")
    missing = set(base).difference(weather_by_sequence)
    if missing:
        raise ValueError(f"weather mapping is missing scenes: {sorted(missing)}")
    weather = {
        scene: normalize_attribute("weather", weather_by_sequence[scene])
        for scene in base
    }
    discovery = [scene for scene in base if scene != initial]

    def make_blocks(first_order, revisit_order):
        seen_weather = {weather[initial]}
        rows = []
        for scene in first_order:
            target = weather[scene]
            novel = target not in seen_weather
            rows.append({
                "sequence": scene,
                "weather": target,
                "split": "train",
                "visit": "novel_weather" if novel else "known_weather",
                "expected_novel": novel,
            })
            seen_weather.add(target)
        for scene in revisit_order:
            rows.append({
                "sequence": scene,
                "weather": weather[scene],
                "split": "test",
                "visit": "known_weather",
                "expected_novel": False,
            })
        return rows

    orders = [("canonical", make_blocks(discovery, base))]
    rng = np.random.RandomState(seed)
    seen_orders = {(tuple(discovery), tuple(base))}
    attempts = 0
    while len(orders) < random_orders + 1:
        attempts += 1
        first = tuple(rng.permutation(discovery))
        revisit = tuple(rng.permutation(base))
        key = (first, revisit)
        if key in seen_orders:
            if attempts > 1000:
                raise RuntimeError("Could not generate enough unique weather orders")
            continue
        seen_orders.add(key)
        orders.append((f"random_{len(orders):02d}", make_blocks(first, revisit)))
    return orders


def annotate_weather_stream(
    frame: pd.DataFrame,
    weather_by_sequence: Mapping[str, str],
) -> pd.DataFrame:
    """Attach weather truth after inference and expose it as evaluator target."""
    output = frame.copy().reset_index(drop=True)
    sequences = output["stream_true_sequence"].map(normalize_sequence)
    missing = set(sequences).difference(weather_by_sequence)
    if missing:
        raise ValueError(f"weather mapping is missing stream scenes: {sorted(missing)}")
    output["stream_true_weather"] = [
        normalize_attribute("weather", weather_by_sequence[scene])
        for scene in sequences
    ]
    output["stream_true_scene"] = sequences
    # The generic event evaluator deliberately receives only an anonymous
    # target label column.  For V5 that target is weather, never sequence.
    output["stream_true_sequence"] = output["stream_true_weather"]
    return output
