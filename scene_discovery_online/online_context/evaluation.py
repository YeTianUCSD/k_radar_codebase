"""Simple phase, context-routing, and causal change-point metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def attach_decisions(stream: pd.DataFrame, decisions: Iterable[object]):
    rows = [decision.to_dict() for decision in decisions]
    if len(rows) != len(stream):
        raise ValueError("decision count does not match stream length")
    output = pd.concat(
        [stream.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)],
        axis=1,
    )
    output["context_correct"] = (
        output["active_key"].astype(str)
        == output["true_semantic_key"].astype(str)
    )
    output["provisional_correct"] = (
        output["provisional_key"].astype(str)
        == output["true_semantic_key"].astype(str)
    )
    predicted = output["provisional_key"].str.split("|", expand=True)
    predicted.columns = ["predicted_weather", "predicted_road", "predicted_lighting"]
    output = pd.concat([output, predicted], axis=1)
    for attribute in ("weather", "road", "lighting"):
        output[f"{attribute}_correct"] = (
            output[f"predicted_{attribute}"].astype(str)
            == output[f"true_{attribute}"].astype(str)
        )
    return output


def block_metrics(frame: pd.DataFrame, stable_grace_frames: int):
    rows = []
    for block_id, block in frame.groupby("block_id", sort=True):
        grace = block["block_local_frame"].astype(int) >= int(stable_grace_frames)
        stable = block.loc[grace]
        creations = block[block["event"].eq("create")]
        switches = block[block["event"].eq("switch")]
        truth = str(block["true_semantic_key"].iloc[0])
        ending_correct = str(block["active_key"].iloc[-1]) == truth
        rows.append({
            "block_id": int(block_id), "phase": str(block["phase"].iloc[0]),
            "visit": str(block["visit"].iloc[0]),
            "sequence": str(block["sequence"].iloc[0]),
            "frames": int(len(block)), "true_semantic_key": truth,
            "expected_novel": bool(block["expected_novel"].iloc[0]),
            "raw_context_accuracy": float(block["context_correct"].mean()),
            "stable_context_accuracy": (
                float(stable["context_correct"].mean()) if len(stable) else float("nan")
            ),
            "provisional_attribute_joint_accuracy": float(
                block["provisional_correct"].mean()
            ),
            "ending_context_correct": bool(ending_correct),
            "creation_events": int(len(creations)),
            "switch_events": int(len(switches)),
            "pending_frames": int(block["event"].eq("pending").sum()),
            "first_correct_frame": (
                int(np.flatnonzero(block["context_correct"].to_numpy())[0])
                if block["context_correct"].any() else -1
            ),
        })
    return pd.DataFrame(rows)


def phase_metrics(frame: pd.DataFrame, blocks: pd.DataFrame):
    rows = []
    for phase, values in frame.groupby("phase", sort=False):
        selected = blocks[blocks["phase"].eq(phase)]
        rows.append({
            "phase": phase, "frames": int(len(values)),
            "blocks": int(len(selected)),
            "raw_context_accuracy": float(values["context_correct"].mean()),
            "provisional_attribute_joint_accuracy": float(
                values["provisional_correct"].mean()
            ),
            "weather_accuracy": float(values["weather_correct"].mean()),
            "road_accuracy": float(values["road_correct"].mean()),
            "lighting_accuracy": float(values["lighting_correct"].mean()),
            "stable_context_accuracy": float(
                selected["stable_context_accuracy"].mean()
            ),
            "ending_block_accuracy": float(
                selected["ending_context_correct"].mean()
            ),
            "creation_events": int(values["event"].eq("create").sum()),
            "switch_events": int(values["event"].eq("switch").sum()),
            "pending_frames": int(values["event"].eq("pending").sum()),
            "update_enabled_rate": float(values["update_enabled"].mean()),
        })
    return pd.DataFrame(rows)


def change_metrics(frame: pd.DataFrame, tolerance_frames: int):
    boundaries = frame.index[frame["true_boundary"].astype(bool)].to_numpy(dtype=int)
    predicted = frame.index[
        frame["event"].isin(["create", "switch"])
    ].to_numpy(dtype=int)
    used = set()
    delays = []
    matches = []
    for boundary in boundaries:
        candidates = [
            (position, int(value)) for position, value in enumerate(predicted)
            if position not in used
            and int(boundary) <= int(value) <= int(boundary) + int(tolerance_frames)
        ]
        if not candidates:
            matches.append((int(boundary), -1, -1))
            continue
        position, detected = candidates[0]
        used.add(position)
        delay = detected - int(boundary)
        delays.append(delay)
        matches.append((int(boundary), detected, delay))
    tp = len(used)
    precision = _safe(tp, len(predicted))
    recall = _safe(tp, len(boundaries))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    metrics = {
        "true_changes": int(len(boundaries)),
        "predicted_changes": int(len(predicted)),
        "matched_changes": int(tp),
        "false_changes": int(len(predicted) - tp),
        "missed_changes": int(len(boundaries) - tp),
        "change_precision": precision, "change_recall": recall,
        "change_f1": float(f1),
        "mean_detection_delay": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay": float(np.median(delays)) if delays else float("nan"),
        "false_changes_per_1000_frames": _safe(
            (len(predicted) - tp) * 1000, len(frame)
        ),
    }
    match_frame = pd.DataFrame(
        matches, columns=["true_boundary_frame", "detected_frame", "delay_frames"]
    )
    return metrics, match_frame


def summarize_run(frame, registry_size, stable_grace_frames, tolerance_frames):
    blocks = block_metrics(frame, stable_grace_frames)
    phases = phase_metrics(frame, blocks)
    changes, matches = change_metrics(frame, tolerance_frames)
    first = blocks[blocks["phase"].eq("first_visit_train")]
    second = blocks[blocks["phase"].eq("second_visit_train")]
    third = blocks[blocks["phase"].eq("third_visit_test")]
    summary = {
        **changes,
        "final_context_count": int(registry_size),
        "first_visit_contexts": int(len(first)),
        "first_visit_correct_creations": int(
            ((first["creation_events"] == 1) & first["ending_context_correct"]).sum()
        ),
        "first_visit_creation_recall": float(
            ((first["creation_events"] == 1) & first["ending_context_correct"]).mean()
        ) if len(first) else float("nan"),
        "first_visit_duplicate_creations": int(
            np.maximum(first["creation_events"].to_numpy() - 1, 0).sum()
        ) if len(first) else 0,
        "second_visit_scene_reuse_accuracy": float(
            ((second["creation_events"] == 0) & second["ending_context_correct"]).mean()
        ) if len(second) else float("nan"),
        "second_visit_false_creations": int(second["creation_events"].sum()),
        "test_scene_reuse_accuracy": float(
            ((third["creation_events"] == 0) & third["ending_context_correct"]).mean()
        ) if len(third) else float("nan"),
        "test_false_creations": int(third["creation_events"].sum()),
        "test_raw_context_accuracy": float(
            frame.loc[frame["phase"].eq("third_visit_test"), "context_correct"].mean()
        ) if len(third) else float("nan"),
        "test_stable_context_accuracy": float(third["stable_context_accuracy"].mean())
        if len(third) else float("nan"),
    }
    return summary, blocks, phases, matches
