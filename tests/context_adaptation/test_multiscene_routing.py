"""Tests for multi-scene stream overrides and routing summaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from easydict import EasyDict

from context_adaptation.stream import (
    StreamSegment,
    build_segment_dataset,
    load_stream_manifest,
)
from tools.context_adaptation.summarize_multiscene_routing import (
    summarize,
    write_outputs,
)


def routing_row(
    segment_index: int,
    label: str,
    local_step: int,
    context_id: int,
    *,
    status: str = "stay",
    created_context_id: int | None = None,
) -> dict[str, object]:
    return {
        "segment_index": segment_index,
        "segment_name_for_metrics_only": label,
        "segment_local_step": local_step,
        "status": status,
        "active_context_id": context_id,
        "created_context_id": created_context_id,
    }


class CapturingDataset:
    last_portion = None
    last_split = None

    def __init__(self, cfg: EasyDict, split: str) -> None:
        type(self).last_portion = list(cfg.DATASET.portion)
        type(self).last_split = split


class MultiSceneRoutingTest(unittest.TestCase):
    def test_stream_manifest_sequence_override_is_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "stream.yml"
            manifest.write_text(
                "STREAM:\n"
                "  - CONFIG: base.yml\n"
                "    SEQUENCES: ['5', '22']\n"
                "    SPLIT: test\n"
                "    NAME_FOR_METRICS_ONLY: mixed\n"
                "  - CONFIG: base.yml\n"
                "    SPLIT: train\n"
            )
            segments = load_stream_manifest(manifest, repository_root=root)
        self.assertEqual(segments[0].sequences, ("5", "22"))
        self.assertEqual(segments[0].split, "test")
        self.assertEqual(segments[1].sequences, ())
        self.assertEqual(segments[1].split, "train")

    def test_stream_manifest_rejects_scalar_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "invalid.yml"
            manifest.write_text(
                "STREAM:\n"
                "  - CONFIG: base.yml\n"
                "    SEQUENCES: '5'\n"
            )
            with self.assertRaisesRegex(ValueError, "must be a list"):
                load_stream_manifest(manifest, repository_root=root)

    def test_build_segment_dataset_applies_sequence_override(self) -> None:
        cfg = EasyDict(
            {
                "DATASET": {
                    "NAME": "CapturingDataset",
                    "portion": ["original"],
                }
            }
        )
        segment = StreamSegment(
            config_path="unused.yml",
            split="test",
            sequences=("5", "22"),
        )
        registry = {"CapturingDataset": CapturingDataset}
        with mock.patch(
            "context_adaptation.stream.load_fresh_config",
            return_value=cfg,
        ), mock.patch("context_adaptation.stream.datasets.__all__", registry):
            build_segment_dataset(segment)
        self.assertEqual(CapturingDataset.last_portion, ["5", "22"])
        self.assertEqual(CapturingDataset.last_split, "test")

    @staticmethod
    def perfect_discovery_rows() -> list[dict[str, object]]:
        return [
            routing_row(0, "seq5", 1, 0),
            routing_row(0, "seq5", 2, 0),
            routing_row(0, "seq5", 3, 0),
            routing_row(1, "seq1", 1, 0, status="pending"),
            routing_row(
                1,
                "seq1",
                2,
                1,
                status="create",
                created_context_id=1,
            ),
            routing_row(1, "seq1", 3, 1),
            routing_row(2, "seq22", 1, 1, status="pending"),
            routing_row(
                2,
                "seq22",
                2,
                2,
                status="create",
                created_context_id=2,
            ),
            routing_row(2, "seq22", 3, 2),
        ]

    @staticmethod
    def perfect_return_rows() -> list[dict[str, object]]:
        return [
            routing_row(0, "seq22", 1, 1, status="pending"),
            routing_row(0, "seq22", 2, 2, status="switch"),
            routing_row(0, "seq22", 3, 2),
            routing_row(1, "seq5", 1, 2, status="pending"),
            routing_row(1, "seq5", 2, 0, status="switch"),
            routing_row(1, "seq5", 3, 0),
            routing_row(2, "seq1", 1, 0, status="pending"),
            routing_row(2, "seq1", 2, 1, status="switch"),
            routing_row(2, "seq1", 3, 1),
        ]

    def test_perfect_discovery_and_return(self) -> None:
        payload = summarize(
            self.perfect_discovery_rows(),
            self.perfect_return_rows(),
            expected_labels=["seq5", "seq1", "seq22"],
            base_label="seq5",
            base_context_id=0,
            ignore_transition_frames=1,
        )
        self.assertEqual(payload["discovery"]["num_contexts"], 3)
        self.assertEqual(payload["discovery"]["new_scene_discovery_recall"], 1.0)
        self.assertEqual(payload["discovery"]["stable_frame_accuracy"], 1.0)
        self.assertEqual(payload["return"]["stable_frame_accuracy"], 1.0)
        self.assertEqual(payload["return"]["false_context_creations"], 0)
        self.assertTrue(payload["pass_criteria"]["overall_pass"])

    def test_summary_outputs_are_written(self) -> None:
        payload = summarize(
            self.perfect_discovery_rows(),
            self.perfect_return_rows(),
            expected_labels=["seq5", "seq1", "seq22"],
            ignore_transition_frames=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_outputs(payload, output_dir)
            expected = {
                "routing_metrics.yml",
                "context_label_mapping.yml",
                "discovery_confusion.csv",
                "return_confusion.csv",
                "creation_events.csv",
                "switch_delays.csv",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected,
            )

    def test_false_return_creation_reports_fragmentation(self) -> None:
        returned = self.perfect_return_rows()
        returned[-1] = routing_row(
            2,
            "seq1",
            3,
            3,
            status="create",
            created_context_id=3,
        )
        payload = summarize(
            self.perfect_discovery_rows(),
            returned,
            expected_labels=["seq5", "seq1", "seq22"],
            ignore_transition_frames=1,
        )
        self.assertEqual(payload["return"]["false_context_creations"], 1)
        self.assertEqual(
            payload["fragmentations"],
            [{"label": "seq1", "context_ids": [1, 3]}],
        )
        self.assertFalse(payload["pass_criteria"]["diagnostics_not_used_for_overall_pass"]["no_false_creations_on_return"])
        self.assertFalse(payload["pass_criteria"]["diagnostics_not_used_for_overall_pass"]["no_fragmentation"])

    def test_missing_creation_reports_merging(self) -> None:
        discovery = self.perfect_discovery_rows()
        discovery[-2] = routing_row(2, "seq22", 2, 1)
        discovery[-1] = routing_row(2, "seq22", 3, 1)
        payload = summarize(
            discovery,
            self.perfect_return_rows(),
            expected_labels=["seq5", "seq1", "seq22"],
            ignore_transition_frames=1,
        )
        self.assertEqual(payload["discovery"]["missing_new_labels"], ["seq22"])
        self.assertEqual(
            payload["merges"],
            [{"context_id": 1, "labels": ["seq1", "seq22"]}],
        )
        self.assertFalse(payload["pass_criteria"]["all_scenes_have_pure_discovery_context"])
        self.assertFalse(payload["pass_criteria"]["diagnostics_not_used_for_overall_pass"]["no_merging_diagnostic"])

    def test_secondary_merged_context_fails_supported_context_purity(self) -> None:
        discovery = self.perfect_discovery_rows()
        discovery.extend([
            routing_row(1, "seq1", 4, 3),
            routing_row(2, "seq22", 4, 3),
        ])
        payload = summarize(
            discovery,
            self.perfect_return_rows(),
            expected_labels=["seq5", "seq1", "seq22"],
            base_label="seq5",
            base_context_id=0,
            ignore_transition_frames=1,
            minimum_context_purity=0.95,
            minimum_context_support=2,
        )
        self.assertEqual(
            payload["context_purity_violations"][0]["context_id"], 3
        )
        self.assertFalse(
            payload["pass_criteria"]["all_supported_contexts_meet_purity"]
        )
        self.assertFalse(payload["pass_criteria"]["overall_pass"] )


if __name__ == "__main__":
    unittest.main()
