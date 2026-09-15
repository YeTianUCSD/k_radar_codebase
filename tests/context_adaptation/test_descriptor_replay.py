"""Lightweight tests for descriptor-only CPU routing replay."""

from __future__ import annotations

import unittest

import numpy as np

from context_adaptation.bandwidth import median_pairwise_distance, resolve_bandwidth

from tools.context_adaptation.replay_context_routing import (
    bootstrap_runtime,
    route_phase,
)


class DescriptorReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "BASE_CONTEXT": {"CONTEXT_NAME": "normal", "MODEL_CONTEXT_NAME": "seq5"},
            "FEATURE": {"KEYS": ["unused"], "POOLING": ["mean", "std"]},
            "PROJECTION": {"DIM": 2, "SEED": 7, "DTYPE": "float32"},
            "KDE": {
                "NUM_RFF_FEATURES": 512, "BANDWIDTH": 1.0, "SEED": 7,
                "THRESHOLD_QUANTILE": 0.05, "DEFAULT_THRESHOLD": 0.01,
                "MIN_CALIBRATION_SAMPLES": 2,
            },
            "CONTEXT_MEMORY": {"PROVISIONAL_MIN_SAMPLES": 3, "PROVISIONAL_MAX_FRAMES": 8},
            "ROUTER": {
                "FEATURE_EMA_BETA": 0.0, "ROUTING_THRESHOLD_SCALE": 0.85,
                "NEAR_THRESHOLD_SCALE": 0.6, "MIN_SCORE_MARGIN": 0.0,
                "NOVELTY_EVIDENCE_THRESHOLD": 0.5, "NOVELTY_DRIFT": 0.0,
                "MIN_CANDIDATE_SAMPLES": 2, "CANDIDATE_EVIDENCE_THRESHOLD": 1.0,
                "CANDIDATE_SIMILARITY_THRESHOLD": 0.0,
                "CANDIDATE_NUM_RFF_FEATURES": 256, "CANDIDATE_BANDWIDTH": 1.0,
                "SEED": 7,
            },
            "ONLINE": {"UPDATE_MEMORY_IF_CONFIDENT": True},
        }

    @staticmethod
    def rows(label: str, values: list[list[float]], segment: int = 0):
        return [
            {
                "segment_index": segment,
                "segment_name_for_metrics_only": label,
                "segment_local_step": index,
                "descriptor": np.asarray(value, dtype=np.float32),
            }
            for index, value in enumerate(values, 1)
        ]

    def test_replay_uses_cached_descriptors_without_model(self) -> None:
        discovery = self.rows(
            "seq5",
            [[-0.2, 0.0, 0.1, 0.0], [0.2, 0.0, -0.1, 0.0],
             [0.0, 0.1, 0.0, -0.1], [0.1, -0.1, 0.0, 0.1]],
        )
        projector, memory, router, samples, bandwidth = bootstrap_runtime(
            self.config, discovery, "seq5"
        )
        output, step = route_phase(
            discovery, phase="discovery", projector=projector, memory=memory,
            router=router, config=self.config, start_step=0,
            allow_context_creation=True, max_steps=-1,
        )
        self.assertEqual(samples, 4)
        self.assertEqual(bandwidth["mode"], "fixed")
        self.assertEqual(step, 4)
        self.assertEqual(len(output), 4)
        self.assertTrue(all(row["model_updated"] == 0 for row in output))
        self.assertTrue(all(len(np.asarray(row["descriptor"])) == 4 for row in discovery))


    def test_median_bandwidth_is_scaled_and_bounded(self) -> None:
        features = np.asarray([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        self.assertEqual(median_pairwise_distance(features), 5.0)
        value, info = resolve_bandwidth(
            {
                "BANDWIDTH_MODE": "median",
                "BANDWIDTH_SCALE": 2.0,
                "BANDWIDTH_MIN": 0.1,
                "BANDWIDTH_MAX": 8.0,
            },
            features,
        )
        self.assertEqual(value, 8.0)
        self.assertEqual(info["base_bandwidth"], 5.0)


if __name__ == "__main__":
    unittest.main()
