import inspect
import unittest

import numpy as np
import pandas as pd

from online_context.controller import OnlineContextController
from online_context.registry import ContextRegistry
from online_context.stream import build_three_pass_stream


LABELS = {
    "weather": ("normal", "heavy_snow"),
    "road": ("urban", "highway"),
    "lighting": ("day", "night"),
}


def probabilities(key):
    output = {}
    for attribute, value in zip(("weather", "road", "lighting"), key):
        vector = np.full(len(LABELS[attribute]), 0.01, dtype=np.float32)
        vector[LABELS[attribute].index(value)] = 0.99
        output[attribute] = vector / vector.sum()
    return output


class RegistryTests(unittest.TestCase):
    def test_registry_reuses_exact_semantic_key(self):
        registry = ContextRegistry(("normal", "urban", "night"))
        base, created = registry.get_or_create(("normal", "urban", "night"), 5)
        self.assertFalse(created)
        self.assertEqual(base.context_id, 0)
        snow, created = registry.get_or_create(
            ("heavy_snow", "highway", "night"), 10
        )
        self.assertTrue(created)
        reused, created = registry.get_or_create(snow.key, 20)
        self.assertFalse(created)
        self.assertEqual(reused.context_id, snow.context_id)


class ControllerTests(unittest.TestCase):
    def test_create_then_revisit_switches_to_original_id(self):
        base = ("normal", "urban", "night")
        novel = ("heavy_snow", "highway", "night")
        controller = OnlineContextController(
            LABELS, base, decision_window=5, confirmation_frames=3,
            majority_ratio=0.8, confidence_policy="observe_only",
        )
        decisions = []
        step = 0
        for key in ([base] * 8 + [novel] * 10 + [base] * 10):
            decisions.append(controller.observe(probabilities(key), step))
            step += 1
        creations = [item for item in decisions if item.event == "create"]
        switches = [item for item in decisions if item.event == "switch"]
        self.assertEqual(len(creations), 1)
        self.assertEqual(creations[0].active_key, novel)
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0].context_id, 0)
        self.assertEqual(len(controller.registry), 2)

    def test_controller_api_has_no_truth_or_boundary_input(self):
        parameters = set(inspect.signature(OnlineContextController.observe).parameters)
        self.assertEqual(parameters, {"self", "probabilities", "step"})

    def test_pending_frames_disable_updates(self):
        base = ("normal", "urban", "night")
        novel = ("heavy_snow", "highway", "night")
        controller = OnlineContextController(
            LABELS, base, decision_window=5, confirmation_frames=3,
            majority_ratio=0.8, confidence_policy="observe_only",
            pause_update_when_pending=True,
        )
        for step in range(5):
            controller.observe(probabilities(base), step)
        decision = controller.observe(probabilities(novel), 5)
        self.assertEqual(decision.event, "pending")
        self.assertFalse(decision.update_enabled)


class StreamTests(unittest.TestCase):
    @staticmethod
    def index(split):
        rows = []
        for sequence, climate, road, lighting in (
            (1, "normal", "urban", "night"),
            (5, "heavy snow", "highway", "night"),
        ):
            for local in range(4):
                rows.append({
                    "row_index": len(rows),
                    "source_row_index": local,
                    "sample_id": f"seq{sequence}_{split}_{local}",
                    "sequence": sequence, "climate": climate,
                    "road_type": road, "capture_time": lighting,
                })
        return pd.DataFrame(rows)

    def test_support_is_excluded_and_train_is_replayed(self):
        train, test = self.index("train"), self.index("test")
        support = pd.DataFrame([
            {"sample_id": "seq1_train_1", "source_split": "train",
             "role": "support_train", "support_budget": 1},
            {"sample_id": "seq5_train_1", "source_split": "train",
             "role": "support_train", "support_budget": 1},
        ])
        stream, orders, support_ids = build_three_pass_stream(
            train, test, support, [1, 5], 1, 7
        )
        self.assertEqual(len(support_ids), 2)
        self.assertFalse(stream["sample_id"].isin(support_ids).any())
        first = stream[stream["phase"].eq("first_visit_train")]
        second = stream[
            stream["phase"].eq("second_visit_train")
            & stream["sequence"].eq("5")
        ]
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(orders["first_visit_train"], ["5"])


if __name__ == "__main__":
    unittest.main()
