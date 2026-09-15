import unittest

import numpy as np

from online_context.controller import OnlineContextController


LABELS = {
    "weather": ("normal", "snow"),
    "road": ("urban", "highway"),
    "lighting": ("day", "night"),
}


def probabilities(key, confidence):
    output = {}
    for attribute, label in zip(("weather", "road", "lighting"), key):
        position = LABELS[attribute].index(label)
        values = np.full(2, 1.0 - confidence, dtype=np.float32)
        values[position] = confidence
        output[attribute] = values
    return output


class ConfidenceAbstainTests(unittest.TestCase):
    def test_low_confidence_frames_do_not_vote_or_create_context(self):
        base = ("normal", "urban", "night")
        novel = ("snow", "highway", "night")
        controller = OnlineContextController(
            LABELS, base, decision_window=5, confirmation_frames=3,
            majority_ratio=0.8,
            confidence_thresholds={key: 0.9 for key in LABELS},
            confidence_policy="enforce",
        )
        decisions = [
            controller.observe(probabilities(novel, 0.8), step)
            for step in range(10)
        ]
        self.assertTrue(all(item.event == "abstain" for item in decisions))
        self.assertTrue(all(not item.update_enabled for item in decisions))
        self.assertEqual(len(controller.registry), 1)
        self.assertEqual(controller.active_entry.key, base)

        decisions = [
            controller.observe(probabilities(novel, 0.99), step)
            for step in range(10, 20)
        ]
        self.assertEqual(sum(item.event == "create" for item in decisions), 1)
        self.assertEqual(controller.active_entry.key, novel)


if __name__ == "__main__":
    unittest.main()
