import unittest

import numpy as np

from online_context.controller import OnlineContextController


LABELS = {
    "weather": ("normal", "snow"),
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


class AsymmetricContextTests(unittest.TestCase):
    def test_unknown_creation_is_stricter_than_registered_switch(self):
        base = ("normal", "urban", "night")
        novel = ("snow", "highway", "night")
        controller = OnlineContextController(
            LABELS,
            base,
            decision_window=5,
            confirmation_frames=3,
            majority_ratio=0.8,
            new_context_confirmation_frames=5,
            new_context_majority_ratio=1.0,
        )

        decisions = []
        stream = [base] * 5 + [novel] * 4 + [base] + [novel] * 9
        for step, key in enumerate(stream):
            decisions.append(controller.observe(probabilities(key), step))
        self.assertEqual(sum(item.event == "create" for item in decisions), 1)
        self.assertEqual(controller.active_entry.key, novel)

        offset = len(stream)
        decisions = [
            controller.observe(probabilities(base), offset + step)
            for step in range(6)
        ]
        self.assertEqual(sum(item.event == "switch" for item in decisions), 1)
        self.assertEqual(controller.active_entry.key, base)


if __name__ == "__main__":
    unittest.main()
