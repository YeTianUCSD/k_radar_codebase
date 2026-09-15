import unittest

from context_adaptation.semantic_superposition.checkpoint_protocol import (
    build_protocol_signature,
    validate_protocol_signature,
)


class SemanticCheckpointProtocolTest(unittest.TestCase):
    def signature(self, config=None, version=3):
        return build_protocol_signature(
            pipeline_id="automatic_semantic_sequential_v3",
            checkpoint_version=version,
            experiment_config=config or {"STREAM": {"VISITS": ["train_first"]}},
            selected_scenes=["seq35", "seq46"],
            init_model="/tmp/base.checkpoint",
            execution_controls={"max_steps_per_scene": -1},
        )

    def test_identical_protocol_validates(self):
        signature = self.signature()
        validate_protocol_signature(signature, self.signature())

    def test_changed_config_is_rejected(self):
        stored = self.signature()
        expected = self.signature({"STREAM": {"VISITS": ["test_revisit"]}})
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_protocol_signature(stored, expected)

    def test_missing_legacy_signature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "V2 and legacy"):
            validate_protocol_signature(None, self.signature())

    def test_corrupt_signature_is_rejected(self):
        stored = self.signature()
        stored["payload"]["selected_scenes"].append("seq19")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            validate_protocol_signature(stored, self.signature())


if __name__ == "__main__":
    unittest.main()
