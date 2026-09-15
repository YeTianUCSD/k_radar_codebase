import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from context_adaptation.dynamic_psp import (
    add_inherited_dynamic_context,
    get_context_manifest,
    iter_scene_residual_banks,
    restore_inherited_contexts,
)
from models.superposition import (
    PSPConv2d,
    PSPLinear,
    PSPLayerNorm,
    SceneResidualBank,
)
from models.superposition.scene_context import SceneContextRegistry
from tools.superposition.oracle_sequential import (
    Scene,
    load_raw_or_oracle_checkpoint,
    parameter_snapshot,
    restore_parameter_snapshot,
    save_oracle_checkpoint,
)


class ToyPSPNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuser = nn.Module()
        self.fuser.linear = PSPLinear(
            5, 4, key_name="fuser.linear", seed=17, scene_names=["seq5"]
        )
        self.fuser.conv = PSPConv2d(
            3, 2, 3, padding=1, key_name="fuser.conv", seed=17,
            scene_names=["seq5"]
        )
        self.fuser.norm = PSPLayerNorm(4, key_name="fuser.norm", seed=17)
        self.fuser.aware_query = nn.Parameter(torch.randn(1, 2, 4))
        self.fuser.aware_query_scene_bank = SceneResidualBank(
            (1, 2, 4), ["seq5"]
        )
        self.default_scene_context = "seq5"

    def forward_context(self, context_name, linear_input, conv_input, norm_input):
        self.fuser.linear.set_scene_context(context_name)
        self.fuser.conv.set_scene_context(context_name)
        self.fuser.norm.set_scene_context(context_name)
        query = (
            self.fuser.aware_query
            + self.fuser.aware_query_scene_bank.get(context_name)
        )
        return (
            self.fuser.linear(linear_input),
            self.fuser.conv(conv_input),
            self.fuser.norm(norm_input),
            query,
        )


class InheritedDynamicContextTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.network = ToyPSPNetwork()
        with torch.no_grad():
            for _, bank in iter_scene_residual_banks(self.network):
                bank.get("seq5").normal_(0.0, 0.1)
        for parameter in self.network.parameters():
            parameter.requires_grad = False
        self.inputs = (
            torch.randn(3, 5),
            torch.randn(2, 3, 6, 6),
            torch.randn(3, 4),
        )

    def assert_outputs_close(self, left, right):
        for left_tensor, right_tensor in zip(left, right):
            torch.testing.assert_close(
                left_tensor, right_tensor, rtol=1e-5, atol=1e-6
            )

    def test_function_preserving_inheritance_and_isolation(self):
        parent_output = self.network.forward_context("seq5", *self.inputs)
        result, audit = add_inherited_dynamic_context(
            self.network, "seq1", "seq5", trainable_scope="full"
        )
        child_output = self.network.forward_context("seq1", *self.inputs)

        self.assert_outputs_close(parent_output, child_output)
        self.assertEqual(get_context_manifest(self.network), ("seq5", "seq1"))
        self.assertLessEqual(audit.max_effective_parameter_diff, 1e-6)
        self.assertGreater(audit.compared_tensor_count, 0)
        self.assertGreater(len(result.trainable_parameter_names), 0)

        for name, parameter in self.network.named_parameters():
            self.assertEqual(
                parameter.requires_grad,
                ".params.scene_0001" in name,
                msg=name,
            )

        conv = self.network.fuser.conv
        parent_sign = conv.scene_context_registry.get_tensor(
            "seq5", conv.key_name, conv.weight.shape[1:],
            conv.weight.device, conv.weight.dtype
        )
        child_sign = conv.scene_context_registry.get_tensor(
            "seq1", conv.key_name, conv.weight.shape[1:],
            conv.weight.device, conv.weight.dtype
        )
        self.assertFalse(torch.equal(parent_sign, child_sign))

    def test_previous_context_chain_and_restore(self):
        add_inherited_dynamic_context(
            self.network, "seq1", "seq5", trainable_scope="full"
        )
        with torch.no_grad():
            for _, bank in iter_scene_residual_banks(self.network):
                bank.get("seq1").add_(0.01)
        seq1_output = self.network.forward_context("seq1", *self.inputs)
        add_inherited_dynamic_context(
            self.network, "seq22", "seq1", trainable_scope="full"
        )
        seq22_output = self.network.forward_context("seq22", *self.inputs)
        self.assert_outputs_close(seq1_output, seq22_output)

        saved_state = {
            name: value.detach().clone()
            for name, value in self.network.state_dict().items()
        }
        restored = ToyPSPNetwork()
        restore_inherited_contexts(
            restored,
            ("seq5", "seq1", "seq22"),
            {"seq1": "seq5", "seq22": "seq1"},
            trainable_scope="full",
        )
        restored.load_state_dict(saved_state, strict=True)
        restored_output = restored.forward_context("seq22", *self.inputs)
        self.assert_outputs_close(seq22_output, restored_output)

    def test_oracle_v2_checkpoint_round_trip(self):
        add_inherited_dynamic_context(
            self.network, "seq1", "seq5", trainable_scope="full"
        )
        with torch.no_grad():
            for name, parameter in self.network.named_parameters():
                if ".params.scene_0001" in name:
                    parameter.add_(0.02)
        add_inherited_dynamic_context(
            self.network, "seq22", "seq1", trainable_scope="full"
        )
        expected = self.network.forward_context("seq22", *self.inputs)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "toy.oracle.checkpoint"
            save_oracle_checkpoint(
                checkpoint,
                network=self.network,
                base_scene=Scene("seq5", "5"),
                processed_scenes=["seq1", "seq22"],
                context_initialization="inherit_previous",
                context_parents={"seq1": "seq5", "seq22": "seq1"},
                scene_to_sequence={"seq5": "5", "seq1": "1", "seq22": "22"},
                step_idx=2,
                update_idx=2,
                experiment_config=Path("toy.yml"),
            )
            restored = ToyPSPNetwork()
            payload = load_raw_or_oracle_checkpoint(
                restored,
                checkpoint,
                base_scene=Scene("seq5", "5"),
                trainable_scope="full",
            )
            actual = restored.forward_context("seq22", *self.inputs)
            self.assert_outputs_close(expected, actual)
            self.assertEqual(payload["oracle_checkpoint_version"], 2)
            self.assertEqual(
                payload["context_parents"],
                {"seq1": "seq5", "seq22": "seq1"},
            )

    def test_selected_context_snapshot_restores_only_active_residuals(self):
        result, _ = add_inherited_dynamic_context(
            self.network, "seq1", "seq5", trainable_scope="full"
        )
        snapshot = parameter_snapshot(
            self.network, result.trainable_parameter_names
        )
        untouched_before = self.network.fuser.linear.weight.detach().clone()
        with torch.no_grad():
            for name, parameter in self.network.named_parameters():
                if name in snapshot:
                    parameter.add_(1.0)
        restore_parameter_snapshot(self.network, snapshot)
        for name, parameter in self.network.named_parameters():
            if name in snapshot:
                torch.testing.assert_close(parameter, snapshot[name])
        torch.testing.assert_close(
            self.network.fuser.linear.weight, untouched_before
        )

    def test_compatible_base_load_prunes_extra_context_residuals(self):
        add_inherited_dynamic_context(
            self.network, "seq1", "seq5", trainable_scope="full"
        )
        expected = self.network.forward_context("seq5", *self.inputs)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "two_contexts.checkpoint"
            torch.save({"model_state_dict": self.network.state_dict()}, checkpoint)
            restored = ToyPSPNetwork()
            payload = load_raw_or_oracle_checkpoint(
                restored,
                checkpoint,
                base_scene=Scene("seq5", "5"),
                trainable_scope="full",
                checkpoint_load_policy="compatible_base_context",
            )
            self.assertIsNone(payload)
            self.assertEqual(get_context_manifest(restored), ("seq5",))
            actual = restored.forward_context("seq5", *self.inputs)
            self.assert_outputs_close(expected, actual)

    def test_alias_cycle_is_rejected(self):
        registry = SceneContextRegistry(seed=3)
        registry.register_alias("seq1", "seq5")
        with self.assertRaises(RuntimeError):
            registry.register_alias("seq5", "seq1")


if __name__ == "__main__":
    unittest.main()
