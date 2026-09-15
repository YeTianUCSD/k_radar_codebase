"""CPU-only tests for the automatic context-adaptation core."""

from __future__ import annotations

import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from context_adaptation.checkpoint import (
    CHECKPOINT_VERSION,
    load_context_checkpoint,
    restore_model_and_context_state,
    save_context_checkpoint,
    validate_optimizer_resume,
)
from context_adaptation.context_memory import ContextMemory
from context_adaptation.context_router import ContextRouter, rank_eligible_contexts
from context_adaptation.context_transaction import create_context_transaction
from context_adaptation.dynamic_psp import (
    activate_context_residuals,
    add_context_optimizer_group,
    add_dynamic_context,
    get_context_manifest,
)
from context_adaptation.feature_projector import RandomFeatureProjector
from context_adaptation.rff_kde import OnlineRFFKDE
from context_adaptation.stream import StreamSegment, iter_stream_batches
from models.superposition import SceneResidualBank, SceneWeightResidualBank
from tools.context_adaptation.eval_auto_context import EvaluationRouter


class ToyPSP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fuser = nn.Module()
        self.fuser.aware_query_scene_bank = SceneResidualBank((3,), ["normal"])
        self.head = nn.Module()
        self.head.scene_weight_bank = SceneWeightResidualBank((2, 3), ["normal"])
        self.default_scene_context = "normal"


class ToyStreamDataset:
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> int:
        return index

    @staticmethod
    def collate_fn(values: list[int]) -> list[int]:
        return values


class ContextAdaptationCoreTest(unittest.TestCase):
    def test_stream_time_window_slicing(self) -> None:
        segment = StreamSegment(
            config_path="unused.yml",
            split="test",
            name_for_metrics_only="normal",
            start=2,
            stop=6,
        )
        batches = list(
            iter_stream_batches(
                [segment],
                batch_size=1,
                dataset_overrides={0: ToyStreamDataset()},
            )
        )
        self.assertEqual([batch.batch[0] for batch in batches], [2, 3, 4, 5])
        self.assertEqual([batch.local_step for batch in batches], [1, 2, 3, 4])
        self.assertTrue(
            all(batch.segment_name_for_metrics_only == "normal" for batch in batches)
        )

    def test_kde_state_and_disk_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        samples = rng.normal(0.0, 0.2, size=(100, 2))
        kde = OnlineRFFKDE(2, n_features=2048, bandwidth=0.5, random_state=7)
        kde.update(samples)
        self.assertGreater(
            kde.query_kernel_mean([0.0, 0.0]),
            kde.query_kernel_mean([4.0, 4.0]),
        )
        restored = OnlineRFFKDE.from_state_dict(kde.state_dict())
        np.testing.assert_allclose(kde.feature_sum, restored.feature_sum)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "kde.npz"
            kde.save(path)
            loaded = OnlineRFFKDE.load(path)
        np.testing.assert_allclose(kde.feature_sum, loaded.feature_sum)

    def test_projector_is_fixed_after_normal_fit(self) -> None:
        projector = RandomFeatureProjector(
            ["camera", "lidar", "radar"],
            projection_dim=2,
            random_state=5,
        )
        batch = {
            "camera": torch.randn(2, 2, 3, 3),
            "lidar": torch.randn(2, 3, 3, 3),
            "radar": torch.randn(2, 4, 3, 3),
        }
        descriptors = projector.extract_descriptor(batch)
        self.assertEqual(descriptors.shape, (2, 18))
        projector.update_normalization(descriptors)
        projector.finalize()
        projected = projector.transform_descriptor(descriptors)
        self.assertEqual(projected.shape, (2, 2))
        restored = RandomFeatureProjector.from_state_dict(projector.state_dict())
        np.testing.assert_allclose(
            projected,
            restored.transform_descriptor(descriptors),
        )
        with self.assertRaises(RuntimeError):
            projector.update_normalization(descriptors)

        constant_projector = RandomFeatureProjector(["feature"], projection_dim=2)
        constant_projector.update_normalization(
            np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        )
        constant_projector.finalize()
        constant_projection = constant_projector.transform_descriptor([1.001, 1.0])
        self.assertLess(float(np.max(np.abs(constant_projection))), 10.0)

    def test_router_ema_and_state_round_trip(self) -> None:
        router = ContextRouter(2, feature_ema_beta=0.5)
        np.testing.assert_allclose(router.prepare_feature([0.0, 0.0]), [0.0, 0.0])
        np.testing.assert_allclose(router.prepare_feature([2.0, 2.0]), [1.0, 1.0])
        restored = ContextRouter.from_state_dict(router.state_dict())
        np.testing.assert_allclose(restored.prepare_feature([1.0, 1.0]), [1.0, 1.0])

    def test_router_accumulates_switch_evidence(self) -> None:
        router = ContextRouter(
            2,
            routing_threshold_scale=0.85,
            switch_evidence_threshold=0.25,
            switch_evidence_margin=0.05,
            switch_cooldown=0,
        )
        router.set_active_context(0)
        first = router.step(
            [0.0, 0.0],
            {0: 0.1, 1: 1.0},
            {0: 1.0, 1: 1.0},
        )
        self.assertEqual(first.status, "pending")
        self.assertEqual(first.reason, "historical_context_evidence_pending")
        second = router.step(
            [0.0, 0.0],
            {0: 0.1, 1: 1.0},
            {0: 1.0, 1: 1.0},
        )
        self.assertEqual(second.status, "switch")
        self.assertEqual(second.context_id, 1)

    def test_router_evidence_decays_without_resetting(self) -> None:
        router = ContextRouter(
            2,
            routing_threshold_scale=0.85,
            switch_evidence_threshold=10.0,
            evidence_miss_penalty=0.01,
        )
        router.set_active_context(0)
        router.step([0.0, 0.0], {0: 0.1, 1: 1.0}, {0: 1.0, 1: 1.0})
        before = router._context_evidence[1]
        router.step([0.0, 0.0], {0: 0.1, 1: 0.8}, {0: 1.0, 1: 1.0})
        after = router._context_evidence[1]
        self.assertGreater(before, after)
        self.assertGreater(after, 0.0)

    def test_provisional_active_context_uses_near_match_for_warmup(self) -> None:
        router = ContextRouter(2)
        router.set_active_context(1)
        decision = router.step(
            [2.0, 2.0],
            {0: 0.1, 1: 0.70},
            {0: 1.0, 1: 1.0},
            context_lifecycles={0: "stable", 1: "provisional"},
        )
        self.assertEqual(decision.status, "stay")
        self.assertEqual(decision.reason, "provisional_context_warmup")
        self.assertTrue(decision.should_update_memory)

    def test_evaluation_router_does_not_mutate_context_memory(self) -> None:
        memory = ContextMemory(2, n_rff_features=64)
        memory.create_context(np.zeros((8, 2)), context_name="normal")
        memory.create_context(np.ones((8, 2)), context_name="context_0001")
        memory.activate(0)
        checkpoint_router = ContextRouter(
            2,
            switch_evidence_threshold=0.25,
            switch_cooldown=0,
        )
        checkpoint_router.set_active_context(0)
        evaluator = EvaluationRouter(memory, checkpoint_router, initial_context_id=0)
        with mock.patch.object(
            memory,
            "score_all",
            return_value={0: 0.0, 1: 1.0},
        ):
            route = evaluator.route([0.0, 0.0])
        self.assertEqual(route["context_id"], 1)
        self.assertEqual(memory.active_context_id, 0)
        self.assertEqual(len(memory), 2)

    def test_default_candidate_evidence_reaches_context_creation(self) -> None:
        router = ContextRouter(2)
        router.set_active_context(0)
        decision = None
        for _ in range(12):
            decision = router.step(
                [3.0, 3.0],
                {0: 0.0},
                {0: 0.1},
            )
            if decision.status == "create":
                break
        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, "create")
        self.assertLessEqual(len(decision.candidate_features), 12)
        router.abort_context_creation()

    def test_router_can_disable_context_creation_for_evaluation(self) -> None:
        router = ContextRouter(
            2,
            min_candidate_samples=1,
            novelty_decay=1.0,
            novelty_drift=0.0,
            novelty_evidence_threshold=0.5,
            candidate_evidence_threshold=0.5,
        )
        router.set_active_context(0)
        decision = router.step(
            [3.0, 3.0],
            {0: 0.0},
            {0: 0.1},
            allow_context_creation=False,
        )
        self.assertEqual(decision.status, "pending")
        self.assertEqual(decision.reason, "novel_context_creation_disabled")
        self.assertEqual(decision.context_id, 0)
        self.assertFalse(router._awaiting_context_creation)

    def test_near_historical_context_suppresses_creation(self) -> None:
        router = ContextRouter(
            2,
            routing_threshold_scale=0.85,
            near_threshold_scale=0.60,
            novelty_evidence_threshold=0.1,
            min_candidate_samples=1,
            candidate_evidence_threshold=0.1,
        )
        router.set_active_context(0)
        for _ in range(5):
            decision = router.step(
                [2.0, 2.0],
                {0: 0.65},
                {0: 1.0},
            )
            self.assertEqual(decision.status, "pending")
            self.assertEqual(decision.reason, "near_historical_context")
        self.assertEqual(router._novelty_evidence, 0.0)

    def test_sustained_ambiguous_match_can_create_split_context(self) -> None:
        router = ContextRouter(
            2, min_score_margin=0.2, ambiguous_creation_enabled=True,
            split_min_candidate_samples=2, split_evidence_threshold=0.0,
            candidate_similarity_threshold=0.0,
        )
        router.set_active_context(0)
        first = router.step(
            [1.0, 1.0], {0: 0.8, 1: 0.79}, {0: 0.5, 1: 0.5}
        )
        self.assertEqual(first.reason, "ambiguous_historical_match")
        restored = ContextRouter.from_state_dict(router.state_dict())
        second = restored.step(
            [1.0, 1.0], {0: 0.8, 1: 0.79}, {0: 0.5, 1: 0.5}
        )
        self.assertEqual(second.status, "create")
        self.assertEqual(second.reason, "ambiguous_context_evidence_confirmed")
        restored.abort_context_creation()

    def test_sustained_near_match_can_create_split_context(self) -> None:
        router = ContextRouter(
            2, near_creation_enabled=True, split_min_candidate_samples=2,
            split_evidence_threshold=0.0, candidate_similarity_threshold=0.0,
        )
        router.set_active_context(0)
        first = router.step([2.0, 2.0], {0: 0.65}, {0: 1.0})
        self.assertEqual(first.reason, "near_historical_context")
        second = router.step([2.0, 2.0], {0: 0.65}, {0: 1.0})
        self.assertEqual(second.status, "create")
        self.assertEqual(second.reason, "near_context_evidence_confirmed")
        router.abort_context_creation()

    def test_sustained_boundary_match_can_create_split_context(self) -> None:
        router = ContextRouter(
            2, strict_update_gate_enabled=True,
            boundary_creation_enabled=True, update_min_confidence=0.2,
            update_min_margin=0.15, split_min_candidate_samples=2,
            split_evidence_threshold=0.0, candidate_similarity_threshold=0.0,
        )
        router.set_active_context(0)
        first = router.step([1.0, 1.0], {0: 0.46}, {0: 0.5})
        self.assertEqual(first.status, "pending")
        self.assertEqual(first.reason, "boundary_historical_match")
        second = router.step([1.0, 1.0], {0: 0.46}, {0: 0.5})
        self.assertEqual(second.status, "create")
        self.assertEqual(second.reason, "boundary_context_evidence_confirmed")
        router.abort_context_creation()

    def test_strict_update_gate_requires_stable_consecutive_confidence(self) -> None:
        router = ContextRouter(
            2, strict_update_gate_enabled=True, update_min_confidence=0.1,
            update_min_margin=0.1, update_min_consecutive_accepts=3,
            update_after_switch_warmup=2,
        )
        router.set_active_context(1)
        decisions = []
        for _ in range(3):
            raw = router.step([0.0, 0.0], {1: 0.9}, {1: 0.5})
            decisions.append(router.apply_update_gate(
                raw, context_id=1, lifecycle="stable"
            ))
        self.assertFalse(decisions[0].should_update_memory)
        self.assertFalse(decisions[1].should_update_memory)
        self.assertTrue(decisions[2].should_update_memory)
        restored = ContextRouter.from_state_dict(router.state_dict())
        self.assertEqual(restored._update_streak, 3)

    def test_strict_update_gate_freezes_base_and_provisional_model(self) -> None:
        router = ContextRouter(
            2, strict_update_gate_enabled=True, update_min_confidence=0.0,
            update_min_margin=0.0, update_min_consecutive_accepts=1,
            update_after_switch_warmup=0, freeze_base_context_memory=True,
            update_provisional_memory=True, update_provisional_model=False,
        )
        router.set_active_context(0)
        raw = router.step([0.0, 0.0], {0: 0.9}, {0: 0.5})
        base = router.apply_update_gate(raw, context_id=0, lifecycle="stable")
        self.assertFalse(base.should_update_memory)
        router.set_active_context(1)
        raw = router.step(
            [0.0, 0.0], {0: 0.0, 1: 0.9}, {0: 0.5, 1: 0.5},
            context_lifecycles={0: "stable", 1: "provisional"},
        )
        provisional = router.apply_update_gate(
            raw, context_id=1, lifecycle="provisional"
        )
        self.assertTrue(provisional.should_update_memory)
        self.assertFalse(provisional.should_update_model)

    def test_strict_update_gate_applies_kde_threshold_to_both_updates(self) -> None:
        router = ContextRouter(
            2, strict_update_gate_enabled=True, update_min_confidence=0.0,
            update_min_margin=0.0, update_min_consecutive_accepts=1,
            update_after_switch_warmup=0,
        )
        router.set_active_context(1)
        raw = router.step([0.0, 0.0], {1: 0.7}, {1: 0.5})
        rejected = router.apply_update_gate(
            raw, context_id=1, lifecycle="stable", update_threshold=0.8
        )
        self.assertFalse(rejected.should_update_memory)
        self.assertFalse(rejected.should_update_model)
        accepted = router.apply_update_gate(
            raw, context_id=1, lifecycle="stable", update_threshold=0.6
        )
        self.assertTrue(accepted.should_update_memory)
        self.assertTrue(accepted.should_update_model)

    def test_contexts_are_filtered_by_their_own_thresholds(self) -> None:
        best_id, best_score, _, _, _ = rank_eligible_contexts(
            {0: 0.60, 1: 0.70},
            {0: 0.50, 1: 0.90},
        )
        self.assertEqual(best_id, 0)
        self.assertEqual(best_score, 0.60)

        router = ContextRouter(2, switch_patience=1)
        router.set_active_context(0)
        decision = router.step(
            [0.0, 0.0],
            {0: 0.60, 1: 0.70},
            {0: 0.50, 1: 0.90},
        )
        self.assertEqual(decision.status, "stay")
        self.assertEqual(decision.context_id, 0)
        self.assertEqual(decision.best_context_id, 0)

    def test_context_memory_uses_separate_update_quantile(self) -> None:
        memory = ContextMemory(
            2, min_calibration_samples=2, threshold_quantile=0.05,
            update_threshold_quantile=0.50,
        )
        entry = memory.create_context(np.asarray([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]
        ]))
        entry.calibration_scores = [0.1, 0.2, 0.8, 0.9]
        entry.density_threshold = 0.1
        self.assertEqual(memory.density_thresholds()[0], 0.1)
        self.assertAlmostEqual(memory.update_thresholds()[0], 0.5)
        restored = ContextMemory.from_state_dict(memory.state_dict())
        self.assertEqual(restored.update_threshold_quantile, 0.5)

    def test_context_memory_lifecycle_and_threshold_gates(self) -> None:
        memory = ContextMemory(
            2,
            n_rff_features=256,
            bandwidth=0.5,
            min_calibration_samples=2,
            provisional_min_samples=5,
            provisional_max_frames=10,
        )
        with self.assertRaisesRegex(ValueError, "base normal context"):
            ContextMemory(2).create_context(
                np.zeros((2, 2)),
                context_name="normal",
                lifecycle="provisional",
            )
        normal = memory.create_context(
            np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]),
            context_name="normal",
        )
        novel = memory.create_context(
            np.asarray([[2.0, 2.0], [2.1, 2.0], [2.0, 2.1], [2.1, 2.1]]),
            context_name="context_0001",
        )
        self.assertTrue(normal.is_stable)
        self.assertTrue(novel.is_provisional)
        self.assertEqual(novel.provisional_samples, 4)
        density = memory.density_thresholds()[0]
        self.assertAlmostEqual(memory.routing_thresholds(0.85)[0], 0.85 * density)
        self.assertAlmostEqual(memory.near_thresholds(0.60)[0], 0.60 * density)
        self.assertAlmostEqual(memory.update_thresholds(1.0)[0], density)

        old_samples = normal.num_samples
        updated = memory.update_stable_if_confident(
            normal.context_id,
            [20.0, 20.0],
            pre_update_score=0.0,
        )
        self.assertFalse(updated)
        self.assertEqual(normal.num_samples, old_samples)

        promoted = memory.update_provisional(
            novel.context_id,
            [2.05, 2.05],
            pre_update_score=0.8,
        )
        self.assertTrue(promoted)
        self.assertTrue(novel.is_stable)
        self.assertEqual(novel.provisional_samples, 5)

        restored = ContextMemory.from_state_dict(memory.state_dict())
        self.assertTrue(restored.get(0).is_stable)
        self.assertTrue(restored.get(1).is_stable)
        self.assertEqual(restored.provisional_min_samples, 5)

    def test_provisional_timeout_retires_instead_of_promoting(self) -> None:
        memory = ContextMemory(
            2, n_rff_features=64, provisional_min_samples=5,
            provisional_max_frames=2,
        )
        memory.create_context(np.zeros((8, 2)), context_name="normal")
        provisional = memory.create_context(
            np.ones((2, 2)), context_name="context_0001"
        )
        memory.mark_provisional_mismatch(provisional.context_id)
        self.assertFalse(memory.should_retire(provisional.context_id))
        memory.mark_provisional_mismatch(provisional.context_id)
        self.assertTrue(memory.should_retire(provisional.context_id))
        self.assertFalse(memory.should_promote(provisional.context_id))
        memory.retire_context(provisional.context_id, fallback_context_id=0)
        self.assertTrue(provisional.is_retired)
        self.assertEqual(set(memory.score_all([0.0, 0.0])), {0})
        self.assertEqual(set(memory.context_lifecycles()), {0})
        router = ContextRouter(2)
        router.set_active_context(1)
        router.retire_context(1, fallback_context_id=0)
        self.assertEqual(router.active_context_id, 0)
        restored = ContextMemory.from_state_dict(memory.state_dict())
        self.assertTrue(restored.get(1).is_retired)
        self.assertEqual(restored.active_context_id, 0)

    def test_context_memory_v2_state_migrates_to_stable(self) -> None:
        memory = ContextMemory(2, n_rff_features=64)
        memory.create_context(np.zeros((8, 2)), context_name="normal")
        memory.create_context(np.ones((8, 2)), context_name="context_0001")
        state = copy.deepcopy(memory.state_dict())
        state["version"] = 2
        state["config"].pop("provisional_min_samples")
        state["config"].pop("provisional_max_frames")
        for item in state["contexts"]:
            item["threshold"] = item.pop("density_threshold")
            item.pop("lifecycle")
            item.pop("provisional_samples")
            item.pop("provisional_frames")
            item.pop("consecutive_mismatches")
        restored = ContextMemory.from_state_dict(state)
        self.assertTrue(all(entry.is_stable for entry in restored.contexts.values()))

    def test_memory_rejects_incompatible_kde_state(self) -> None:
        memory = ContextMemory(2, n_rff_features=64, bandwidth=0.5)
        memory.create_context(np.zeros((8, 2)), context_name="normal")
        memory.create_context(np.ones((8, 2)), context_name="context_0001")
        invalid = copy.deepcopy(memory.state_dict())
        invalid["contexts"][1]["kde_x"]["bandwidth"] = 10.0
        with self.assertRaisesRegex(ValueError, "KDE configuration"):
            ContextMemory.from_state_dict(invalid)

    def test_context_creation_transaction_rolls_back(self) -> None:
        memory = ContextMemory(2, n_rff_features=64)
        memory.create_context(np.zeros((8, 2)), context_name="normal")
        router = ContextRouter(
            2,
            min_candidate_samples=1,
            novelty_decay=1.0,
            novelty_drift=0.0,
            novelty_evidence_threshold=0.5,
            candidate_evidence_threshold=0.5,
            candidate_max_size=2,
        )
        router.set_active_context(0)
        decision = router.step(
            [3.0, 3.0],
            {0: 0.0},
            {0: 0.1},
        )
        self.assertEqual(decision.status, "create")

        model = ToyPSP()
        _, parameters = activate_context_residuals(model, "normal")
        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer.param_groups[0]["context_name"] = "normal"
        with mock.patch.object(
            memory,
            "create_context",
            side_effect=RuntimeError("injected failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                create_context_transaction(
                    network=model,
                    optimizer=optimizer,
                    memory=memory,
                    router=router,
                    initial_features=decision.candidate_features,
                    created_step=1,
                    trainable_scope="fuser_head",
                    learning_rate=1e-4,
                )
        self.assertEqual(get_context_manifest(model), ("normal",))
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory.active_context_id, 0)
        self.assertEqual(router.active_context_id, 0)

    def test_memory_router_and_dynamic_model_growth(self) -> None:
        rng = np.random.default_rng(11)
        normal = rng.normal([0.0, 0.0], 0.1, size=(40, 2))
        novel = rng.normal([3.0, 3.0], 0.1, size=(3, 2))
        memory = ContextMemory(
            2,
            n_rff_features=2048,
            bandwidth=0.5,
            random_state=13,
            min_calibration_samples=4,
        )
        memory.create_context(normal, context_name="normal")
        router = ContextRouter(
            2,
            min_candidate_samples=3,
            novelty_decay=1.0,
            novelty_drift=0.0,
            novelty_evidence_threshold=1.5,
            candidate_evidence_threshold=1.5,
            candidate_similarity_threshold=0.4,
            candidate_n_rff_features=2048,
            candidate_bandwidth=0.5,
            random_state=17,
        )
        router.set_active_context(0)

        decision = None
        for value in novel:
            decision = router.step(
                value,
                memory.score_all(value),
                memory.thresholds(),
            )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, "create")

        rollback_model = ToyPSP()
        with self.assertRaises(ValueError):
            add_dynamic_context(
                rollback_model,
                "context_0001",
                trainable_scope="invalid",
            )
        self.assertEqual(get_context_manifest(rollback_model), ("normal",))

        model = ToyPSP()
        _, normal_parameters = activate_context_residuals(model, "normal")
        optimizer = torch.optim.AdamW(normal_parameters, lr=1e-4)
        dynamic = add_dynamic_context(model, "context_0001")
        self.assertEqual(add_context_optimizer_group(optimizer, dynamic), 2)
        entry = memory.create_context(
            decision.candidate_features,
            context_name="context_0001",
            model_context_name="context_0001",
        )
        router.confirm_created_context(entry.context_id)
        self.assertEqual(get_context_manifest(model), ("normal", "context_0001"))
        self.assertEqual(router.active_context_id, 1)

    def test_dynamic_checkpoint_rebuild(self) -> None:
        rng = np.random.default_rng(19)
        projector = RandomFeatureProjector(["feature"], projection_dim=2)
        projector.update_normalization(rng.normal(size=(8, 4)))
        projector.finalize()
        memory = ContextMemory(2, n_rff_features=64, min_calibration_samples=2)
        memory.create_context(
            rng.normal(0.0, 0.1, size=(8, 2)),
            context_name="normal",
        )
        memory.create_context(
            rng.normal(2.0, 0.1, size=(8, 2)),
            context_name="context_0001",
        )
        router = ContextRouter(2)
        router.set_active_context(1)
        model = ToyPSP()
        _, normal_parameters = activate_context_residuals(model, "normal")
        optimizer = torch.optim.AdamW(normal_parameters, lr=1e-4)
        optimizer.param_groups[0]["context_name"] = "normal"
        dynamic = add_dynamic_context(model, "context_0001")
        add_context_optimizer_group(optimizer, dynamic, learning_rate=1e-4)
        _, parameters = activate_context_residuals(model, "context_0001")
        memory.get(1).model_update_enabled = False

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "dynamic.checkpoint"
            router._awaiting_context_creation = True
            with self.assertRaisesRegex(RuntimeError, "uncommitted"):
                save_context_checkpoint(
                    path,
                    network=model,
                    context_memory=memory,
                    router=router,
                    projector=projector,
                    optimizer=optimizer,
                    step_idx=12,
                    update_idx=7,
                )
            router._awaiting_context_creation = False
            save_context_checkpoint(
                path,
                network=model,
                context_memory=memory,
                router=router,
                projector=projector,
                optimizer=optimizer,
                step_idx=12,
                update_idx=7,
            )
            payload = load_context_checkpoint(path)
            self.assertEqual(payload["checkpoint_version"], CHECKPOINT_VERSION)
            self.assertEqual(
                payload["component_state_versions"],
                {
                    "context_memory": ContextMemory.STATE_VERSION,
                    "router": ContextRouter.STATE_VERSION,
                    "projector": RandomFeatureProjector.STATE_VERSION,
                },
            )

            legacy_path = Path(temporary_directory) / "legacy_v2.checkpoint"
            legacy_payload = copy.deepcopy(payload)
            legacy_payload["checkpoint_version"] = 2
            legacy_payload.pop("component_state_versions")
            torch.save(legacy_payload, legacy_path)
            self.assertEqual(
                load_context_checkpoint(legacy_path)["checkpoint_version"],
                2,
            )

            invalid_path = Path(temporary_directory) / "invalid_version.checkpoint"
            invalid_payload = copy.deepcopy(payload)
            invalid_payload["component_state_versions"]["router"] = 99
            torch.save(invalid_payload, invalid_path)
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                load_context_checkpoint(invalid_path)
        incompatible_optimizer = torch.optim.AdamW(parameters, lr=2e-4)
        incompatible_optimizer.param_groups[0]["context_name"] = "context_0001"
        with self.assertRaisesRegex(ValueError, "Optimizer configuration"):
            validate_optimizer_resume(incompatible_optimizer, payload)

        restored_model = ToyPSP()
        restored_memory, restored_router, _ = restore_model_and_context_state(
            restored_model,
            payload,
        )
        self.assertEqual(get_context_manifest(restored_model), ("normal", "context_0001"))
        self.assertEqual(len(restored_memory), 2)
        self.assertEqual(restored_router.active_context_id, 1)
        self.assertFalse(restored_memory.get(1).model_update_enabled)


if __name__ == "__main__":
    unittest.main()
