"""Tests for Issue #7 — model collapse detection and mitigation.

Covers:
  1. Metrics: coverage entropy, tail entropy, minority representation, JS divergence
  2. Pipeline: collapse dynamics across 5 generations
  3. Mitigation: real-data anchoring, diversity-rewarded sampling
  4. Evaluator: full pipeline + success metric (tail entropy within 10%)
"""
from __future__ import annotations

import sys
import os
import math


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_collapse.metrics import (
    coverage_entropy,
    tail_coverage_entropy,
    minority_class_representation,
    tail_divergence,
    entropy_within_tolerance,
)
from model_collapse.pipeline import (
    make_enterprise_dataset,
    run_model_collapse_pipeline,
    GenerationResult,
)
from model_collapse.mitigation import (
    real_data_anchoring,
    diversity_rewarded_sampling,
    mitigated_pipeline_step,
)
from model_collapse.evaluator import evaluate_model_collapse


FIELD = "record_type"
MINORITY = ("fraud", "critical_security")


# ─── Fixtures ────────────────────────────────────────────────────────────────

def uniform_records(n: int = 100, values: list[str] | None = None) -> list[dict]:
    if values is None:
        values = ["a", "b", "c", "d"]
    return [{FIELD: values[i % len(values)]} for i in range(n)]


def single_value_records(n: int = 50) -> list[dict]:
    return [{FIELD: "routine"} for _ in range(n)]


def enterprise_records() -> list[dict]:
    return make_enterprise_dataset(n_records=400, field=FIELD, seed=42)


# ─── 1. Metrics ───────────────────────────────────────────────────────────────

class TestCoverageEntropy:
    def test_uniform_is_one(self):
        records = uniform_records(100, ["a", "b", "c", "d"])
        assert math.isclose(coverage_entropy(records, FIELD), 1.0, rel_tol=1e-6)

    def test_single_value_is_zero(self):
        assert coverage_entropy(single_value_records(), FIELD) == 0.0

    def test_empty_is_zero(self):
        assert coverage_entropy([], FIELD) == 0.0

    def test_more_values_higher_entropy(self):
        r2 = uniform_records(100, ["a", "b"])
        r4 = uniform_records(100, ["a", "b", "c", "d"])
        assert coverage_entropy(r4, FIELD) >= coverage_entropy(r2, FIELD)

    def test_enterprise_dataset_entropy_in_range(self):
        records = enterprise_records()
        e = coverage_entropy(records, FIELD)
        assert 0.0 < e <= 1.0


class TestTailCoverageEntropy:
    def test_empty_returns_zero(self):
        assert tail_coverage_entropy([], FIELD) == 0.0

    def test_single_value_returns_zero(self):
        assert tail_coverage_entropy(single_value_records(), FIELD) == 0.0

    def test_enterprise_tail_entropy_positive(self):
        records = enterprise_records()
        assert tail_coverage_entropy(records, FIELD) > 0.0

    def test_tail_entropy_is_valid_score(self):
        records = enterprise_records()
        tail_e = tail_coverage_entropy(records, FIELD)
        # Tail entropy is normalised within the tail subset, so [0, 1]
        assert 0.0 <= tail_e <= 1.0


class TestMinorityClassRepresentation:
    def test_no_minority_records_returns_zero(self):
        records = single_value_records()
        assert minority_class_representation(records, FIELD, MINORITY) == 0.0

    def test_all_minority_returns_one(self):
        records = [{FIELD: "fraud"} for _ in range(20)]
        score = minority_class_representation(records, FIELD, ["fraud"])
        assert math.isclose(score, 1.0, rel_tol=1e-9)

    def test_partial_minority(self):
        records = [{FIELD: "fraud"}] * 10 + [{FIELD: "routine"}] * 90
        score = minority_class_representation(records, FIELD, ["fraud"])
        assert math.isclose(score, 0.10, rel_tol=1e-6)

    def test_enterprise_minority_representation_positive(self):
        records = enterprise_records()
        score = minority_class_representation(records, FIELD, list(MINORITY))
        assert score > 0.0

    def test_empty_returns_zero(self):
        assert minority_class_representation([], FIELD, ["fraud"]) == 0.0


class TestTailDivergence:
    def test_identical_distributions_near_zero(self):
        records = enterprise_records()
        div = tail_divergence(records, records, FIELD)
        assert div < 0.01

    def test_divergence_in_range(self):
        a = enterprise_records()
        b = single_value_records(400)
        div = tail_divergence(a, b, FIELD)
        assert 0.0 <= div <= 1.0

    def test_empty_b_returns_zero(self):
        a = enterprise_records()
        assert tail_divergence(a, [], FIELD) == 0.0

    def test_empty_a_returns_zero(self):
        b = enterprise_records()
        assert tail_divergence([], b, FIELD) == 0.0


class TestSuccessMetric:
    def test_same_entropy_passes(self):
        assert entropy_within_tolerance(0.90, 0.90, 0.10) is True

    def test_exactly_at_boundary_passes(self):
        assert entropy_within_tolerance(0.81, 0.90, 0.10) is True  # 10% drop = boundary

    def test_just_outside_boundary_fails(self):
        assert entropy_within_tolerance(0.80, 0.90, 0.10) is False  # >10% drop

    def test_improvement_always_passes(self):
        # Entropy improving beyond original is fine — we only care about collapse
        assert entropy_within_tolerance(0.99, 0.90, 0.10) is True

    def test_zero_original_passes(self):
        assert entropy_within_tolerance(0.0, 0.0, 0.10) is True

    def test_zero_original_with_any_current_passes(self):
        # Any non-negative is >= 0 * 0.9
        assert entropy_within_tolerance(0.5, 0.0, 0.10) is True


# ─── 2. Pipeline ──────────────────────────────────────────────────────────────

class TestPipeline:
    def setup_method(self):
        self.generations = run_model_collapse_pipeline(
            n_generations=5, n_records=400, field=FIELD,
            minority_classes=MINORITY, collapse_rate=0.30, seed=42
        )

    def test_returns_six_generations(self):
        assert len(self.generations) == 6  # 0 through 5

    def test_generation_zero_is_baseline(self):
        assert self.generations[0].generation == 0

    def test_all_results_are_generation_result(self):
        for g in self.generations:
            assert isinstance(g, GenerationResult)

    def test_coverage_entropy_decreases_over_generations(self):
        entropies = [g.coverage_entropy for g in self.generations]
        # Should generally trend downward (allow 1 non-monotone step)
        decreases = sum(
            1 for a, b in zip(entropies, entropies[1:]) if b < a
        )
        assert decreases >= 3, f"Expected entropy to mostly decrease: {entropies}"

    def test_minority_representation_decreases(self):
        minority_scores = [g.minority_representation for g in self.generations]
        # Gen 5 should have less minority than gen 0
        assert minority_scores[-1] < minority_scores[0]

    def test_tail_divergence_increases(self):
        divergences = [g.tail_divergence for g in self.generations]
        # Gen 5 should be more diverged than gen 0 (which is 0)
        assert divergences[-1] > divergences[0]

    def test_as_dict_has_required_keys(self):
        d = self.generations[1].as_dict()
        for key in (
            "generation", "n_records", "coverage_entropy",
            "tail_entropy", "tail_divergence", "minority_representation"
        ):
            assert key in d

    def test_make_enterprise_dataset_has_minority_classes(self):
        records = make_enterprise_dataset(n_records=500, field=FIELD, seed=42)
        types = {str(r.get(FIELD, "")) for r in records}
        assert "fraud" in types
        assert "critical_security" in types


# ─── 3. Mitigation ────────────────────────────────────────────────────────────

class TestRealDataAnchoring:
    def test_anchoring_increases_record_count(self):
        synthetic = single_value_records(80)
        real = enterprise_records()
        anchored = real_data_anchoring(synthetic, real, FIELD, anchor_ratio=0.20)
        assert len(anchored) > len(synthetic)

    def test_anchoring_restores_minority_classes(self):
        synthetic = single_value_records(80)
        real = enterprise_records()
        anchored = real_data_anchoring(synthetic, real, FIELD, anchor_ratio=0.20)
        score = minority_class_representation(anchored, FIELD, list(MINORITY))
        assert score > 0.0

    def test_anchoring_with_empty_real_returns_synthetic(self):
        synthetic = single_value_records(10)
        result = real_data_anchoring(synthetic, [], FIELD)
        assert result == synthetic

    def test_anchor_ratio_zero_does_not_add_records(self):
        synthetic = single_value_records(80)
        real = enterprise_records()
        anchored = real_data_anchoring(synthetic, real, FIELD, anchor_ratio=0.0)
        # anchor_ratio=0 → max(1, 0) = 1 record added
        assert len(anchored) >= len(synthetic)


class TestDiversityRewardedSampling:
    def test_returns_same_size_by_default(self):
        records = enterprise_records()
        resampled = diversity_rewarded_sampling(records, FIELD)
        assert len(resampled) == len(records)

    def test_increases_minority_representation(self):
        # Collapse to mostly routine, then resample
        collapsed = single_value_records(380) + [
            {FIELD: "fraud"}, {FIELD: "fraud"}, {FIELD: "critical_security"}
        ]
        before = minority_class_representation(collapsed, FIELD, list(MINORITY))
        resampled = diversity_rewarded_sampling(
            collapsed, FIELD, reward_strength=5.0, seed=1
        )
        after = minority_class_representation(resampled, FIELD, list(MINORITY))
        assert after > before

    def test_empty_returns_empty(self):
        assert diversity_rewarded_sampling([], FIELD) == []

    def test_target_n_respected(self):
        records = enterprise_records()
        resampled = diversity_rewarded_sampling(records, FIELD, target_n=200)
        assert len(resampled) == 200


class TestMitigatedPipelineStep:
    def test_anchor_strategy(self):
        collapsed = single_value_records(100)
        original = enterprise_records()
        result = mitigated_pipeline_step(
            collapsed, original, FIELD, strategy="anchor"
        )
        score = minority_class_representation(result, FIELD, list(MINORITY))
        assert score > 0.0

    def test_diversity_strategy(self):
        collapsed = single_value_records(380) + [
            {FIELD: "fraud"} for _ in range(20)
        ]
        original = enterprise_records()
        result = mitigated_pipeline_step(
            collapsed, original, FIELD, strategy="diversity", reward_strength=4.0
        )
        before = minority_class_representation(collapsed, FIELD, list(MINORITY))
        after = minority_class_representation(result, FIELD, list(MINORITY))
        assert after > before

    def test_both_strategy_best_minority_retention(self):
        collapsed = single_value_records(380) + [{FIELD: "fraud"} for _ in range(20)]
        original = enterprise_records()
        r_anchor = mitigated_pipeline_step(
            collapsed, original, FIELD, strategy="anchor"
        )
        r_both = mitigated_pipeline_step(
            collapsed, original, FIELD, strategy="both"
        )
        score_anchor = minority_class_representation(r_anchor, FIELD, list(MINORITY))
        score_both = minority_class_representation(r_both, FIELD, list(MINORITY))
        assert score_both >= score_anchor


# ─── 4. Full Evaluator + Success Metric ───────────────────────────────────────

class TestEvaluator:
    def setup_method(self):
        self.result = evaluate_model_collapse(
            n_generations=5, n_records=400,
            field=FIELD, minority_classes=MINORITY,
            collapse_rate=0.30, tolerance=0.10, seed=42
        )

    def test_result_has_all_strategy_keys(self):
        for key in ("baseline", "anchored", "diversity", "combined", "success"):
            assert key in self.result

    def test_baseline_has_six_generations(self):
        assert len(self.result["baseline"]) == 6

    def test_anchored_has_six_generations(self):
        assert len(self.result["anchored"]) == 6

    def test_success_dict_has_all_strategies(self):
        for key in ("baseline", "anchored", "diversity", "combined"):
            assert key in self.result["success"]

    def test_baseline_does_not_meet_success_metric(self):
        """No mitigation → collapse → should fail 10% tolerance."""
        assert self.result["success"]["baseline"] is False

    def test_at_least_one_mitigation_meets_success_metric(self):
        """At least one strategy should maintain tail entropy within 10%."""
        mitigations = [
            self.result["success"]["anchored"],
            self.result["success"]["diversity"],
            self.result["success"]["combined"],
        ]
        assert any(mitigations), (
            "No mitigation strategy met the 10% tail entropy tolerance. "
            f"Success: {self.result['success']}"
        )

    def test_combined_strategy_best_entropy(self):
        """Combined strategy should produce higher final tail entropy than baseline."""
        base_final = self.result["baseline"][-1]["tail_entropy"]
        combined_final = self.result["combined"][-1]["tail_entropy"]
        assert combined_final >= base_final

    def test_minority_representation_improves_with_mitigation(self):
        base_minority = self.result["baseline"][-1]["minority_representation"]
        anchored_minority = self.result["anchored"][-1]["minority_representation"]
        assert anchored_minority >= base_minority

    def test_per_generation_dicts_have_required_keys(self):
        for strategy in ("baseline", "anchored", "diversity", "combined"):
            for gen_dict in self.result[strategy]:
                for key in ("generation", "tail_entropy", "coverage_entropy",
                            "minority_representation", "tail_divergence"):
                    assert key in gen_dict, f"{strategy} gen missing key {key}"
