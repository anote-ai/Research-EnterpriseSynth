"""Tests for Issue #16 — statistical rigor additions.

Covers:
  1. Bootstrap confidence intervals on TSTR metrics
  2. DP training variance across seeds
  3. Significance testing with Bonferroni correction
  4. Privacy accountant epsilon verification (unit test that fails on drift)
  5. Dataset-type stratification
  6. Updated evaluator with CI and multi-seed support
"""
from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy_benchmark.stats import (
    bootstrap_ci,
    bootstrap_tstr_ci,
    dp_training_variance,
    epsilon_variance_profile,
    significance_test,
    stratified_utility_privacy,
    verify_epsilon_accounting,
)
from privacy_benchmark.evaluator import (
    evaluate_configuration,
    evaluate_with_ci,
    evaluate_multi_seed,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

TSTR_SCORES_HIGH = [0.90, 0.91, 0.89, 0.92, 0.88, 0.90, 0.91, 0.89, 0.92, 0.90]
TSTR_SCORES_LOW  = [0.70, 0.71, 0.69, 0.72, 0.68, 0.70, 0.71, 0.69, 0.72, 0.70]

SEED_RESULTS_TIGHT_PRIVACY = [
    {"utility_score": 0.72, "privacy_score": 0.94, "fidelity_score": 0.80},
    {"utility_score": 0.68, "privacy_score": 0.96, "fidelity_score": 0.77},
    {"utility_score": 0.74, "privacy_score": 0.93, "fidelity_score": 0.81},
    {"utility_score": 0.70, "privacy_score": 0.95, "fidelity_score": 0.79},
    {"utility_score": 0.69, "privacy_score": 0.94, "fidelity_score": 0.78},
]

SEED_RESULTS_LOOSE_PRIVACY = [
    {"utility_score": 0.93, "privacy_score": 0.72, "fidelity_score": 0.94},
    {"utility_score": 0.94, "privacy_score": 0.71, "fidelity_score": 0.95},
    {"utility_score": 0.92, "privacy_score": 0.73, "fidelity_score": 0.93},
    {"utility_score": 0.93, "privacy_score": 0.72, "fidelity_score": 0.94},
    {"utility_score": 0.94, "privacy_score": 0.70, "fidelity_score": 0.95},
]


# ─── 1. Bootstrap CIs ─────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_returns_three_values(self):
        mean, lo, hi = bootstrap_ci(TSTR_SCORES_HIGH)
        assert isinstance(mean, float)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_le_mean_le_upper(self):
        mean, lo, hi = bootstrap_ci(TSTR_SCORES_HIGH)
        assert lo <= mean <= hi

    def test_ci_width_positive(self):
        _, lo, hi = bootstrap_ci(TSTR_SCORES_HIGH)
        assert hi > lo

    def test_tighter_with_less_variance(self):
        """High-variance data should produce wider CIs."""
        varied = [0.5, 0.9, 0.1, 0.95, 0.05, 0.8, 0.2, 0.85, 0.15, 0.7]
        _, lo_v, hi_v = bootstrap_ci(varied)
        _, lo_h, hi_h = bootstrap_ci(TSTR_SCORES_HIGH)
        assert (hi_v - lo_v) > (hi_h - lo_h)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_tstr_ci_dict_keys(self):
        result = bootstrap_tstr_ci(TSTR_SCORES_HIGH)
        for key in ("mean", "ci_lower", "ci_upper", "ci_half_width"):
            assert key in result

    def test_tstr_ci_half_width_matches(self):
        result = bootstrap_tstr_ci(TSTR_SCORES_HIGH)
        expected_hw = (result["ci_upper"] - result["ci_lower"]) / 2.0
        assert math.isclose(result["ci_half_width"], expected_hw, rel_tol=1e-9)

    def test_high_scores_have_higher_mean(self):
        hi_result = bootstrap_tstr_ci(TSTR_SCORES_HIGH)
        lo_result = bootstrap_tstr_ci(TSTR_SCORES_LOW)
        assert hi_result["mean"] > lo_result["mean"]


# ─── 2. DP Training Variance ──────────────────────────────────────────────────

class TestDPTrainingVariance:
    def test_returns_summary_per_metric(self):
        summary = dp_training_variance(SEED_RESULTS_TIGHT_PRIVACY)
        for metric in ("utility_score", "privacy_score", "fidelity_score"):
            assert metric in summary
            for stat in ("mean", "std", "min", "max", "n_seeds"):
                assert stat in summary[metric]

    def test_n_seeds_correct(self):
        summary = dp_training_variance(SEED_RESULTS_TIGHT_PRIVACY)
        assert summary["utility_score"]["n_seeds"] == 5.0

    def test_tight_privacy_has_higher_variance(self):
        """ε=0.1 (tight) should show higher utility variance than ε=10 (loose)."""
        var_tight = dp_training_variance(SEED_RESULTS_TIGHT_PRIVACY)
        var_loose = dp_training_variance(SEED_RESULTS_LOOSE_PRIVACY)
        assert var_tight["utility_score"]["std"] > var_loose["utility_score"]["std"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            dp_training_variance([])

    def test_single_seed_zero_std(self):
        result = dp_training_variance([{"utility_score": 0.85}])
        assert result["utility_score"]["std"] == 0.0

    def test_epsilon_variance_profile_sorted(self):
        ep_map = {
            0.1: SEED_RESULTS_TIGHT_PRIVACY,
            10.0: SEED_RESULTS_LOOSE_PRIVACY,
        }
        profile = epsilon_variance_profile(ep_map, metric="utility_score")
        epsilons = [p["epsilon"] for p in profile]
        assert epsilons == sorted(epsilons)

    def test_tight_epsilon_higher_std_in_profile(self):
        ep_map = {
            0.1: SEED_RESULTS_TIGHT_PRIVACY,
            10.0: SEED_RESULTS_LOOSE_PRIVACY,
        }
        profile = epsilon_variance_profile(ep_map, metric="utility_score")
        std_by_eps = {p["epsilon"]: p["std"] for p in profile}
        assert std_by_eps[0.1] > std_by_eps[10.0]


# ─── 3. Significance Testing ──────────────────────────────────────────────────

class TestSignificanceTesting:
    def test_result_has_required_keys(self):
        result = significance_test(TSTR_SCORES_HIGH, TSTR_SCORES_LOW)
        for key in ("w_stat", "p_value", "p_adjusted_threshold", "significant", "test_label"):
            assert key in result

    def test_clearly_different_scores_significant(self):
        """A 20-point difference should be statistically significant."""
        result = significance_test(TSTR_SCORES_HIGH, TSTR_SCORES_LOW, n_comparisons=1)
        assert result["significant"] is True
        assert result["test_label"] == "†"

    def test_identical_scores_not_significant(self):
        same = [0.85] * 10
        result = significance_test(same, same, n_comparisons=1)
        assert result["significant"] is False

    def test_bonferroni_raises_threshold(self):
        """With many comparisons the adjusted threshold should be smaller."""
        r1 = significance_test(TSTR_SCORES_HIGH, TSTR_SCORES_LOW, n_comparisons=1)
        r6 = significance_test(TSTR_SCORES_HIGH, TSTR_SCORES_LOW, n_comparisons=6)
        assert float(r6["p_adjusted_threshold"]) < float(r1["p_adjusted_threshold"])

    def test_p_value_in_range(self):
        result = significance_test(TSTR_SCORES_HIGH, TSTR_SCORES_LOW)
        assert 0.0 <= float(result["p_value"]) <= 1.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            significance_test([0.9, 0.8], [0.7])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            significance_test([0.9], [0.7])


# ─── 4. Privacy Accountant Epsilon Verification ───────────────────────────────

class TestEpsilonVerification:
    def test_exact_match_passes(self):
        result = verify_epsilon_accounting(1.0, 1.0)
        assert result["passed"] is True

    def test_within_tolerance_passes(self):
        result = verify_epsilon_accounting(1.0, 1.005, tolerance=0.01)
        assert result["passed"] is True

    def test_outside_tolerance_fails(self):
        """This is the unit test that should fail if the privacy accountant drifts."""
        result = verify_epsilon_accounting(1.0, 1.05, tolerance=0.01)
        assert result["passed"] is False
        assert "FAIL" in result["message"]

    def test_large_drift_detected(self):
        result = verify_epsilon_accounting(0.1, 0.5, tolerance=0.01)
        assert result["passed"] is False

    def test_result_contains_delta(self):
        result = verify_epsilon_accounting(1.0, 1.03, tolerance=0.05)
        assert math.isclose(float(result["delta"]), 0.03, rel_tol=1e-6)

    def test_pytest_assertion_on_drift(self):
        """Demonstrates how to embed this as a pytest assertion guard.
        If DP implementation drifts, this test will fail CI."""
        reported_eps = 1.0
        # In a real pipeline, computed_eps comes from the privacy accountant lib
        computed_eps = 1.0  # mock: accountant agrees with reported
        result = verify_epsilon_accounting(reported_eps, computed_eps, tolerance=0.01)
        assert result["passed"], result["message"]


# ─── 5. Dataset-Type Stratification ──────────────────────────────────────────

CONFIGS_BY_TYPE = {
    "tabular": [
        {"privacy_score": 0.80, "utility_score": 0.85, "fidelity_score": 0.90},
        {"privacy_score": 0.82, "utility_score": 0.83, "fidelity_score": 0.88},
    ],
    "time_series": [
        {"privacy_score": 0.75, "utility_score": 0.78, "fidelity_score": 0.82},
        {"privacy_score": 0.77, "utility_score": 0.76, "fidelity_score": 0.80},
    ],
    "text": [
        {"privacy_score": 0.70, "utility_score": 0.88, "fidelity_score": 0.85},
        {"privacy_score": 0.72, "utility_score": 0.86, "fidelity_score": 0.83},
    ],
}


class TestStratification:
    def test_all_types_present(self):
        report = stratified_utility_privacy(CONFIGS_BY_TYPE)
        assert set(report.keys()) == {"tabular", "time_series", "text"}

    def test_metrics_present_for_each_type(self):
        report = stratified_utility_privacy(CONFIGS_BY_TYPE)
        for dtype in report:
            for metric in ("privacy_score", "utility_score", "fidelity_score"):
                assert metric in report[dtype], f"{metric} missing for {dtype}"

    def test_mean_values_in_range(self):
        report = stratified_utility_privacy(CONFIGS_BY_TYPE)
        for dtype in report:
            for metric, stats in report[dtype].items():
                assert 0.0 <= stats["mean"] <= 1.0, f"mean out of range for {dtype}/{metric}"

    def test_std_nonnegative(self):
        report = stratified_utility_privacy(CONFIGS_BY_TYPE)
        for dtype in report:
            for metric, stats in report[dtype].items():
                assert stats["std"] >= 0.0

    def test_empty_type_skipped(self):
        configs = {"tabular": CONFIGS_BY_TYPE["tabular"], "empty_type": []}
        report = stratified_utility_privacy(configs)
        assert "empty_type" not in report
        assert "tabular" in report


# ─── 6. Updated Evaluator ─────────────────────────────────────────────────────

class TestEvaluatorWithStats:
    def test_evaluate_with_ci_keys(self):
        result = evaluate_with_ci(1.0, 0.60, TSTR_SCORES_HIGH, 0.91)
        for key in ("tstr_ci_lower", "tstr_ci_upper", "tstr_ci_half_width", "tstr_n_samples"):
            assert key in result, f"missing key: {key}"

    def test_ci_half_width_positive(self):
        result = evaluate_with_ci(1.0, 0.60, TSTR_SCORES_HIGH, 0.91)
        assert result["tstr_ci_half_width"] > 0

    def test_evaluate_multi_seed_keys(self):
        seed_configs = [
            {"auc": 0.55, "tstr_score": 0.88, "fidelity": 0.91},
            {"auc": 0.57, "tstr_score": 0.86, "fidelity": 0.90},
            {"auc": 0.54, "tstr_score": 0.89, "fidelity": 0.92},
            {"auc": 0.56, "tstr_score": 0.87, "fidelity": 0.91},
            {"auc": 0.55, "tstr_score": 0.88, "fidelity": 0.91},
        ]
        result = evaluate_multi_seed(1.0, seed_configs)
        assert "variance" in result
        assert result["n_seeds"] == 5

    def test_epsilon_check_included_when_computed_provided(self):
        seed_configs = [{"auc": 0.55, "tstr_score": 0.88, "fidelity": 0.91}]
        result = evaluate_multi_seed(1.0, seed_configs, computed_epsilon=1.0)
        assert "epsilon_check" in result
        assert result["epsilon_check"]["passed"] is True

    def test_epsilon_drift_detected_in_evaluator(self):
        """CI guard: evaluator surfaces epsilon drift from privacy accountant."""
        seed_configs = [{"auc": 0.55, "tstr_score": 0.88, "fidelity": 0.91}]
        result = evaluate_multi_seed(1.0, seed_configs, computed_epsilon=1.5)
        assert result["epsilon_check"]["passed"] is False
