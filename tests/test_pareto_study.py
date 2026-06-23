"""Tests for issue #35: 5-domain utility-privacy Pareto study.

Covers:
  - DomainSpec: utility/privacy/fidelity curves and ε recommendation
  - Synthesis: tabular, time-series, relational per-column fidelity and FK handling
  - Pareto: is_pareto_efficient, compare_domains
"""
from __future__ import annotations

import math
import pytest

from privacy_benchmark.domains import (
    DOMAINS,
    DOMAIN_NAMES,
    DomainSpec,
    get_domain,
    domains_by_schema_type,
    recommend_epsilon,
)
from privacy_benchmark.synthesis import (
    SynthesisResult,
    synthesise,
    synthesise_all_domains,
)
from privacy_benchmark.pareto import (
    generate_pareto_frontier,
    is_pareto_efficient,
    pareto_efficient_configs,
    compare_domains,
)
from privacy_benchmark.config import EPSILON_VALUES


# ── DomainSpec ─────────────────────────────────────────────────────────────────

class TestDomainSpec:
    def test_all_five_domains_defined(self):
        assert len(DOMAINS) == 5
        expected = {"tabular_hr", "healthcare_ehr", "financial_transactions", "iot_sensor", "ecommerce_relational"}
        assert set(DOMAIN_NAMES) == expected

    def test_get_domain_valid(self):
        spec = get_domain("tabular_hr")
        assert isinstance(spec, DomainSpec)

    def test_get_domain_invalid_raises(self):
        with pytest.raises(KeyError):
            get_domain("nonexistent_domain")

    def test_schema_types_cover_all_three(self):
        types = {spec.schema_type for spec in DOMAINS.values()}
        assert "tabular" in types
        assert "timeseries" in types
        assert "relational" in types

    def test_tabular_domains_no_fk(self):
        for spec in domains_by_schema_type("tabular"):
            assert not spec.has_foreign_keys

    def test_relational_domain_has_fk(self):
        spec = get_domain("ecommerce_relational")
        assert spec.has_foreign_keys
        assert spec.schema_type == "relational"

    def test_timeseries_domains_have_temporal_dependency(self):
        for spec in domains_by_schema_type("timeseries"):
            assert spec.has_temporal_dependency

    def test_utility_increases_with_epsilon(self):
        for name, spec in DOMAINS.items():
            prev = spec.utility_at_epsilon(0.1)
            for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
                curr = spec.utility_at_epsilon(eps)
                assert curr >= prev, f"{name}: utility should increase with ε"
                prev = curr

    def test_privacy_leakage_increases_with_epsilon(self):
        spec = get_domain("tabular_hr")
        prev = spec.privacy_leakage_at_epsilon(0.1)
        for eps in [1.0, 5.0, 10.0]:
            curr = spec.privacy_leakage_at_epsilon(eps)
            assert curr >= prev
            prev = curr

    def test_utility_bounded_by_baseline(self):
        for name, spec in DOMAINS.items():
            for eps in EPSILON_VALUES:
                u = spec.utility_at_epsilon(eps)
                assert u <= spec.baseline_tstr + 1e-9, f"{name} at ε={eps}"
                assert u >= 0.0

    def test_fidelity_bounded(self):
        for name, spec in DOMAINS.items():
            for eps in EPSILON_VALUES:
                f = spec.fidelity_at_epsilon(eps)
                assert 0.0 <= f <= 1.0

    def test_downstream_accuracy_bounded(self):
        for name, spec in DOMAINS.items():
            for eps in EPSILON_VALUES:
                d = spec.downstream_accuracy_at_epsilon(eps)
                assert 0.0 <= d <= 1.0

    def test_iot_harder_than_tabular(self):
        iot = get_domain("iot_sensor")
        hr = get_domain("tabular_hr")
        for eps in [0.1, 0.5, 1.0, 2.0]:
            assert iot.utility_at_epsilon(eps) <= hr.utility_at_epsilon(eps)

    def test_domains_by_schema_type_tabular(self):
        tabular = domains_by_schema_type("tabular")
        assert len(tabular) == 2
        names = {s.name for s in tabular}
        assert "tabular_hr" in names
        assert "healthcare_ehr" in names

    def test_domains_by_schema_type_timeseries(self):
        ts = domains_by_schema_type("timeseries")
        assert len(ts) == 2

    def test_domains_by_schema_type_relational(self):
        rel = domains_by_schema_type("relational")
        assert len(rel) == 1
        assert rel[0].name == "ecommerce_relational"

    def test_recommend_epsilon_low_loss_budget(self):
        spec = get_domain("tabular_hr")
        eps = recommend_epsilon(spec, max_utility_loss=0.05)
        assert eps in EPSILON_VALUES
        # If the constraint is achievable, the returned ε should satisfy it.
        # If no ε in range satisfies it, the function correctly returns the max available.
        target = spec.baseline_tstr * 0.95
        achievable = any(spec.utility_at_epsilon(e) >= target for e in EPSILON_VALUES)
        if achievable:
            assert spec.utility_at_epsilon(eps) >= target - 1e-6
        else:
            assert eps == max(EPSILON_VALUES)

    def test_recommend_epsilon_high_loss_budget(self):
        spec = get_domain("tabular_hr")
        # 50% loss budget → minimum ε should be very small
        eps = recommend_epsilon(spec, max_utility_loss=0.50)
        assert eps == min(EPSILON_VALUES) or spec.utility_at_epsilon(eps) >= spec.baseline_tstr * 0.50 - 1e-6

    def test_relational_has_higher_sensitivity_multiplier_than_tabular(self):
        hr = get_domain("tabular_hr")
        ecom = get_domain("ecommerce_relational")
        assert ecom.dp_sensitivity_multiplier > hr.dp_sensitivity_multiplier


# ── SynthesisResult ───────────────────────────────────────────────────────────

class TestSynthesisResult:
    def test_synthesise_returns_result(self):
        result = synthesise("tabular_hr", epsilon=2.0)
        assert isinstance(result, SynthesisResult)

    def test_synthesise_domain_field(self):
        result = synthesise("healthcare_ehr", epsilon=1.0)
        assert result.domain == "healthcare_ehr"

    def test_synthesise_epsilon_field(self):
        result = synthesise("tabular_hr", epsilon=5.0)
        assert result.epsilon == 5.0

    def test_tstr_bounded(self):
        for name in DOMAIN_NAMES:
            result = synthesise(name, epsilon=1.0)
            assert 0.0 <= result.tstr_score <= 1.0

    def test_fidelity_bounded(self):
        for name in DOMAIN_NAMES:
            result = synthesise(name, epsilon=1.0)
            assert 0.0 <= result.fidelity_score <= 1.0

    def test_privacy_auc_bounded(self):
        for name in DOMAIN_NAMES:
            result = synthesise(name, epsilon=1.0)
            assert 0.50 <= result.privacy_auc <= 1.0

    def test_tstr_increases_with_epsilon(self):
        r1 = synthesise("tabular_hr", epsilon=0.1, seed=0)
        r2 = synthesise("tabular_hr", epsilon=10.0, seed=0)
        assert r2.tstr_score >= r1.tstr_score

    def test_privacy_auc_increases_with_epsilon(self):
        r1 = synthesise("tabular_hr", epsilon=0.1, seed=0)
        r2 = synthesise("tabular_hr", epsilon=10.0, seed=0)
        assert r2.privacy_auc >= r1.privacy_auc

    def test_relational_has_fk_violations_at_low_epsilon(self):
        result = synthesise("ecommerce_relational", epsilon=0.1)
        assert result.fk_violation_rate > 0.0

    def test_relational_fk_violations_decrease_with_epsilon(self):
        r_low = synthesise("ecommerce_relational", epsilon=0.1, seed=0)
        r_high = synthesise("ecommerce_relational", epsilon=10.0, seed=0)
        assert r_high.fk_violation_rate < r_low.fk_violation_rate

    def test_tabular_no_fk_violations(self):
        result = synthesise("tabular_hr", epsilon=0.1)
        assert result.fk_violation_rate == 0.0

    def test_timeseries_has_autocorr_loss(self):
        result = synthesise("financial_transactions", epsilon=0.5)
        assert result.temporal_autocorr_loss > 0.0

    def test_autocorr_loss_decreases_with_epsilon(self):
        r_low = synthesise("iot_sensor", epsilon=0.1, seed=0)
        r_high = synthesise("iot_sensor", epsilon=10.0, seed=0)
        assert r_high.temporal_autocorr_loss <= r_low.temporal_autocorr_loss

    def test_tabular_no_autocorr_loss(self):
        result = synthesise("tabular_hr", epsilon=1.0)
        assert result.temporal_autocorr_loss == 0.0

    def test_column_fidelities_nonempty(self):
        for name in DOMAIN_NAMES:
            result = synthesise(name, epsilon=2.0)
            assert len(result.column_fidelities) > 0

    def test_column_fidelity_wasserstein_decreases_with_epsilon(self):
        r_low = synthesise("tabular_hr", epsilon=0.1, seed=0)
        r_high = synthesise("tabular_hr", epsilon=10.0, seed=0)
        wd_low = sum(c.wasserstein_distance for c in r_low.column_fidelities)
        wd_high = sum(c.wasserstein_distance for c in r_high.column_fidelities)
        assert wd_high < wd_low

    def test_utility_retention_bounded(self):
        result = synthesise("tabular_hr", epsilon=2.0)
        assert 0.0 <= result.utility_retention <= 1.05  # allow tiny fp overshoot

    def test_downstream_accuracy_bounded(self):
        for name in DOMAIN_NAMES:
            result = synthesise(name, epsilon=2.0)
            assert 0.0 <= result.downstream_accuracy <= 1.0

    def test_downstream_retention_high_at_large_epsilon(self):
        result = synthesise("tabular_hr", epsilon=10.0)
        assert result.downstream_retention >= 0.90

    def test_deterministic_with_seed(self):
        r1 = synthesise("healthcare_ehr", epsilon=1.0, seed=7)
        r2 = synthesise("healthcare_ehr", epsilon=1.0, seed=7)
        assert abs(r1.tstr_score - r2.tstr_score) < 1e-9

    def test_as_dict_keys(self):
        result = synthesise("tabular_hr", epsilon=2.0)
        d = result.as_dict()
        expected = {
            "domain", "epsilon", "n_rows", "n_seeds", "tstr_score",
            "fidelity_score", "privacy_auc", "utility_retention",
            "fk_violation_rate", "temporal_autocorr_loss",
            "downstream_accuracy", "downstream_retention",
        }
        assert set(d.keys()) == expected

    def test_synthesise_all_domains_returns_all_five(self):
        results = synthesise_all_domains(epsilon=2.0, n_seeds=2)
        assert set(results.keys()) == set(DOMAIN_NAMES)
        for r in results.values():
            assert isinstance(r, SynthesisResult)


# ── Pareto frontier ────────────────────────────────────────────────────────────

class TestPareto:
    def _sample_configs(self) -> list[dict]:
        return [
            {"epsilon": 0.1, "auc": 0.52, "tstr_score": 0.74, "fidelity": 0.71},
            {"epsilon": 1.0, "auc": 0.61, "tstr_score": 0.88, "fidelity": 0.85},
            {"epsilon": 10.0, "auc": 0.82, "tstr_score": 0.98, "fidelity": 0.95},
        ]

    def test_generate_pareto_frontier_length(self):
        configs = self._sample_configs()
        results = generate_pareto_frontier(configs)
        assert len(results) == len(configs)

    def test_generate_pareto_frontier_has_scores(self):
        results = generate_pareto_frontier(self._sample_configs())
        for r in results:
            assert "privacy_score" in r
            assert "utility_score" in r
            assert "fidelity_score" in r

    def test_is_pareto_efficient_length_matches(self):
        results = generate_pareto_frontier(self._sample_configs())
        mask = is_pareto_efficient(results)
        assert len(mask) == len(results)

    def test_dominated_point_not_efficient(self):
        # Point A dominates B on all three dimensions
        results = [
            {"privacy_score": 0.9, "utility_score": 0.9, "fidelity_score": 0.9},  # A
            {"privacy_score": 0.5, "utility_score": 0.5, "fidelity_score": 0.5},  # B (dominated)
        ]
        mask = is_pareto_efficient(results)
        assert mask[0] is True
        assert mask[1] is False

    def test_nondominated_both_efficient(self):
        # A beats B on privacy; B beats A on utility — neither dominates
        results = [
            {"privacy_score": 0.9, "utility_score": 0.5, "fidelity_score": 0.7},  # A
            {"privacy_score": 0.4, "utility_score": 0.9, "fidelity_score": 0.7},  # B
        ]
        mask = is_pareto_efficient(results)
        assert mask[0] is True
        assert mask[1] is True

    def test_pareto_efficient_configs_subset(self):
        results = generate_pareto_frontier(self._sample_configs())
        efficient = pareto_efficient_configs(results)
        assert len(efficient) <= len(results)
        assert all(r in results for r in efficient)

    def test_pareto_at_least_one_efficient(self):
        results = generate_pareto_frontier(self._sample_configs())
        efficient = pareto_efficient_configs(results)
        assert len(efficient) >= 1

    def test_compare_domains_returns_all_domains(self):
        domain_results: dict[str, list[dict]] = {}
        for name in DOMAIN_NAMES:
            configs = []
            for eps in EPSILON_VALUES:
                result = synthesise(name, eps, n_seeds=2)
                ev = {"epsilon": eps, "privacy_score": 1 - result.privacy_auc,
                      "utility_score": result.tstr_score, "fidelity_score": result.fidelity_score}
                configs.append(ev)
            domain_results[name] = configs

        summary = compare_domains(domain_results)
        assert set(summary.keys()) == set(DOMAIN_NAMES)

    def test_compare_domains_summary_keys(self):
        domain_results = {
            "tabular_hr": [
                {"epsilon": 2.0, "privacy_score": 0.35, "utility_score": 0.92, "fidelity_score": 0.89},
            ]
        }
        summary = compare_domains(domain_results)
        assert "n_pareto" in summary["tabular_hr"]
        assert "best_utility" in summary["tabular_hr"]
        assert "utility_at_eps2" in summary["tabular_hr"]
        assert "epsilon_for_90pct_utility" in summary["tabular_hr"]

    def test_relational_needs_higher_epsilon_for_90pct(self):
        domain_results: dict[str, list[dict]] = {}
        for name in ["tabular_hr", "ecommerce_relational"]:
            configs = []
            for eps in EPSILON_VALUES:
                result = synthesise(name, eps, n_seeds=2, seed=0)
                ev = {"epsilon": eps, "privacy_score": 1 - result.privacy_auc,
                      "utility_score": result.tstr_score, "fidelity_score": result.fidelity_score}
                configs.append(ev)
            domain_results[name] = configs

        summary = compare_domains(domain_results)
        hr_eps90 = summary["tabular_hr"]["epsilon_for_90pct_utility"]
        ecom_eps90 = summary["ecommerce_relational"]["epsilon_for_90pct_utility"]
        # Relational domain needs ≥ as much ε as tabular to hit 90% utility
        if not math.isnan(hr_eps90) and not math.isnan(ecom_eps90):
            assert ecom_eps90 >= hr_eps90
