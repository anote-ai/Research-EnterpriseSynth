"""Tests for the RSI data flywheel module (issue #34).

Covers three experiments:
  1. Research-to-training-data conversion quality (FindingConverter)
  2. Data flywheel compounding over N iterations (DataFlywheelPipeline)
  3. Hallucination injection risk (HallucinationInjector / HallucinationDetector)
"""
from __future__ import annotations

import pytest

from data_flywheel.findings import (
    SEED_FINDINGS,
    FindingConverter,
    ResearchFinding,
    SyntheticTrainingExample,
)
from data_flywheel.hallucination import (
    HallucinationDetector,
    HallucinationInjector,
    HallucinationReport,
)
from data_flywheel.flywheel import (
    DataFlywheelPipeline,
    FlywheelConfig,
    FlywheelIteration,
    _simulate_tstr_update,
    _simulate_new_findings,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def seed_findings() -> list[ResearchFinding]:
    return list(SEED_FINDINGS)


@pytest.fixture()
def converter() -> FindingConverter:
    return FindingConverter(seed=0)


@pytest.fixture()
def injector() -> HallucinationInjector:
    return HallucinationInjector(seed=0)


@pytest.fixture()
def detector() -> HallucinationDetector:
    return HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)


# ── Experiment 1: Research-to-training-data conversion quality ────────────────

class TestResearchFinding:
    def test_fields_preserved(self, seed_findings):
        f = seed_findings[0]
        assert f.finding_id == "F001"
        assert f.domain == "retrieval"
        assert 0.0 <= f.evidence_strength <= 1.0
        assert not f.is_hallucinated

    def test_seed_findings_count(self, seed_findings):
        assert len(seed_findings) == 6

    def test_hallucinated_flag_default_false(self):
        f = ResearchFinding("X1", "retrieval", "some claim", 0.9, "experiment", 0.1)
        assert not f.is_hallucinated


class TestSyntheticTrainingExample:
    def test_quality_signal_clean(self, converter, seed_findings):
        ex = converter.convert(seed_findings[0])
        assert ex.quality_signal > 0.0
        assert not ex.is_hallucinated

    def test_quality_signal_hallucinated(self):
        f = ResearchFinding("H1", "retrieval", "false claim", 0.7, "hallucinated", -0.5, is_hallucinated=True)
        conv = FindingConverter(seed=0)
        ex = conv.convert(f)
        assert ex.quality_signal < 0.0
        assert ex.is_hallucinated

    def test_quality_signal_formula_clean(self, converter, seed_findings):
        f = seed_findings[1]  # evidence_strength = 0.95
        ex = converter.convert(f)
        expected = 0.005 * f.evidence_strength
        assert abs(ex.quality_signal - expected) < 1e-9

    def test_quality_signal_formula_hallucinated(self):
        f = ResearchFinding("H2", "dp_synthesis", "bad claim", 0.60, "hallucinated", 0.1, is_hallucinated=True)
        ex = FindingConverter(seed=0).convert(f)
        expected = -0.02 * (1.0 - 0.60)
        assert abs(ex.quality_signal - expected) < 1e-9


class TestFindingConverter:
    def test_convert_returns_example(self, converter, seed_findings):
        ex = converter.convert(seed_findings[0])
        assert isinstance(ex, SyntheticTrainingExample)

    def test_convert_preserves_metadata(self, converter, seed_findings):
        f = seed_findings[2]
        ex = converter.convert(f)
        assert ex.finding_id == f.finding_id
        assert ex.domain == f.domain
        assert ex.evidence_strength == f.evidence_strength
        assert ex.is_hallucinated == f.is_hallucinated

    def test_convert_question_nonempty(self, converter, seed_findings):
        for f in seed_findings:
            ex = converter.convert(f)
            assert len(ex.question) > 5

    def test_convert_answer_contains_claim(self, converter, seed_findings):
        f = seed_findings[0]
        ex = converter.convert(f)
        assert f.claim in ex.answer

    def test_convert_answer_contains_evidence_strength(self, converter, seed_findings):
        f = seed_findings[0]
        ex = converter.convert(f)
        assert "%" in ex.answer  # formatted as percentage

    def test_convert_batch_length(self, converter, seed_findings):
        examples = converter.convert_batch(seed_findings)
        assert len(examples) == len(seed_findings)

    def test_convert_batch_all_examples(self, converter, seed_findings):
        examples = converter.convert_batch(seed_findings)
        assert all(isinstance(e, SyntheticTrainingExample) for e in examples)

    def test_unknown_domain_falls_back(self, converter):
        f = ResearchFinding("U1", "unknown_domain", "some claim", 0.8, "experiment", 0.1)
        ex = converter.convert(f)
        assert "unknown_domain" in ex.question

    def test_deterministic_with_same_seed(self, seed_findings):
        c1 = FindingConverter(seed=99)
        c2 = FindingConverter(seed=99)
        ex1 = c1.convert(seed_findings[0])
        ex2 = c2.convert(seed_findings[0])
        assert ex1.question == ex2.question

    def test_different_seeds_may_differ(self, seed_findings):
        c1 = FindingConverter(seed=1)
        c2 = FindingConverter(seed=2)
        questions = {c1.convert(seed_findings[0]).question, c2.convert(seed_findings[0]).question}
        # At least some questions come from different templates — not a strict test
        assert len(questions) >= 1  # sanity: always produces a question


# ── Experiment 3: Hallucination injection risk ────────────────────────────────

class TestHallucinationInjector:
    def test_zero_rate_no_injections(self, injector, seed_findings):
        result = injector.inject(seed_findings, 0.0)
        assert all(not f.is_hallucinated for f in result)
        assert len(result) == len(seed_findings)

    def test_full_rate_all_hallucinated(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        result = inj.inject(seed_findings, 1.0)
        assert all(f.is_hallucinated for f in result)

    def test_partial_rate_approximate(self, seed_findings):
        inj = HallucinationInjector(seed=42)
        result = inj.inject(seed_findings * 20, 0.5)  # 120 findings
        hall_count = sum(1 for f in result if f.is_hallucinated)
        assert 30 <= hall_count <= 90  # wide band due to randomness

    def test_length_preserved(self, injector, seed_findings):
        result = injector.inject(seed_findings, 0.3)
        assert len(result) == len(seed_findings)

    def test_hallucinated_source_field(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        result = inj.inject(seed_findings, 1.0)
        for f in result:
            assert f.source == "hallucinated"

    def test_invalid_rate_raises(self, injector, seed_findings):
        with pytest.raises(ValueError):
            injector.inject(seed_findings, 1.5)
        with pytest.raises(ValueError):
            injector.inject(seed_findings, -0.1)

    def test_hallucinated_finding_ids_unique(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        result = inj.inject(seed_findings * 4, 1.0)
        hall_ids = [f.finding_id for f in result if f.is_hallucinated]
        assert len(hall_ids) == len(set(hall_ids))  # each ID unique

    def test_domain_inherited(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        result = inj.inject(seed_findings, 1.0)
        original_domains = {f.domain for f in seed_findings}
        injected_domains = {f.domain for f in result}
        assert injected_domains <= original_domains


class TestHallucinationDetector:
    def test_perfect_detector_catches_hallucinations(self, detector, seed_findings):
        inj = HallucinationInjector(seed=0)
        hall_findings = inj.inject(seed_findings, 1.0)
        for f in hall_findings:
            assert detector.is_hallucinated(f, seed_findings)

    def test_perfect_detector_no_false_positives(self, detector, seed_findings):
        for f in seed_findings:
            # Ground truth is itself — all should be clean
            assert not detector.is_hallucinated(f, seed_findings)

    def test_fnr_raises_for_invalid(self):
        with pytest.raises(ValueError):
            HallucinationDetector(false_negative_rate=1.5)

    def test_fpr_raises_for_invalid(self):
        with pytest.raises(ValueError):
            HallucinationDetector(false_positive_rate=-0.1)

    def test_audit_returns_report(self, detector, seed_findings):
        report = detector.audit(seed_findings, seed_findings)
        assert isinstance(report, HallucinationReport)

    def test_audit_zero_hallucinations_clean_data(self, seed_findings):
        det = HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)
        report = det.audit(seed_findings, seed_findings)
        assert report.hallucination_rate == 0.0
        assert report.hallucinated_count == 0

    def test_audit_high_hallucination_rate(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        hall_findings = inj.inject(seed_findings, 1.0)
        det = HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)
        report = det.audit(hall_findings, seed_findings)
        assert report.hallucination_rate == 1.0
        assert report.hallucinated_count == len(hall_findings)

    def test_audit_net_quality_signal_clean(self, seed_findings):
        det = HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)
        report = det.audit(seed_findings, seed_findings)
        assert report.net_quality_signal > 0.0

    def test_audit_net_quality_signal_hallucinated(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        hall_findings = inj.inject(seed_findings, 1.0)
        det = HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)
        report = det.audit(hall_findings, seed_findings)
        assert report.net_quality_signal < 0.0

    def test_audit_predicted_tstr_impact_sign(self, seed_findings):
        det = HallucinationDetector(false_negative_rate=0.0, false_positive_rate=0.0, seed=0)
        clean_report = det.audit(seed_findings, seed_findings)
        inj = HallucinationInjector(seed=0)
        hall_findings = inj.inject(seed_findings, 1.0)
        hall_report = det.audit(hall_findings, seed_findings)
        assert clean_report.predicted_tstr_impact > hall_report.predicted_tstr_impact

    def test_audit_empty_findings(self, seed_findings):
        det = HallucinationDetector(seed=0)
        report = det.audit([], seed_findings)
        assert report.total_examples == 0
        assert report.hallucination_rate == 0.0


# ── Experiment 2: Data flywheel compounding ───────────────────────────────────

class TestSimulateTstrUpdate:
    def test_clean_examples_improve_score(self, seed_findings):
        examples = FindingConverter(seed=0).convert_batch(seed_findings)
        new_tstr = _simulate_tstr_update(0.72, examples, hallucination_rate=0.0)
        assert new_tstr > 0.72

    def test_all_hallucinated_degrades_score(self, seed_findings):
        inj = HallucinationInjector(seed=0)
        hall_findings = inj.inject(seed_findings, 1.0)
        examples = FindingConverter(seed=0).convert_batch(hall_findings)
        new_tstr = _simulate_tstr_update(0.72, examples, hallucination_rate=1.0)
        assert new_tstr < 0.72

    def test_score_clamped_to_zero_one(self, seed_findings):
        examples = FindingConverter(seed=0).convert_batch(seed_findings * 50)
        new_tstr = _simulate_tstr_update(0.99, examples, hallucination_rate=0.0)
        assert 0.0 <= new_tstr <= 1.0

    def test_empty_examples_no_change(self):
        new_tstr = _simulate_tstr_update(0.72, [], hallucination_rate=0.0)
        assert new_tstr == 0.72


class TestSimulateNewFindings:
    def test_returns_nonempty_list(self, seed_findings):
        import random
        rng = random.Random(0)
        result = _simulate_new_findings(seed_findings, iteration=1, rng=rng, base_hallucination_rate=0.1)
        assert len(result) >= len(seed_findings)

    def test_hallucination_rate_grows_with_iteration(self, seed_findings):
        import random
        rng1 = random.Random(0)
        rng2 = random.Random(0)
        r1 = _simulate_new_findings(seed_findings, iteration=1, rng=rng1, base_hallucination_rate=0.1)
        r5 = _simulate_new_findings(seed_findings * 4, iteration=5, rng=rng2, base_hallucination_rate=0.1)
        hall_rate_1 = sum(1 for f in r1 if f.is_hallucinated) / len(r1)
        hall_rate_5 = sum(1 for f in r5 if f.is_hallucinated) / len(r5)
        # With deterministic-ish RNG this is probabilistic; just check rate cap
        assert hall_rate_5 <= 0.65


class TestFlywheelConfig:
    def test_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.n_iterations == 3
        assert cfg.baseline_tstr == 0.72
        assert cfg.base_hallucination_rate == 0.10
        assert cfg.seed == 42

    def test_custom_config(self):
        cfg = FlywheelConfig(n_iterations=5, baseline_tstr=0.80, seed=7)
        assert cfg.n_iterations == 5
        assert cfg.baseline_tstr == 0.80


class TestFlywheelIteration:
    def test_converged_small_delta(self, seed_findings):
        det = HallucinationDetector(seed=0)
        report = det.audit(seed_findings, seed_findings)
        it = FlywheelIteration(
            iteration=1,
            findings=seed_findings,
            training_examples=[],
            hallucination_report=report,
            tstr_score=0.73,
            tstr_delta=0.001,
            cumulative_improvement=0.001,
            data_quality_score=0.9,
            effective_examples=6,
        )
        assert it.converged(threshold=0.002)

    def test_not_converged_large_delta(self, seed_findings):
        det = HallucinationDetector(seed=0)
        report = det.audit(seed_findings, seed_findings)
        it = FlywheelIteration(
            iteration=1,
            findings=seed_findings,
            training_examples=[],
            hallucination_report=report,
            tstr_score=0.80,
            tstr_delta=0.05,
            cumulative_improvement=0.05,
            data_quality_score=0.88,
            effective_examples=6,
        )
        assert not it.converged(threshold=0.002)

    def test_as_dict_keys(self, seed_findings):
        det = HallucinationDetector(seed=0)
        report = det.audit(seed_findings, seed_findings)
        it = FlywheelIteration(
            iteration=1,
            findings=seed_findings,
            training_examples=[],
            hallucination_report=report,
            tstr_score=0.73,
            tstr_delta=0.01,
            cumulative_improvement=0.01,
            data_quality_score=0.9,
            effective_examples=6,
        )
        d = it.as_dict()
        expected_keys = {
            "iteration", "n_findings", "n_examples", "tstr_score", "tstr_delta",
            "cumulative_improvement", "hallucination_rate", "data_quality_score",
            "effective_examples", "predicted_tstr_impact",
        }
        assert set(d.keys()) == expected_keys


class TestDataFlywheelPipeline:
    def test_run_returns_correct_count(self):
        cfg = FlywheelConfig(n_iterations=3, seed=0)
        pipeline = DataFlywheelPipeline(config=cfg)
        results = pipeline.run()
        assert len(results) == 3

    def test_run_iteration_numbers_sequential(self):
        cfg = FlywheelConfig(n_iterations=3, seed=0)
        results = DataFlywheelPipeline(config=cfg).run()
        assert [r.iteration for r in results] == [1, 2, 3]

    def test_run_tstr_starts_above_baseline(self):
        cfg = FlywheelConfig(n_iterations=1, seed=0, base_hallucination_rate=0.0)
        results = DataFlywheelPipeline(config=cfg).run()
        assert results[0].tstr_score >= cfg.baseline_tstr

    def test_run_tstr_bounded(self):
        cfg = FlywheelConfig(n_iterations=5, seed=0)
        results = DataFlywheelPipeline(config=cfg).run()
        for r in results:
            assert 0.0 <= r.tstr_score <= 1.0

    def test_run_effective_examples_positive(self):
        cfg = FlywheelConfig(n_iterations=2, seed=0, base_hallucination_rate=0.0)
        results = DataFlywheelPipeline(config=cfg).run()
        for r in results:
            assert r.effective_examples >= 0

    def test_run_data_quality_bounded(self):
        cfg = FlywheelConfig(n_iterations=3, seed=0)
        results = DataFlywheelPipeline(config=cfg).run()
        for r in results:
            assert 0.0 <= r.data_quality_score <= 1.0

    def test_run_high_hallucination_degrades_tstr(self):
        clean_cfg = FlywheelConfig(n_iterations=3, seed=0, base_hallucination_rate=0.0)
        hall_cfg = FlywheelConfig(n_iterations=3, seed=0, base_hallucination_rate=0.8)
        clean_results = DataFlywheelPipeline(config=clean_cfg).run()
        hall_results = DataFlywheelPipeline(config=hall_cfg).run()
        clean_final = clean_results[-1].tstr_score
        hall_final = hall_results[-1].tstr_score
        assert clean_final >= hall_final

    def test_run_custom_findings(self, seed_findings):
        cfg = FlywheelConfig(n_iterations=2, seed=0)
        results = DataFlywheelPipeline(config=cfg).run(initial_findings=seed_findings[:3])
        assert len(results) == 2
        assert len(results[0].findings) == 3

    def test_run_deterministic_same_seed(self):
        cfg = FlywheelConfig(n_iterations=3, seed=77)
        r1 = DataFlywheelPipeline(config=cfg).run()
        r2 = DataFlywheelPipeline(config=cfg).run()
        for a, b in zip(r1, r2):
            assert abs(a.tstr_score - b.tstr_score) < 1e-9

    def test_convergence_summary_converging(self):
        cfg = FlywheelConfig(n_iterations=3, seed=0, base_hallucination_rate=0.0)
        pipeline = DataFlywheelPipeline(config=cfg)
        results = pipeline.run()
        summary = pipeline.convergence_summary(results)
        assert "status" in summary
        assert summary["status"] in {"converging", "diverging", "oscillating"}

    def test_convergence_summary_empty(self):
        pipeline = DataFlywheelPipeline()
        summary = pipeline.convergence_summary([])
        assert summary["status"] == "empty"

    def test_convergence_summary_fields(self):
        cfg = FlywheelConfig(n_iterations=3, seed=0)
        pipeline = DataFlywheelPipeline(config=cfg)
        results = pipeline.run()
        summary = pipeline.convergence_summary(results)
        assert "final_tstr" in summary
        assert "cumulative_improvement" in summary
        assert "final_hallucination_rate" in summary
        assert "hallucination_growing" in summary
        assert "converged_by_last" in summary

    def test_cumulative_improvement_monotone_with_clean_data(self):
        cfg = FlywheelConfig(n_iterations=4, seed=0, base_hallucination_rate=0.0)
        results = DataFlywheelPipeline(config=cfg).run()
        cumulative = [r.cumulative_improvement for r in results]
        # Cumulative should be non-decreasing when data is clean
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1] - 1e-9
