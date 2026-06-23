"""Run all three RSI data flywheel experiments (issue #34).

Usage:
    python scripts/run_data_flywheel.py

No external dependencies required. All experiments use deterministic
simulation with a fixed seed and no real LLM calls.
"""
from __future__ import annotations

import sys
import os

# Allow running from repo root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_flywheel.findings import SEED_FINDINGS, FindingConverter
from data_flywheel.hallucination import HallucinationInjector, HallucinationDetector
from data_flywheel.flywheel import DataFlywheelPipeline, FlywheelConfig


def _header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _separator() -> None:
    print("-" * 70)


# ── Experiment 1: Research-to-Training-Data Conversion Quality ────────────────

def experiment_1() -> None:
    _header("Experiment 1: Research-to-Training-Data Conversion Quality")
    print(f"  Converting {len(SEED_FINDINGS)} seed findings → synthetic QA pairs\n")

    converter = FindingConverter(seed=42)
    examples = converter.convert_batch(SEED_FINDINGS)

    _separator()
    print(f"  {'Finding':6}  {'Domain':16}  {'Strength':8}  {'Q.Signal':8}  Question (truncated)")
    _separator()
    for ex in examples:
        q_short = ex.question[:45] + "…" if len(ex.question) > 45 else ex.question
        print(
            f"  {ex.finding_id:6}  {ex.domain:16}  {ex.evidence_strength:.2f}     "
            f"{ex.quality_signal:+.4f}   {q_short}"
        )
    _separator()

    total_signal = sum(e.quality_signal for e in examples)
    avg_strength = sum(e.evidence_strength for e in examples) / len(examples)
    print(f"\n  Total quality signal : {total_signal:+.4f}")
    print(f"  Avg evidence strength: {avg_strength:.3f}")
    print(f"  All examples clean   : {all(not e.is_hallucinated for e in examples)}")
    print("\n  RESULT: High-quality seed findings produce net-positive training signal.")


# ── Experiment 2: Data Flywheel Compounding Over N Iterations ─────────────────

def experiment_2() -> None:
    _header("Experiment 2: Data Flywheel Compounding (3 Iterations)")

    # Run A: clean flywheel (no hallucinations)
    clean_cfg = FlywheelConfig(n_iterations=3, base_hallucination_rate=0.0, seed=42)
    clean_pipeline = DataFlywheelPipeline(config=clean_cfg)
    clean_results = clean_pipeline.run()

    # Run B: degraded flywheel (growing hallucinations)
    hall_cfg = FlywheelConfig(n_iterations=3, base_hallucination_rate=0.30, seed=42)
    hall_pipeline = DataFlywheelPipeline(config=hall_cfg)
    hall_results = hall_pipeline.run()

    print(f"\n  {'Iter':4}  {'TSTR (clean)':12}  {'TSTR (hall)':11}  {'Hall Rate (hall)':17}  {'Effective Ex':12}")
    _separator()
    for c, h in zip(clean_results, hall_results):
        print(
            f"  {c.iteration:4}  {c.tstr_score:.4f}       {h.tstr_score:.4f}      "
            f"{h.hallucination_report.hallucination_rate:.2%}             {h.effective_examples:>4}"
        )
    _separator()

    clean_summary = clean_pipeline.convergence_summary(clean_results)
    hall_summary = hall_pipeline.convergence_summary(hall_results)

    print(f"\n  Clean flywheel status         : {clean_summary['status']}")
    print(f"  Clean cumulative TSTR gain    : {clean_summary['cumulative_improvement']:+.4f}")
    print(f"  Noisy flywheel status         : {hall_summary['status']}")
    print(f"  Noisy cumulative TSTR change  : {hall_summary['cumulative_improvement']:+.4f}")
    print(f"  Hallucination rate growing    : {hall_summary['hallucination_growing']}")
    print("\n  RESULT: Clean flywheel compounds gains; hallucination injection erodes TSTR.")


# ── Experiment 3: Hallucination Injection Risk ────────────────────────────────

def experiment_3() -> None:
    _header("Experiment 3: Hallucination Injection Risk")

    hallucination_rates = [0.0, 0.10, 0.20, 0.30, 0.50]
    injector = HallucinationInjector(seed=42)
    detector = HallucinationDetector(false_negative_rate=0.15, false_positive_rate=0.05, seed=42)

    print(f"\n  {'Hall Rate':9}  {'Detected%':10}  {'Net Signal':10}  {'Pred TSTR Impact':18}  {'Avg Str (clean)':15}")
    _separator()

    for rate in hallucination_rates:
        findings = HallucinationInjector(seed=42).inject(SEED_FINDINGS, rate)
        report = detector.audit(findings, SEED_FINDINGS)
        print(
            f"  {rate:.0%}        {report.hallucination_rate:.1%}        "
            f"{report.net_quality_signal:+.4f}     {report.predicted_tstr_impact:+.4f}            "
            f"{report.avg_evidence_strength_clean:.3f}"
        )
    _separator()

    # Demonstrate model-collapse amplification
    print("\n  --- Iterative drift (growing injection rate per iteration) ---")
    print(f"\n  {'Iter':4}  {'Injection Rate':14}  {'Detected Hall%':15}  {'Net Signal':10}")
    _separator()
    curr_findings = list(SEED_FINDINGS)
    for iteration in range(1, 6):
        drift_rate = min(0.60, 0.10 + 0.08 * iteration)
        curr_findings = HallucinationInjector(seed=iteration).inject(curr_findings, drift_rate)
        report = detector.audit(curr_findings, SEED_FINDINGS)
        print(
            f"  {iteration:4}  {drift_rate:.0%}             "
            f"{report.hallucination_rate:.1%}             {report.net_quality_signal:+.4f}"
        )
    _separator()
    print("\n  RESULT: Without detection, hallucination rate compounds across iterations,")
    print("          driving net training signal negative by iteration 3–4.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\nEnterpriseSynth RSI Data Flywheel — Issue #34")
    print("Three experiments demonstrating recursive self-improvement dynamics")

    experiment_1()
    experiment_2()
    experiment_3()

    print()
    _header("Summary")
    print("  Exp 1 — High-quality findings → net-positive training signal (all clean)")
    print("  Exp 2 — Clean flywheel converges; noisy flywheel stagnates/degrades")
    print("  Exp 3 — Hallucination compounds per iteration; detection is essential")
    print()


if __name__ == "__main__":
    main()
