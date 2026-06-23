"""Multi-round RSI data flywheel pipeline.

Experiment 2: Data Flywheel Compounding.

Each iteration:
  1. Convert research findings → synthetic training examples
  2. Simulate model fine-tuning (TSTR score update)
  3. Simulate the fine-tuned model generating new findings
     (with a hallucination_rate that grows as the model drifts)
  4. Repeat for n_iterations

Tracks whether the flywheel converges, diverges, or oscillates.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from data_flywheel.findings import (
    SEED_FINDINGS,
    FindingConverter,
    ResearchFinding,
    SyntheticTrainingExample,
)
from data_flywheel.hallucination import HallucinationDetector, HallucinationInjector, HallucinationReport


# ── Iteration result ──────────────────────────────────────────────────────────

@dataclass
class FlywheelIteration:
    """Metrics captured at the end of one RSI flywheel iteration."""

    iteration: int
    findings: list[ResearchFinding]
    training_examples: list[SyntheticTrainingExample]
    hallucination_report: HallucinationReport

    # TSTR score: 0.0 = no improvement over baseline; 1.0 = oracle
    tstr_score: float
    tstr_delta: float           # change vs previous iteration
    cumulative_improvement: float

    # Flywheel health signals
    data_quality_score: float   # mean evidence_strength of clean examples
    effective_examples: int     # non-hallucinated training examples used

    def converged(self, threshold: float = 0.002) -> bool:
        """True if improvement per iteration is below threshold."""
        return abs(self.tstr_delta) < threshold

    def as_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "n_findings": len(self.findings),
            "n_examples": len(self.training_examples),
            "tstr_score": round(self.tstr_score, 4),
            "tstr_delta": round(self.tstr_delta, 4),
            "cumulative_improvement": round(self.cumulative_improvement, 4),
            "hallucination_rate": round(self.hallucination_report.hallucination_rate, 4),
            "data_quality_score": round(self.data_quality_score, 4),
            "effective_examples": self.effective_examples,
            "predicted_tstr_impact": round(self.hallucination_report.predicted_tstr_impact, 4),
        }


# ── TSTR simulation model ─────────────────────────────────────────────────────

def _simulate_tstr_update(
    baseline_tstr: float,
    examples: list[SyntheticTrainingExample],
    hallucination_rate: float,
) -> float:
    """Simulate a TSTR score after fine-tuning on the given examples.

    Clean examples contribute +signal; hallucinated ones subtract.
    Diminishing returns: improvement scales as sqrt of clean examples.
    Hallucination drag is proportional to the hallucination rate.
    """
    clean = [e for e in examples if not e.is_hallucinated]
    n_clean = len(clean)
    n_hall = len(examples) - n_clean

    # Positive contribution from clean examples (diminishing returns)
    positive_signal = 0.03 * math.sqrt(n_clean) if n_clean > 0 else 0.0

    # Drag from hallucinated examples (linear, weighted by rate)
    negative_drag = 0.05 * hallucination_rate * n_hall

    delta = positive_signal - negative_drag
    new_score = min(1.0, max(0.0, baseline_tstr + delta))
    return new_score


def _simulate_new_findings(
    current_findings: list[ResearchFinding],
    iteration: int,
    rng: random.Random,
    base_hallucination_rate: float,
) -> list[ResearchFinding]:
    """Simulate the fine-tuned model generating new research findings.

    Hallucination rate grows with each iteration as the model drifts
    from ground-truth (model collapse analog for factual knowledge).
    Growth rate: rate = base + 0.05 * iteration (capped at 0.6).
    """
    drift_rate = min(0.60, base_hallucination_rate + 0.05 * iteration)
    injector = HallucinationInjector(seed=rng.randint(0, 2**31))

    # Generate slightly more findings each iteration (model explores more)
    n_new = len(current_findings) + rng.randint(0, 2)
    extended = (current_findings * ((n_new // len(current_findings)) + 1))[:n_new]
    return injector.inject(extended, drift_rate)


# ── Main pipeline ─────────────────────────────────────────────────────────────

@dataclass
class FlywheelConfig:
    n_iterations: int = 3
    base_hallucination_rate: float = 0.10   # starting hallucination rate (iteration 0)
    baseline_tstr: float = 0.72             # model TSTR before any flywheel fine-tuning
    detector_fnr: float = 0.15             # hallucination detector false-negative rate
    detector_fpr: float = 0.05             # hallucination detector false-positive rate
    seed: int = 42


class DataFlywheelPipeline:
    """Run an N-iteration RSI data flywheel and return per-iteration metrics."""

    def __init__(self, config: FlywheelConfig | None = None) -> None:
        self.config = config or FlywheelConfig()

    def run(
        self,
        initial_findings: list[ResearchFinding] | None = None,
    ) -> list[FlywheelIteration]:
        cfg = self.config
        rng = random.Random(cfg.seed)
        converter = FindingConverter(seed=cfg.seed)
        detector = HallucinationDetector(
            false_negative_rate=cfg.detector_fnr,
            false_positive_rate=cfg.detector_fpr,
            seed=cfg.seed,
        )
        ground_truth = list(SEED_FINDINGS)
        current_findings = list(initial_findings or SEED_FINDINGS)
        tstr = cfg.baseline_tstr
        cumulative = 0.0
        results: list[FlywheelIteration] = []

        for i in range(cfg.n_iterations):
            # Step 1: Convert findings → training examples
            examples = converter.convert_batch(current_findings)

            # Step 2: Audit for hallucinations
            report = detector.audit(current_findings, ground_truth)
            hall_rate = report.hallucination_rate

            # Step 3: Simulate fine-tuning TSTR update
            new_tstr = _simulate_tstr_update(tstr, examples, hall_rate)
            delta = new_tstr - tstr
            cumulative += delta

            # Data quality = mean evidence strength of detected-clean findings
            clean_findings = [
                f for f in current_findings
                if not detector.is_hallucinated(f, ground_truth)
            ]
            dq = (
                sum(f.evidence_strength for f in clean_findings) / len(clean_findings)
                if clean_findings else 0.0
            )

            results.append(FlywheelIteration(
                iteration=i + 1,
                findings=current_findings,
                training_examples=examples,
                hallucination_report=report,
                tstr_score=new_tstr,
                tstr_delta=delta,
                cumulative_improvement=cumulative,
                data_quality_score=dq,
                effective_examples=len(clean_findings),
            ))
            tstr = new_tstr

            # Step 4: Generate new findings for next iteration
            current_findings = _simulate_new_findings(
                current_findings, i + 1, rng, cfg.base_hallucination_rate
            )

        return results

    def convergence_summary(self, results: list[FlywheelIteration]) -> dict[str, object]:
        """Summarise whether the flywheel converges, diverges, or oscillates."""
        if not results:
            return {"status": "empty"}

        deltas = [r.tstr_delta for r in results]
        hall_rates = [r.hallucination_report.hallucination_rate for r in results]

        last_delta = deltas[-1]
        is_converging = all(abs(d) <= abs(deltas[0]) + 1e-9 for d in deltas)
        is_diverging = any(abs(d) > abs(deltas[0]) * 1.5 for d in deltas[1:])
        hall_growing = hall_rates[-1] > hall_rates[0] if len(hall_rates) > 1 else False

        if is_converging and not is_diverging:
            status = "converging"
        elif is_diverging:
            status = "diverging"
        else:
            status = "oscillating"

        return {
            "status": status,
            "n_iterations": len(results),
            "final_tstr": round(results[-1].tstr_score, 4),
            "cumulative_improvement": round(results[-1].cumulative_improvement, 4),
            "final_hallucination_rate": round(hall_rates[-1], 4),
            "hallucination_growing": hall_growing,
            "converged_by_last": results[-1].converged(),
        }
