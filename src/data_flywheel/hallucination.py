"""Hallucination detection and injection for RSI data flywheel.

Experiment 3: Hallucination Injection Risk.

Models the risk that a fine-tuned model generates plausible-but-wrong
research findings, which then contaminate the next training iteration.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from data_flywheel.findings import ResearchFinding


# ── Hallucinated finding templates ────────────────────────────────────────────

_HALLUCINATED_CLAIMS: list[tuple[str, float, float]] = [
    # (claim, evidence_strength, metric_delta) — all wrong
    ("Sentence-level chunking underperforms paragraph chunking by 22% on financial QA", 0.70, -0.22),
    ("At ε=0.5, CTGAN TSTR F1 is within 1% of non-DP baseline", 0.65, -0.01),
    ("Tail entropy improves by 30% over 5 generations without mitigation", 0.60, 0.30),
    ("Discrete Gaussian mechanism increases constraint violations by 15%", 0.55, 0.15),
    ("BM25 hybrid retrieval degrades recall@10 by 8% on legal documents", 0.72, -0.08),
    ("Real-data anchoring at 50% is required to maintain tail diversity", 0.68, 0.50),
]


@dataclass
class HallucinationReport:
    total_examples: int
    hallucinated_count: int
    hallucination_rate: float
    avg_evidence_strength_clean: float
    avg_evidence_strength_hallucinated: float
    net_quality_signal: float
    predicted_tstr_impact: float  # positive = improvement, negative = degradation


class HallucinationInjector:
    """Simulate a fine-tuned model injecting hallucinated findings."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._counter = 0

    def inject(
        self,
        findings: list[ResearchFinding],
        hallucination_rate: float,
    ) -> list[ResearchFinding]:
        """Return a copy of findings with some fraction replaced by hallucinations.

        Args:
            findings: Real research findings from the current iteration.
            hallucination_rate: Fraction of findings to replace (0.0–1.0).
        """
        if not 0.0 <= hallucination_rate <= 1.0:
            raise ValueError(f"hallucination_rate must be in [0, 1], got {hallucination_rate}")

        result: list[ResearchFinding] = []
        for f in findings:
            if self._rng.random() < hallucination_rate:
                claim, strength, delta = self._rng.choice(_HALLUCINATED_CLAIMS)
                self._counter += 1
                result.append(ResearchFinding(
                    finding_id=f"H{self._counter:03d}",
                    domain=f.domain,
                    claim=claim,
                    evidence_strength=strength,
                    source="hallucinated",
                    metric_delta=delta,
                    is_hallucinated=True,
                ))
            else:
                result.append(f)
        return result


class HallucinationDetector:
    """Detect hallucinated findings by comparing against a ground-truth corpus.

    In real RSI systems this would use an LLM judge or retrieval-augmented
    fact-checker. Here we simulate it with a recall-precision model:
    the detector flags a finding as hallucinated if its claim does not
    appear (substring match) in the ground-truth corpus — with a tunable
    false-negative rate (missed hallucinations) and false-positive rate.
    """

    def __init__(
        self,
        false_negative_rate: float = 0.15,
        false_positive_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= false_negative_rate <= 1.0:
            raise ValueError("false_negative_rate must be in [0, 1]")
        if not 0.0 <= false_positive_rate <= 1.0:
            raise ValueError("false_positive_rate must be in [0, 1]")
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self._rng = random.Random(seed)

    def _ground_truth_match(
        self, finding: ResearchFinding, ground_truth: list[ResearchFinding]
    ) -> bool:
        """True if any ground-truth claim substantially overlaps this finding's claim."""
        claim_lower = finding.claim.lower()
        for gt in ground_truth:
            # Use word-overlap heuristic (avoids any external NLP dependency)
            gt_words = set(gt.claim.lower().split())
            claim_words = set(claim_lower.split())
            overlap = len(gt_words & claim_words) / max(len(gt_words), 1)
            if overlap > 0.40:
                return True
        return False

    def is_hallucinated(
        self, finding: ResearchFinding, ground_truth: list[ResearchFinding]
    ) -> bool:
        """Predict whether a finding is hallucinated (with simulated detection noise)."""
        truly_hallucinated = finding.is_hallucinated or not self._ground_truth_match(
            finding, ground_truth
        )
        if truly_hallucinated:
            # Miss it with false_negative_rate
            return self._rng.random() > self.false_negative_rate
        else:
            # Flag it incorrectly with false_positive_rate
            return self._rng.random() < self.false_positive_rate

    def audit(
        self,
        findings: list[ResearchFinding],
        ground_truth: list[ResearchFinding],
        examples_quality_signals: list[float] | None = None,
    ) -> HallucinationReport:
        """Audit a batch of findings and produce a HallucinationReport."""
        from data_flywheel.findings import FindingConverter
        converter = FindingConverter()

        detected_hallucinated = [
            f for f in findings if self.is_hallucinated(f, ground_truth)
        ]
        clean = [f for f in findings if f not in detected_hallucinated]

        hallucination_rate = len(detected_hallucinated) / max(len(findings), 1)

        avg_clean_strength = (
            sum(f.evidence_strength for f in clean) / len(clean) if clean else 0.0
        )
        avg_hall_strength = (
            sum(f.evidence_strength for f in detected_hallucinated) / len(detected_hallucinated)
            if detected_hallucinated else 0.0
        )

        examples = converter.convert_batch(findings)
        net_signal = sum(e.quality_signal for e in examples)
        # Predicted TSTR impact: positive signal compounds, negative drags
        predicted_impact = net_signal / max(len(examples), 1)

        return HallucinationReport(
            total_examples=len(findings),
            hallucinated_count=len(detected_hallucinated),
            hallucination_rate=hallucination_rate,
            avg_evidence_strength_clean=avg_clean_strength,
            avg_evidence_strength_hallucinated=avg_hall_strength,
            net_quality_signal=net_signal,
            predicted_tstr_impact=predicted_impact,
        )
