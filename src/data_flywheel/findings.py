"""Research finding → synthetic training example conversion.

Experiment 1: Research-to-Training-Data Conversion Quality.

A ResearchFinding is a structured claim derived from an experiment
(e.g. "chunking strategy X outperforms Y by 15% on domain Z").
A FindingConverter converts each finding into an instruction-tuning QA
pair suitable for fine-tuning a model on the research domain.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ResearchFinding:
    """A single empirical finding from a research experiment."""

    finding_id: str
    domain: str            # e.g. "retrieval", "dp_synthesis", "model_collapse"
    claim: str             # human-readable finding
    evidence_strength: float  # 0.0–1.0; <0.5 = weak/speculative
    source: str            # "experiment" | "ablation" | "theoretical" | "hallucinated"
    metric_delta: float    # quantitative change (positive = improvement)
    is_hallucinated: bool = False  # True if this finding was fabricated


@dataclass
class SyntheticTrainingExample:
    """One QA training example synthesised from a ResearchFinding."""

    example_id: str
    finding_id: str
    domain: str
    question: str
    answer: str
    evidence_strength: float
    is_hallucinated: bool  # inherited from source finding
    # Estimated contribution to model improvement (positive or negative)
    quality_signal: float = field(init=False)

    def __post_init__(self) -> None:
        # Hallucinated examples inject negative signal; real ones contribute positively
        if self.is_hallucinated:
            self.quality_signal = -0.02 * (1.0 - self.evidence_strength)
        else:
            self.quality_signal = 0.005 * self.evidence_strength


# ── Canonical seed findings (stand-in for real experimental results) ──────────

SEED_FINDINGS: list[ResearchFinding] = [
    ResearchFinding(
        finding_id="F001",
        domain="retrieval",
        claim="Sentence-level chunking outperforms paragraph chunking by 15% on financial QA",
        evidence_strength=0.92,
        source="experiment",
        metric_delta=0.15,
    ),
    ResearchFinding(
        finding_id="F002",
        domain="dp_synthesis",
        claim="At ε=2.0, CTGAN TSTR F1 is within 4% of non-DP baseline on HR records",
        evidence_strength=0.95,
        source="experiment",
        metric_delta=-0.04,
    ),
    ResearchFinding(
        finding_id="F003",
        domain="model_collapse",
        claim="Tail entropy drops 51% over 5 generations without mitigation (fraud rate 1%)",
        evidence_strength=0.88,
        source="experiment",
        metric_delta=-0.51,
    ),
    ResearchFinding(
        finding_id="F004",
        domain="model_collapse",
        claim="Real-data anchoring at 10% maintains tail entropy within 10% of original",
        evidence_strength=0.85,
        source="experiment",
        metric_delta=0.10,
    ),
    ResearchFinding(
        finding_id="F005",
        domain="dp_synthesis",
        claim="Discrete Gaussian mechanism reduces integer-field constraint violations by 8%",
        evidence_strength=0.80,
        source="ablation",
        metric_delta=0.08,
    ),
    ResearchFinding(
        finding_id="F006",
        domain="retrieval",
        claim="BM25 hybrid retrieval improves recall@10 by 12% over dense-only on legal documents",
        evidence_strength=0.90,
        source="experiment",
        metric_delta=0.12,
    ),
]


# ── QA template library ───────────────────────────────────────────────────────

_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "retrieval": [
        "Which chunking strategy performs best for {domain} question answering?",
        "How does sentence-level chunking compare to paragraph chunking on {domain} tasks?",
        "What retrieval improvement does BM25 hybrid provide over dense-only on {domain}?",
    ],
    "dp_synthesis": [
        "What privacy budget (ε) is recommended for {domain} synthetic data generation?",
        "How much utility loss occurs at ε=2.0 for {domain} records?",
        "Which DP mechanism reduces constraint violations in {domain} tabular synthesis?",
    ],
    "model_collapse": [
        "What happens to tail-class representation after iterative retraining on synthetic data?",
        "How does real-data anchoring mitigate model collapse on {domain}?",
        "By how much does tail entropy decline over {n} retraining generations?",
    ],
}

_ANSWER_TEMPLATE = (
    "Based on empirical results (evidence strength {strength:.0%}): {claim}. "
    "The measured effect size is {delta:+.1%}."
)


class FindingConverter:
    """Convert ResearchFindings into SyntheticTrainingExamples."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def convert(self, finding: ResearchFinding) -> SyntheticTrainingExample:
        templates = _QUESTION_TEMPLATES.get(finding.domain, [
            "What does the research show about {domain}?",
        ])
        template = self._rng.choice(templates)
        question = template.format(
            domain=finding.domain,
            n=5,
        )
        answer = _ANSWER_TEMPLATE.format(
            strength=finding.evidence_strength,
            claim=finding.claim,
            delta=finding.metric_delta,
        )
        example_id = hashlib.md5(
            f"{finding.finding_id}-{question[:20]}".encode()
        ).hexdigest()[:8]
        return SyntheticTrainingExample(
            example_id=example_id,
            finding_id=finding.finding_id,
            domain=finding.domain,
            question=question,
            answer=answer,
            evidence_strength=finding.evidence_strength,
            is_hallucinated=finding.is_hallucinated,
        )

    def convert_batch(
        self, findings: Sequence[ResearchFinding]
    ) -> list[SyntheticTrainingExample]:
        return [self.convert(f) for f in findings]
