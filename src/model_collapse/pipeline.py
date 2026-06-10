"""Multi-generation synthetic data pipeline with model collapse simulation.

Models the iterative retraining loop:
  Gen 0 (original) → Gen 1 (trained on Gen 0 synthetic) → … → Gen N

Collapse mechanism: each generation, values in the tail of the distribution
are dropped at rate *collapse_rate*, simulating how fine-tuned models stop
generating rare-but-critical records (fraud, security incidents).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

from model_collapse.metrics import (
    Record,
    coverage_entropy,
    tail_coverage_entropy,
    tail_divergence,
    minority_class_representation,
)


# ---------------------------------------------------------------------------
# Dataset generation helpers
# ---------------------------------------------------------------------------

def make_enterprise_dataset(
    n_records: int = 500,
    *,
    field: str = "record_type",
    seed: int = 42,
) -> list[Record]:
    """Generate a synthetic enterprise dataset with realistic long-tail distribution.

    Categories mimic enterprise schemas:
      - routine (60%), review (20%), flagged (10%), incident (6%),
        fraud (3%), critical_security (1%)

    The last two are the minority / tail classes that collapse first.
    """
    rng = random.Random(seed)
    categories = [
        ("routine",           0.60),
        ("review",            0.20),
        ("flagged",           0.10),
        ("incident",          0.06),
        ("fraud",             0.03),
        ("critical_security", 0.01),
    ]
    records: list[Record] = []
    for _ in range(n_records):
        r = rng.random()
        cumulative = 0.0
        chosen = categories[-1][0]
        for cat, prob in categories:
            cumulative += prob
            if r < cumulative:
                chosen = cat
                break
        records.append({field: chosen, "value": round(rng.gauss(0, 1), 3)})
    return records


def _simulate_collapse_step(
    records: list[Record],
    field: str,
    collapse_rate: float,
    rng: random.Random,
) -> list[Record]:
    """One generation of collapse: rare values are dropped at *collapse_rate*.

    Values with frequency below the median frequency are treated as tail.
    Each tail record is kept with probability (1 - collapse_rate).
    Common records are kept with probability (1 - collapse_rate * 0.1).
    """
    counts: Counter[str] = Counter(str(r.get(field, "")) for r in records)
    total = len(records)
    median_freq = sorted(counts.values())[len(counts) // 2] / total

    result: list[Record] = []
    for rec in records:
        val = str(rec.get(field, ""))
        freq = counts[val] / total
        if freq < median_freq:
            # Tail record — drop with collapse_rate probability
            if rng.random() > collapse_rate:
                result.append(rec)
        else:
            # Common record — slight drift away
            if rng.random() > collapse_rate * 0.1:
                result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------

class GenerationResult:
    """Metrics snapshot for one generation."""

    def __init__(
        self,
        generation: int,
        records: list[Record],
        original_records: list[Record],
        field: str,
        minority_classes: Sequence[str],
    ) -> None:
        self.generation = generation
        self.n_records = len(records)
        self.coverage_entropy = coverage_entropy(records, field)
        self.tail_entropy = tail_coverage_entropy(records, field)
        self.tail_divergence = tail_divergence(original_records, records, field)
        self.minority_representation = minority_class_representation(
            records, field, minority_classes
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "n_records": self.n_records,
            "coverage_entropy": self.coverage_entropy,
            "tail_entropy": self.tail_entropy,
            "tail_divergence": self.tail_divergence,
            "minority_representation": self.minority_representation,
        }


# ---------------------------------------------------------------------------
# Multi-generation pipeline
# ---------------------------------------------------------------------------

def run_model_collapse_pipeline(
    n_generations: int = 5,
    n_records: int = 500,
    *,
    field: str = "record_type",
    minority_classes: Sequence[str] = ("fraud", "critical_security"),
    collapse_rate: float = 0.30,
    seed: int = 42,
) -> list[GenerationResult]:
    """Run an N-generation collapse pipeline.

    Returns one GenerationResult per generation (0 = original baseline).
    """
    rng = random.Random(seed)
    original = make_enterprise_dataset(n_records, field=field, seed=seed)
    current = list(original)
    results: list[GenerationResult] = [
        GenerationResult(0, current, original, field, minority_classes)
    ]
    for gen in range(1, n_generations + 1):
        current = _simulate_collapse_step(current, field, collapse_rate, rng)
        if not current:
            # Full collapse — fill with single dominant class
            current = [{field: "routine", "value": 0.0}]
        results.append(
            GenerationResult(gen, current, original, field, minority_classes)
        )
    return results
