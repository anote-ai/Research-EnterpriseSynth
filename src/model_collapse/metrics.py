"""Diversity metrics for model collapse detection.

All metrics operate on lists of flat dicts (records) and a named field.
No external dependencies — pure Python stdlib only.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


Record = dict[str, object]


# ---------------------------------------------------------------------------
# Shannon entropy helpers
# ---------------------------------------------------------------------------

def _value_distribution(records: Sequence[Record], field: str) -> dict[str, float]:
    """Return a normalised probability distribution over values of *field*."""
    counts: Counter[str] = Counter(str(r.get(field, "__missing__")) for r in records)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {v: c / total for v, c in counts.items()}


def _shannon_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy in bits, normalised to [0, 1] by H_max = log2(n)."""
    n = len(dist)
    if n <= 1:
        return 0.0
    raw = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    return raw / math.log2(n)


# ---------------------------------------------------------------------------
# 1. Coverage entropy — diversity of a field's value distribution
# ---------------------------------------------------------------------------

def coverage_entropy(records: Sequence[Record], field: str) -> float:
    """Normalised Shannon entropy [0, 1] for *field* across all records.

    1.0 = perfectly uniform distribution; 0.0 = single value.
    """
    if not records:
        return 0.0
    return _shannon_entropy(_value_distribution(records, field))


# ---------------------------------------------------------------------------
# 2. Tail coverage entropy — diversity within the rare-value tail
# ---------------------------------------------------------------------------

def tail_coverage_entropy(
    records: Sequence[Record],
    field: str,
    *,
    tail_percentile: float = 0.20,
) -> float:
    """Entropy computed only over values that fall in the bottom
    *tail_percentile* of the frequency distribution.

    For enterprise schemas, these are low-frequency but critical records
    (fraud cases, security incidents).  Returns 0.0 if no tail values exist.
    """
    if not records:
        return 0.0
    counts: Counter[str] = Counter(str(r.get(field, "__missing__")) for r in records)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Threshold: values whose frequency <= tail_percentile are "tail"
    threshold = tail_percentile * total
    tail_counts = {v: c for v, c in counts.items() if c <= threshold}
    if not tail_counts:
        return 0.0
    tail_total = sum(tail_counts.values())
    tail_dist = {v: c / tail_total for v, c in tail_counts.items()}
    return _shannon_entropy(tail_dist)


# ---------------------------------------------------------------------------
# 3. Minority-class representation — fraction of rare records retained
# ---------------------------------------------------------------------------

def minority_class_representation(
    records: Sequence[Record],
    field: str,
    minority_classes: Sequence[str],
) -> float:
    """Fraction of records in *records* that belong to *minority_classes*.

    Returns a score in [0, 1]; higher is better (more tail records present).
    """
    if not records:
        return 0.0
    minority_set = set(minority_classes)
    count = sum(1 for r in records if str(r.get(field, "")) in minority_set)
    return count / len(records)


# ---------------------------------------------------------------------------
# 4. Jensen-Shannon divergence — tail distribution shift across generations
# ---------------------------------------------------------------------------

def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Symmetric Jensen-Shannon divergence in [0, 1].

    0.0 = identical distributions; 1.0 = maximally different.
    """
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(
            a[k] * math.log2(a[k] / b[k])
            for k in a
            if a.get(k, 0) > 0 and b.get(k, 0) > 0
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tail_divergence(
    records_a: Sequence[Record],
    records_b: Sequence[Record],
    field: str,
    *,
    tail_percentile: float = 0.20,
) -> float:
    """JS divergence between the tail distributions of *records_a* and *records_b*.

    Uses the tail from *records_a* as the reference; measures how much
    generation N has drifted from generation 0 in low-frequency records.
    """
    def tail_dist(records: Sequence[Record]) -> dict[str, float]:
        counts: Counter[str] = Counter(str(r.get(field, "__missing__")) for r in records)
        total = sum(counts.values())
        threshold = tail_percentile * total
        tail = {v: c for v, c in counts.items() if c <= threshold}
        t_total = sum(tail.values())
        if t_total == 0:
            return {}
        return {v: c / t_total for v, c in tail.items()}

    dist_a = tail_dist(records_a)
    dist_b = tail_dist(records_b)
    if not dist_a or not dist_b:
        return 0.0
    # Align keys: add zero for missing values (smoothed)
    keys = set(dist_a) | set(dist_b)
    eps = 1e-9
    a = {k: dist_a.get(k, eps) for k in keys}
    b = {k: dist_b.get(k, eps) for k in keys}
    # Renormalise after smoothing
    a_sum = sum(a.values())
    b_sum = sum(b.values())
    a = {k: v / a_sum for k, v in a.items()}
    b = {k: v / b_sum for k, v in b.items()}
    return _js_divergence(a, b)


# ---------------------------------------------------------------------------
# 5. Success metric
# ---------------------------------------------------------------------------

def entropy_within_tolerance(
    current_entropy: float,
    original_entropy: float,
    tolerance: float = 0.10,
) -> bool:
    """Return True if *current_entropy* has not degraded by more than *tolerance*.

    Issue #7 success metric: tail coverage entropy within 10% of the original
    distribution after five retraining iterations.  Improvement (current >
    original) always passes — the concern is collapse, not over-diversity.
    """
    if original_entropy == 0.0:
        return current_entropy >= 0.0
    # Only fail if we dropped below (1 - tolerance) * original
    return current_entropy >= (1.0 - tolerance) * original_entropy
