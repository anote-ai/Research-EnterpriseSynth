"""Mitigation strategies to counter model collapse in iterative pipelines.

Strategies:
  1. Real-data anchoring   — periodically mix original real records back in
  2. Diversity-rewarded sampling — up-weight under-represented tail records
"""
from __future__ import annotations

import random
from collections import Counter

from model_collapse.metrics import Record


def real_data_anchoring(
    synthetic_records: list[Record],
    real_records: list[Record],
    field: str,
    *,
    anchor_ratio: float = 0.20,
    seed: int = 0,
) -> list[Record]:
    """Mix *anchor_ratio* fraction of *real_records* into *synthetic_records*.

    Ensures that minority records from the real distribution are always
    represented, preventing tail collapse.  Returns a new combined list.
    """
    if not real_records:
        return list(synthetic_records)
    rng = random.Random(seed)
    n_anchor = max(1, int(len(synthetic_records) * anchor_ratio))
    anchors = rng.sample(real_records, min(n_anchor, len(real_records)))
    return list(synthetic_records) + anchors


def diversity_rewarded_sampling(
    records: list[Record],
    field: str,
    *,
    reward_strength: float = 3.0,
    target_n: int | None = None,
    seed: int = 0,
) -> list[Record]:
    """Resample *records* with weights inversely proportional to value frequency.

    Rare values are up-weighted by *reward_strength* relative to common ones,
    encouraging the next generation to cover the tail.  Returns a resampled
    list of the same size (or *target_n* if specified).
    """
    if not records:
        return []
    rng = random.Random(seed)
    target = target_n if target_n is not None else len(records)
    counts: Counter[str] = Counter(str(r.get(field, "")) for r in records)
    max_count = max(counts.values())

    def weight(rec: Record) -> float:
        freq = counts[str(rec.get(field, ""))]
        # Infrequent values get higher weight
        return 1.0 + reward_strength * (1.0 - freq / max_count)

    weights = [weight(r) for r in records]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Weighted sampling with replacement
    chosen: list[Record] = []
    cumulative = []
    acc = 0.0
    for p in probs:
        acc += p
        cumulative.append(acc)

    for _ in range(target):
        r = rng.random()
        # Binary search
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        chosen.append(records[lo])
    return chosen


def mitigated_pipeline_step(
    collapsed_records: list[Record],
    original_records: list[Record],
    field: str,
    *,
    strategy: str = "both",
    anchor_ratio: float = 0.20,
    reward_strength: float = 3.0,
    seed: int = 0,
) -> list[Record]:
    """Apply mitigation strategy to one generation of collapsed data.

    *strategy* is one of:
      - ``"anchor"``    — real-data anchoring only
      - ``"diversity"`` — diversity-rewarded sampling only
      - ``"both"``      — diversity sampling first, then anchor injection
    """
    result = list(collapsed_records)
    if strategy in ("diversity", "both"):
        result = diversity_rewarded_sampling(
            result, field, reward_strength=reward_strength, seed=seed
        )
    if strategy in ("anchor", "both"):
        result = real_data_anchoring(
            result, original_records, field,
            anchor_ratio=anchor_ratio, seed=seed
        )
    return result
