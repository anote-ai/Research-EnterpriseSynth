"""High-level evaluator: runs all three strategies and reports collapse vs mitigation."""
from __future__ import annotations

from model_collapse.pipeline import run_model_collapse_pipeline
from model_collapse.mitigation import mitigated_pipeline_step
from model_collapse.metrics import (
    coverage_entropy,
    tail_coverage_entropy,
    minority_class_representation,
    tail_divergence,
    entropy_within_tolerance,
)


def evaluate_model_collapse(
    n_generations: int = 5,
    n_records: int = 500,
    *,
    field: str = "record_type",
    minority_classes: tuple[str, ...] = ("fraud", "critical_security"),
    collapse_rate: float = 0.30,
    tolerance: float = 0.10,
    seed: int = 42,
) -> dict:
    """Run the full collapse + mitigation evaluation.

    Returns a dict with keys:
      - ``baseline``   — list of per-generation dicts (no mitigation)
      - ``anchored``   — real-data anchoring applied each generation
      - ``diversity``  — diversity-rewarded sampling each generation
      - ``combined``   — both strategies applied
      - ``success``    — dict mapping strategy → bool (met 10% tolerance at gen N)
    """
    base_generations = run_model_collapse_pipeline(
        n_generations, n_records,
        field=field,
        minority_classes=minority_classes,
        collapse_rate=collapse_rate,
        seed=seed,
    )
    # Re-generate original records for mitigation strategies
    from model_collapse.pipeline import make_enterprise_dataset
    original = make_enterprise_dataset(n_records, field=field, seed=seed)
    original_tail_entropy = tail_coverage_entropy(original, field)

    results: dict = {
        "baseline": [g.as_dict() for g in base_generations],
        "anchored": [],
        "diversity": [],
        "combined": [],
        "success": {},
    }

    # Simulate each mitigation strategy across generations
    for strategy_key, strategy_name in [
        ("anchored", "anchor"),
        ("diversity", "diversity"),
        ("combined", "both"),
    ]:
        # Start from the collapsed data at each generation and apply mitigation
        current = list(original)
        strategy_results = []
        import random
        rng = random.Random(seed)

        # Gen 0 baseline
        strategy_results.append({
            "generation": 0,
            "tail_entropy": tail_coverage_entropy(current, field),
            "coverage_entropy": coverage_entropy(current, field),
            "minority_representation": minority_class_representation(
                current, field, minority_classes
            ),
            "tail_divergence": 0.0,
        })

        for gen in range(1, n_generations + 1):
            # Collapse the current generation
            from model_collapse.pipeline import _simulate_collapse_step
            collapsed = _simulate_collapse_step(current, field, collapse_rate, rng)
            if not collapsed:
                collapsed = [{field: "routine", "value": 0.0}]
            # Apply mitigation
            mitigated = mitigated_pipeline_step(
                collapsed, original, field,
                strategy=strategy_name,
                seed=gen,
            )
            strategy_results.append({
                "generation": gen,
                "tail_entropy": tail_coverage_entropy(mitigated, field),
                "coverage_entropy": coverage_entropy(mitigated, field),
                "minority_representation": minority_class_representation(
                    mitigated, field, minority_classes
                ),
                "tail_divergence": tail_divergence(original, mitigated, field),
            })
            current = mitigated

        results[strategy_key] = strategy_results

        # Success metric: final generation tail entropy within tolerance of original
        final_tail = strategy_results[-1]["tail_entropy"]
        results["success"][strategy_key] = entropy_within_tolerance(
            float(final_tail), original_tail_entropy, tolerance
        )

    # Baseline success check
    base_final_tail = base_generations[-1].tail_entropy
    results["success"]["baseline"] = entropy_within_tolerance(
        base_final_tail, original_tail_entropy, tolerance
    )

    return results
