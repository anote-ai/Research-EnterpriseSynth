#!/usr/bin/env python3
"""Q2: Model collapse timeline — 10-generation pipeline across collapse rates.

Tracks coverage_entropy, tail_entropy, minority_representation, and tail_divergence
at each generation and reports the generation at which each warning threshold is crossed.

Usage:
    python scripts/run_collapse_study.py
    python scripts/run_collapse_study.py --generations 10 --collapse-rate 0.30
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_collapse.pipeline import run_model_collapse_pipeline, make_enterprise_dataset
from model_collapse.metrics import (
    coverage_entropy,
    tail_coverage_entropy,
    minority_class_representation,
    entropy_within_tolerance,
)
from model_collapse.evaluator import evaluate_model_collapse

FIELD = "record_type"
MINORITY_CLASSES = ("fraud", "critical_security")

# Warning thresholds (fraction of baseline entropy that must be maintained)
THRESHOLDS = {
    "warning":  0.90,   # 10% drop — early monitoring alert
    "moderate": 0.75,   # 25% drop — compliance review triggered
    "critical": 0.50,   # 50% drop — pipeline halt recommended
}

COLLAPSE_RATES = [0.10, 0.20, 0.30, 0.40]


def run_collapse_study(
    n_generations: int,
    collapse_rate: float,
    n_records: int,
    seed: int,
    verbose: bool,
) -> dict:
    """Run one collapse study and return per-generation metrics."""
    generations = run_model_collapse_pipeline(
        n_generations=n_generations,
        n_records=n_records,
        field=FIELD,
        minority_classes=MINORITY_CLASSES,
        collapse_rate=collapse_rate,
        seed=seed,
    )

    baseline = generations[0]
    baseline_tail_entropy = baseline.tail_entropy
    baseline_coverage_entropy = baseline.coverage_entropy
    baseline_minority = baseline.minority_representation

    # Find first generation crossing each warning threshold
    crossing_gen: dict[str, int | None] = {k: None for k in THRESHOLDS}
    minority_depletion_gen: int | None = None  # first gen where minority < 0.005

    per_gen = []
    for g in generations:
        tail_pct = g.tail_entropy / baseline_tail_entropy if baseline_tail_entropy > 0 else 1.0
        cov_pct = g.coverage_entropy / baseline_coverage_entropy if baseline_coverage_entropy > 0 else 1.0
        per_gen.append({
            "generation": g.generation,
            "n_records": g.n_records,
            "tail_entropy": round(g.tail_entropy, 4),
            "tail_pct_of_baseline": round(tail_pct, 4),
            "coverage_entropy": round(g.coverage_entropy, 4),
            "coverage_pct_of_baseline": round(cov_pct, 4),
            "minority_representation": round(g.minority_representation, 5),
            "tail_divergence": round(g.tail_divergence, 4),
        })
        for threshold_name, threshold_val in THRESHOLDS.items():
            if crossing_gen[threshold_name] is None and tail_pct < threshold_val:
                crossing_gen[threshold_name] = g.generation

        if minority_depletion_gen is None and g.minority_representation < 0.005:
            minority_depletion_gen = g.generation

    if verbose:
        print(f"\n  collapse_rate={collapse_rate:.2f}  n_generations={n_generations}  n_records={n_records}")
        print(f"  {'Gen':>4}  {'TailH':>7}  {'TailH%':>7}  {'CovH':>7}  {'Minority%':>10}  {'JSD':>7}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*7}")
        for pg in per_gen:
            minority_pct = pg["minority_representation"] * 100
            warning = ""
            if pg["tail_pct_of_baseline"] < THRESHOLDS["critical"]:
                warning = " ❌ CRITICAL"
            elif pg["tail_pct_of_baseline"] < THRESHOLDS["moderate"]:
                warning = " ⚠ moderate"
            elif pg["tail_pct_of_baseline"] < THRESHOLDS["warning"]:
                warning = " • warning"
            print(
                f"  {pg['generation']:>4}  {pg['tail_entropy']:>7.4f}  "
                f"{pg['tail_pct_of_baseline']:>6.1%}  {pg['coverage_entropy']:>7.4f}  "
                f"{minority_pct:>9.2f}%  {pg['tail_divergence']:>7.4f}{warning}"
            )

        print(f"\n  Threshold crossings (tail entropy):")
        for t, gen in crossing_gen.items():
            gen_str = f"gen {gen}" if gen is not None else f"not reached in {n_generations} gens"
            print(f"    {t} ({THRESHOLDS[t]:.0%}): {gen_str}")
        dep_str = (
            f"gen {minority_depletion_gen}" if minority_depletion_gen is not None
            else f"not reached in {n_generations} gens"
        )
        print(f"    fraud/critical_security < 0.5%: {dep_str}")

    return {
        "collapse_rate": collapse_rate,
        "n_generations": n_generations,
        "n_records": n_records,
        "baseline_tail_entropy": round(baseline_tail_entropy, 4),
        "baseline_minority_representation": round(baseline_minority, 5),
        "per_generation": per_gen,
        "threshold_crossings": {k: v for k, v in crossing_gen.items()},
        "minority_depletion_generation": minority_depletion_gen,
    }


def run_mitigation_comparison(n_generations: int, collapse_rate: float) -> None:
    """Print the mitigation comparison table (from evaluator.py)."""
    print(f"\n  === Mitigation comparison (collapse_rate={collapse_rate}) ===")
    result = evaluate_model_collapse(
        n_generations=n_generations,
        n_records=400,
        field=FIELD,
        minority_classes=MINORITY_CLASSES,
        collapse_rate=collapse_rate,
        tolerance=0.10,
        seed=42,
    )
    print(f"  {'Strategy':<25} {'Final tail H':>13} {'vs. baseline':>14} {'Passes?':>8}")
    print(f"  {'-'*25}  {'-'*13}  {'-'*14}  {'-'*8}")

    original_results = run_model_collapse_pipeline(
        n_generations=0, n_records=400, field=FIELD,
        minority_classes=MINORITY_CLASSES, collapse_rate=collapse_rate, seed=42,
    )
    baseline_h = original_results[0].tail_entropy

    for strategy in ("baseline", "anchored", "diversity", "combined"):
        final = result[strategy][-1]["tail_entropy"]
        change_pct = (final - baseline_h) / baseline_h if baseline_h > 0 else 0.0
        passes = result["success"].get(strategy, False)
        sign = "+" if change_pct >= 0 else ""
        print(
            f"  {strategy:<25}  {final:>13.4f}  {sign}{change_pct:>13.1%}  {'✅' if passes else '❌':>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Model collapse study (Q2)")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--collapse-rate", type=float, default=None,
                        help="Single rate to study (default: sweep all rates)")
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mitigation", action="store_true",
                        help="Also run mitigation comparison")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rates = [args.collapse_rate] if args.collapse_rate is not None else COLLAPSE_RATES
    all_results = {}

    if not args.json:
        print(f"Model Collapse Study — {args.generations} generations")
        print(f"n_records={args.records}  minority={MINORITY_CLASSES}")

    for rate in rates:
        result = run_collapse_study(
            args.generations, rate, args.records, args.seed, verbose=not args.json
        )
        all_results[str(rate)] = result

        if args.mitigation and not args.json:
            n_gen_mit = min(5, args.generations)
            run_mitigation_comparison(n_gen_mit, rate)

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n\nCollapse timeline summary")
        print("-" * 65)
        print(f"  {'Rate':>6}  {'Warning (90%)':>14}  {'Moderate (75%)':>16}  {'Critical (50%)':>16}  {'Minority depleted':>18}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*16}  {'-'*18}")
        for rate, r in all_results.items():
            tc = r["threshold_crossings"]
            gen_str = lambda g: f"gen {g}" if g is not None else f">{args.generations}"
            dep = r["minority_depletion_generation"]
            print(
                f"  {float(rate):>6.2f}  {gen_str(tc.get('warning')):>14}  "
                f"{gen_str(tc.get('moderate')):>16}  {gen_str(tc.get('critical')):>16}  "
                f"{gen_str(dep):>18}"
            )


if __name__ == "__main__":
    main()
