#!/usr/bin/env python3
"""Q1: ε sweep across compliance tiers — privacy-utility Pareto curve per asset type.

Sweeps ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} with multiple seeds and reports
the utility cliff (first ε where utility drops below 90% of no-DP baseline).

Usage:
    python scripts/run_epsilon_sweep.py
    python scripts/run_epsilon_sweep.py --asset-type tabular --seeds 10
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy_benchmark.config import EPSILON_VALUES, COMPLIANCE_TIERS
from privacy_benchmark.evaluator import evaluate_with_ci, evaluate_multi_seed
from privacy_benchmark.stats import epsilon_variance_profile, significance_test

# ---------------------------------------------------------------------------
# Simulated per-ε scores (replace with real SDG runs when available)
# The utility / privacy / fidelity values follow the empirical trends
# characterised in the benchmark paper (Appendix B).
# ---------------------------------------------------------------------------

def _simulated_scores(epsilon: float, seed: int) -> dict[str, float]:
    """Return simulated (auc, tstr_scores, fidelity) for a given ε and seed."""
    import random
    rng = random.Random(seed * 1000 + int(epsilon * 10))

    # Privacy: AUC decreases (more leakage) as ε increases
    # At ε=0.1 AUC ≈ 0.52 (near-random); at ε=10 AUC ≈ 0.82
    auc_base = 0.52 + 0.30 * (1 - 1 / (1 + epsilon / 2))
    auc = min(0.95, auc_base + rng.gauss(0, 0.01))

    # Utility: TSTR F1 increases with ε (more utility at looser privacy)
    # Logistic-shaped curve; utility cliff around ε ≈ 0.5
    utility_base = 0.60 + 0.38 * (1 - 1 / (1 + epsilon / 1.5))
    # More variance at small ε (DP noise)
    noise = 0.05 / (epsilon + 0.1)
    tstr_mean = min(0.99, utility_base + rng.gauss(0, noise))
    tstr_scores = [
        min(0.99, max(0.10, tstr_mean + rng.gauss(0, 0.02)))
        for _ in range(20)
    ]

    # Fidelity: similar shape to utility
    fidelity = min(0.99, 0.62 + 0.35 * (1 - 1 / (1 + epsilon / 2)) + rng.gauss(0, 0.01))

    return {"auc": auc, "tstr_scores": tstr_scores, "fidelity": fidelity}


ASSET_TYPES = ["tabular_hr", "financial_transactions", "healthcare_ehr"]

# No-DP baseline (ε → ∞) for normalisation
BASELINE_UTILITY = 0.98  # approximate TSTR F1 without DP


def run_epsilon_sweep(asset_type: str, n_seeds: int, verbose: bool) -> list[dict]:
    """Run the full ε sweep for one asset type."""
    seed_configs_by_eps: dict[float, list[dict]] = {}

    for eps in EPSILON_VALUES:
        configs = []
        for seed in range(n_seeds):
            sim = _simulated_scores(eps, seed)
            configs.append({
                "auc": sim["auc"],
                "tstr_score": sum(sim["tstr_scores"]) / len(sim["tstr_scores"]),
                "fidelity": sim["fidelity"],
            })
        seed_configs_by_eps[eps] = configs

    # Variance profile across ε
    eps_to_seed_results: dict[float, list[dict[str, float]]] = {}
    for eps, cfgs in seed_configs_by_eps.items():
        seed_results = []
        for c in cfgs:
            from privacy_benchmark.metrics import (
                compute_privacy_score,
                compute_utility_score,
                compute_fidelity_score,
            )
            seed_results.append({
                "privacy_score": compute_privacy_score(c["auc"]),
                "utility_score": compute_utility_score(c["tstr_score"]),
                "fidelity_score": compute_fidelity_score(c["fidelity"]),
            })
        eps_to_seed_results[eps] = seed_results

    variance_profile = epsilon_variance_profile(eps_to_seed_results, metric="utility_score")

    # Full CI evaluation per ε
    results = []
    for eps in EPSILON_VALUES:
        sim = _simulated_scores(eps, seed=0)
        eval_result = evaluate_with_ci(
            epsilon=eps,
            auc=sum(c["auc"] for c in seed_configs_by_eps[eps]) / n_seeds,
            tstr_scores=sim["tstr_scores"],
            fidelity=sum(c["fidelity"] for c in seed_configs_by_eps[eps]) / n_seeds,
        )
        eval_result["asset_type"] = asset_type
        eval_result["n_seeds"] = n_seeds

        # Determine compliance tier
        for tier_name, tier_epsilons in COMPLIANCE_TIERS.items():
            if eps in tier_epsilons:
                eval_result["compliance_tier"] = tier_name
                break

        # Utility cliff flag: < 90% of no-DP baseline
        utility_cliff = eval_result["utility_score"] < 0.90 * BASELINE_UTILITY
        eval_result["below_utility_cliff"] = utility_cliff

        results.append(eval_result)

    # Significance test: strict tier (ε=0.5) vs. balanced (ε=2)
    strict_scores = [_simulated_scores(0.5, s)["tstr_scores"] for s in range(min(n_seeds, 10))]
    balanced_scores = [_simulated_scores(2.0, s)["tstr_scores"] for s in range(min(n_seeds, 10))]
    strict_mean = [sum(s) / len(s) for s in strict_scores]
    balanced_mean = [sum(s) / len(s) for s in balanced_scores]

    sig = significance_test(
        balanced_mean,
        strict_mean,
        n_comparisons=len(EPSILON_VALUES) - 1,
        alpha=0.05,
    )

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Asset type: {asset_type}  (n_seeds={n_seeds})")
        print(f"{'ε':>6}  {'Tier':<20} {'Privacy':>8} {'Utility':>8} {'Utility CI':>18} {'Cliff?':>8}")
        print("-" * 70)
        for r in results:
            ci = f"[{r['tstr_ci_lower']:.3f}, {r['tstr_ci_upper']:.3f}]"
            cliff = "⚠️ YES" if r["below_utility_cliff"] else "no"
            print(
                f"  {r['epsilon']:4.1f}  {r.get('compliance_tier', '?'):<20}"
                f"  {r['privacy_score']:.3f}   {r['utility_score']:.3f}  {ci}  {cliff}"
            )

        # Find utility cliff
        cliff_eps = [r["epsilon"] for r in results if r["below_utility_cliff"]]
        if cliff_eps:
            print(f"\nUtility cliff (utility < 90% baseline) starts at ε ≤ {max(cliff_eps):.1f}")
        else:
            print("\nNo utility cliff detected (all ε values retain ≥ 90% utility)")

        print(f"\nStrict vs. balanced utility difference: {sig['test_label'] or 'n.s.'}"
              f"  (p={sig['p_value']:.4f}, Bonferroni-adjusted α={sig['p_adjusted_threshold']:.4f})")

        print(f"\nVariance profile (utility_score std across seeds):")
        for vp in variance_profile:
            print(f"  ε={vp['epsilon']:.1f}  mean={vp['mean']:.3f}  std={vp['std']:.4f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ε sweep experiment (Q1)")
    parser.add_argument("--asset-type", default="all", choices=ASSET_TYPES + ["all"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    asset_types = ASSET_TYPES if args.asset_type == "all" else [args.asset_type]
    all_results = {}

    for at in asset_types:
        results = run_epsilon_sweep(at, args.seeds, verbose=not args.json)
        all_results[at] = results

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n\nSummary: utility cliff ε by asset type")
        print("-" * 50)
        for at, results in all_results.items():
            cliff_eps = [r["epsilon"] for r in results if r["below_utility_cliff"]]
            cliff_str = f"ε ≤ {max(cliff_eps):.1f}" if cliff_eps else "none detected"
            print(f"  {at:<30}: utility cliff at {cliff_str}")


if __name__ == "__main__":
    main()
