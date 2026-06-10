#!/usr/bin/env python3
"""Cross-vendor comparison harness for EnterpriseSynth.

Vendors submit a JSON results file; this script evaluates using the
EnterpriseSynth standardized scoring and produces a comparison table.

Usage:
    # Run with bundled example vendor results (for demo/CI purposes):
    python scripts/run_vendor_comparison.py

    # Load real vendor-submitted results:
    python scripts/run_vendor_comparison.py --results-dir path/to/vendor_results/

    # Print JSON for downstream processing:
    python scripts/run_vendor_comparison.py --json
"""
import argparse
import sys
import os
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy_benchmark.evaluator import evaluate_configuration
from privacy_benchmark.config import COMPLIANCE_TIERS

# ---------------------------------------------------------------------------
# Vendor results schema
# ---------------------------------------------------------------------------
# A vendor submits a JSON file per (tool, asset_type, epsilon) combination:
#
# {
#   "vendor_name": "Gretel AI",
#   "model": "ACTGAN",
#   "asset_type": "tabular_hr",
#   "epsilon": 1.0,
#   "verified": false,           # true only if independently verified
#   "results": [
#     {"metric": "mia_auc",                   "value": 0.612},
#     {"metric": "tstr_f1",                   "value": 0.871},
#     {"metric": "constraint_violation_rate", "value": 0.031},
#     {"metric": "fidelity",                  "value": 0.847}
#   ]
# }

# ---------------------------------------------------------------------------
# Bundled example results (simulated; to be replaced by real vendor submissions)
# ---------------------------------------------------------------------------

EXAMPLE_VENDOR_RESULTS = [
    # EnterpriseSynth reference implementation
    {"vendor_name": "EnterpriseSynth (Reference)", "model": "DP-CTGAN",
     "asset_type": "tabular_hr", "epsilon": 1.0, "verified": True,
     "results": [
         {"metric": "mia_auc", "value": 0.614},
         {"metric": "tstr_f1", "value": 0.880},
         {"metric": "constraint_violation_rate", "value": 0.018},
         {"metric": "fidelity", "value": 0.850},
     ]},
    {"vendor_name": "EnterpriseSynth (Reference)", "model": "DP-CTGAN",
     "asset_type": "tabular_hr", "epsilon": 5.0, "verified": True,
     "results": [
         {"metric": "mia_auc", "value": 0.733},
         {"metric": "tstr_f1", "value": 0.960},
         {"metric": "constraint_violation_rate", "value": 0.008},
         {"metric": "fidelity", "value": 0.930},
     ]},
    # Simulated Gretel AI results (placeholder for real vendor submission)
    {"vendor_name": "Gretel AI", "model": "ACTGAN", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 1.0,
     "results": [
         {"metric": "mia_auc", "value": 0.628},
         {"metric": "tstr_f1", "value": 0.865},
         {"metric": "constraint_violation_rate", "value": 0.024},
         {"metric": "fidelity", "value": 0.841},
     ]},
    {"vendor_name": "Gretel AI", "model": "ACTGAN", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 5.0,
     "results": [
         {"metric": "mia_auc", "value": 0.748},
         {"metric": "tstr_f1", "value": 0.952},
         {"metric": "constraint_violation_rate", "value": 0.011},
         {"metric": "fidelity", "value": 0.921},
     ]},
    # Simulated Mostly AI results
    {"vendor_name": "Mostly AI", "model": "MOSA", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 1.0,
     "results": [
         {"metric": "mia_auc", "value": 0.607},
         {"metric": "tstr_f1", "value": 0.871},
         {"metric": "constraint_violation_rate", "value": 0.029},
         {"metric": "fidelity", "value": 0.843},
     ]},
    {"vendor_name": "Mostly AI", "model": "MOSA", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 5.0,
     "results": [
         {"metric": "mia_auc", "value": 0.726},
         {"metric": "tstr_f1", "value": 0.947},
         {"metric": "constraint_violation_rate", "value": 0.016},
         {"metric": "fidelity", "value": 0.918},
     ]},
    # Simulated SDV (Synthetic Data Vault) — open-source baseline
    {"vendor_name": "SDV (CTGAN)", "model": "CTGAN", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 1.0,
     "results": [
         {"metric": "mia_auc", "value": 0.635},
         {"metric": "tstr_f1", "value": 0.858},
         {"metric": "constraint_violation_rate", "value": 0.038},
         {"metric": "fidelity", "value": 0.831},
     ]},
    {"vendor_name": "SDV (CTGAN)", "model": "CTGAN", "verified": False,
     "asset_type": "tabular_hr", "epsilon": 5.0,
     "results": [
         {"metric": "mia_auc", "value": 0.741},
         {"metric": "tstr_f1", "value": 0.941},
         {"metric": "constraint_violation_rate", "value": 0.021},
         {"metric": "fidelity", "value": 0.913},
     ]},
]


# ---------------------------------------------------------------------------
# Evaluation and ranking
# ---------------------------------------------------------------------------

def parse_vendor_results(vendor_data: dict) -> dict:
    """Convert vendor submission to EnterpriseSynth evaluated scores."""
    metrics = {r["metric"]: r["value"] for r in vendor_data["results"]}

    mia_auc = metrics.get("mia_auc", 0.5)
    tstr_f1 = metrics.get("tstr_f1", 0.0)
    fidelity = metrics.get("fidelity", 0.0)
    cv_rate = metrics.get("constraint_violation_rate", 0.0)

    evaluated = evaluate_configuration(
        epsilon=vendor_data["epsilon"],
        auc=mia_auc,
        tstr_score=tstr_f1,
        fidelity=fidelity,
    )
    evaluated["vendor_name"] = vendor_data["vendor_name"]
    evaluated["model"] = vendor_data.get("model", "?")
    evaluated["asset_type"] = vendor_data["asset_type"]
    evaluated["verified"] = vendor_data.get("verified", False)
    evaluated["constraint_violation_rate"] = cv_rate

    tier = next(
        (t for t, eps_list in COMPLIANCE_TIERS.items()
         if vendor_data["epsilon"] in eps_list),
        "?"
    )
    evaluated["tier"] = tier

    # Composite score: harmonic mean of privacy, utility, fidelity
    scores = [evaluated["privacy_score"], evaluated["utility_score"], evaluated["fidelity_score"]]
    evaluated["composite_score"] = len(scores) / sum(1 / s for s in scores) if all(s > 0 for s in scores) else 0.0

    return evaluated


def load_vendor_results(results_dir: str | None) -> list[dict]:
    """Load vendor results from directory of JSON files, or use bundled examples."""
    if results_dir and os.path.isdir(results_dir):
        all_data = []
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    all_data.extend(json.load(f) if isinstance(json.load(open(os.path.join(results_dir, fname))), list) else [json.load(f)])
        return all_data
    return EXAMPLE_VENDOR_RESULTS


def print_comparison_table(results: list[dict], epsilon: float | None) -> None:
    filtered = results if epsilon is None else [r for r in results if r["epsilon"] == epsilon]
    if not filtered:
        print("No results for the specified filters.")
        return

    eps_values = sorted({r["epsilon"] for r in filtered})
    for eps in eps_values:
        eps_results = sorted(
            [r for r in filtered if r["epsilon"] == eps],
            key=lambda x: x["composite_score"],
            reverse=True,
        )
        print(f"\n  ε = {eps}  (tier: {eps_results[0].get('tier', '?')})")
        print(f"  {'Vendor':<28}  {'Model':<12}  {'Privacy':>8}  {'Utility':>8}  {'Fidelity':>9}  {'Composite':>10}  {'Verified':>9}")
        print(f"  {'-'*28}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*9}")
        for r in eps_results:
            rank_marker = "🥇" if r == eps_results[0] else "  "
            verified_str = "✅ Yes" if r["verified"] else "   Self-reported"
            print(
                f"  {rank_marker}{r['vendor_name']:<26}  {r['model']:<12}  "
                f"{r['privacy_score']:>8.4f}  {r['utility_score']:>8.4f}  "
                f"{r['fidelity_score']:>9.4f}  {r['composite_score']:>10.4f}  {verified_str}"
            )

        print(f"\n  Constraint violation rate comparison:")
        for r in eps_results:
            bar = "█" * int(r["constraint_violation_rate"] * 200)
            flag = " ⚠" if r["constraint_violation_rate"] > 0.05 else "  "
            print(f"    {r['vendor_name']:<28}: {r['constraint_violation_rate']:.4f}{flag}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-vendor comparison")
    parser.add_argument("--results-dir", default=None,
                        help="Directory of vendor JSON result files")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    raw_results = load_vendor_results(args.results_dir)
    evaluated = [parse_vendor_results(r) for r in raw_results]

    if args.json:
        print(json.dumps(evaluated, indent=2))
        return

    print("EnterpriseSynth Cross-Vendor Comparison")
    print("Note: results marked 'Self-reported' are vendor-submitted and unverified.")
    print("Only 'Verified' results are independently confirmed by EnterpriseSynth.\n")

    asset_types = sorted({r["asset_type"] for r in evaluated})
    for asset in asset_types:
        print(f"\n{'=' * 80}")
        print(f"  Asset type: {asset.replace('_', ' ').title()}")
        print(f"{'=' * 80}")
        asset_results = [r for r in evaluated if r["asset_type"] == asset]
        print_comparison_table(asset_results, args.epsilon)

    print("\n\nBadge eligibility:")
    print("-" * 55)
    vendors_by_name: dict[str, list[dict]] = {}
    for r in evaluated:
        vendors_by_name.setdefault(r["vendor_name"], []).append(r)
    for vendor, results_list in sorted(vendors_by_name.items()):
        n_assets = len({r["asset_type"] for r in results_list})
        all_reported = all(not r["verified"] for r in results_list if r["vendor_name"] != "EnterpriseSynth (Reference)")
        is_pareto_leader = any(r["composite_score"] >= 0.80 for r in results_list)

        if n_assets >= 2 and not all_reported:
            badge = "★★★ Pareto Leader" if is_pareto_leader else "★★  Transparent Evaluator"
        elif n_assets >= 2:
            badge = "★   Benchmark Participant (self-reported)"
        else:
            badge = "    Not yet eligible (need ≥ 2 asset types)"
        print(f"  {vendor:<30}: {badge}")


if __name__ == "__main__":
    main()
