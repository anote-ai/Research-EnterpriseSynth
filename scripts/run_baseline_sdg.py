#!/usr/bin/env python3
"""Real SDG baseline experiments — issue #44.

Runs CTGAN, TVAE, and GaussianCopula on 3 public enterprise-proxy datasets
and measures TSTR F1, AUC-ROC, Wasserstein distance, and utility retention
vs. the real-data oracle.

Usage:
    python scripts/run_baseline_sdg.py                     # all 3 datasets
    python scripts/run_baseline_sdg.py --datasets diabetes # single dataset
    python scripts/run_baseline_sdg.py --quick             # GaussianCopula only
    python scripts/run_baseline_sdg.py --json              # JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline.datasets import LOADERS, load_dataset
from baseline.synthesizers import SYNTHESIZER_NAMES, fit_and_sample
from baseline.evaluator import evaluate


def _header(text: str) -> None:
    print()
    print("=" * 76)
    print(f"  {text}")
    print("=" * 76)


def _sep() -> None:
    print("-" * 76)


def run_dataset(
    dataset_name: str,
    synthesizers: list[str],
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    if verbose:
        _header(f"Loading dataset: {dataset_name}")

    df, spec = load_dataset(dataset_name)

    if verbose:
        print(f"  Domain   : {spec.enterprise_domain}")
        print(f"  Task     : {spec.task}")
        print(f"  Rows     : {spec.n_rows:,}")
        print(f"  Features : {spec.n_features}")
        print(f"  Target   : {spec.target_col}")
        pos_rate = df[spec.target_col].mean()
        print(f"  Positive rate: {pos_rate:.1%}  (class imbalance check)")

    from sklearn.model_selection import train_test_split
    train_df, _ = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df[spec.target_col]
    )
    n_samples = len(train_df)

    results = []
    for synth_name in synthesizers:
        if verbose:
            print(f"\n  [{synth_name}] fitting on {n_samples:,} rows...", end="", flush=True)
        t0 = time.time()
        try:
            synth_df = fit_and_sample(
                synth_name, train_df, spec.target_col, n_samples=n_samples, seed=seed
            )
            elapsed = time.time() - t0
            result = evaluate(
                real_df=df,
                synth_df=synth_df,
                target_col=spec.target_col,
                dataset_name=dataset_name,
                synthesizer_name=synth_name,
                seed=seed,
            )
            result_dict = result.as_dict()
            result_dict["elapsed_sec"] = round(elapsed, 1)
            result_dict["enterprise_domain"] = spec.enterprise_domain
            result_dict["dataset_display"] = spec.display_name
            results.append(result_dict)

            if verbose:
                print(f" done ({elapsed:.0f}s)")
                print(f"       Oracle  F1={result.oracle_f1:.3f}  AUC={result.oracle_auc:.3f}")
                print(f"       TSTR    F1={result.tstr_f1:.3f}  AUC={result.tstr_auc:.3f}  "
                      f"(retention F1={result.utility_retention_f1:.1%})")
                print(f"       Fidelity  Wasserstein={result.wasserstein_mean:.3f}  "
                      f"CorrDelta={result.correlation_delta:.3f}")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f" FAILED ({elapsed:.0f}s): {exc}")
            results.append({
                "dataset": dataset_name,
                "synthesizer": synth_name,
                "error": str(exc),
                "elapsed_sec": round(elapsed, 1),
            })

    return results


def print_summary(all_results: list[dict]) -> None:
    _header("Summary: Real SDG Baseline Results")
    ok = [r for r in all_results if "error" not in r]
    if not ok:
        print("  No successful runs.")
        return

    print(f"\n  {'Dataset':<28}  {'Synthesizer':<16}  {'Oracle F1':>9}  "
          f"{'TSTR F1':>8}  {'Retention':>9}  {'WDist':>6}  {'Time':>6}")
    _sep()

    grouped: dict[str, list[dict]] = {}
    for r in ok:
        grouped.setdefault(r["dataset"], []).append(r)

    for ds, rows in grouped.items():
        for i, r in enumerate(rows):
            display = r.get("dataset_display", ds) if i == 0 else ""
            print(
                f"  {display:<28}  {r['synthesizer']:<16}  "
                f"{r['oracle_f1']:>9.3f}  {r['tstr_f1']:>8.3f}  "
                f"{r['utility_retention_f1']:>8.1%}  {r['wasserstein_mean']:>6.3f}  "
                f"{r.get('elapsed_sec', 0):>5.0f}s"
            )
        _sep()

    # Key finding
    avg_retention = sum(r["utility_retention_f1"] for r in ok) / len(ok)
    best = max(ok, key=lambda r: r["tstr_f1"])
    worst = min(ok, key=lambda r: r["tstr_f1"])

    print(f"""
  Key findings:
    Average TSTR utility retention  : {avg_retention:.1%} of real-data oracle
    Best configuration              : {best['synthesizer']} on {best['dataset']}
                                      (F1={best['tstr_f1']:.3f}, retention={best['utility_retention_f1']:.1%})
    Most challenging                : {worst['synthesizer']} on {worst['dataset']}
                                      (F1={worst['tstr_f1']:.3f}, retention={worst['utility_retention_f1']:.1%})

  Interpretation for paper:
    These non-DP baselines establish the UTILITY CEILING — what is achievable
    without privacy constraints. DP experiments (run_pareto_study.py) are then
    expressed as percentage of this baseline, giving interpretable retention numbers.
    """)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real SDG Baseline Experiments (issue #44)")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(LOADERS.keys()),
        default=list(LOADERS.keys()),
    )
    parser.add_argument(
        "--synthesizers", nargs="+",
        choices=SYNTHESIZER_NAMES,
        default=SYNTHESIZER_NAMES,
    )
    parser.add_argument("--quick", action="store_true",
                        help="GaussianCopula only (fastest, ~30s per dataset)")
    parser.add_argument("--json", action="store_true",
                        help="Write JSON to results/baseline_sdg.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synthesizers = ["GaussianCopula"] if args.quick else args.synthesizers

    print("\nEnterpriseSynth — Real SDG Baseline Experiments (issue #44)")
    print(f"Datasets    : {', '.join(args.datasets)}")
    print(f"Synthesizers: {', '.join(synthesizers)}")

    all_results: list[dict] = []
    for ds in args.datasets:
        results = run_dataset(ds, synthesizers, seed=args.seed)
        all_results.extend(results)

    print_summary(all_results)

    if args.json:
        os.makedirs("results", exist_ok=True)
        out_path = "results/baseline_sdg.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results written to {out_path}")


if __name__ == "__main__":
    main()
