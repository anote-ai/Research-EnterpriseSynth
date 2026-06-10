#!/usr/bin/env python3
"""Anote Synthetic-Data product audit — runs all four investigations from issue #14.

Simulates product-generated data using the benchmark infrastructure and reports
pass/fail for each investigation. Replace the _simulate_product_output() calls
with actual Anote Synthetic-Data API calls when running against the live product.

Usage:
    python scripts/run_product_audit.py
    python scripts/run_product_audit.py --investigation 1   # constraint violations only
    python scripts/run_product_audit.py --epsilon 2.0 --json
"""
import argparse
import sys
import os
import math
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from consistency.evaluator import evaluate_dataset
from consistency.schemas import HR_SCHEMA
from privacy_benchmark.config import COMPLIANCE_TIERS
from privacy_benchmark.evaluator import evaluate_configuration
from tstr_eval.evaluator import evaluate_document_category
from model_collapse.metrics import (
    tail_coverage_entropy,
    minority_class_representation,
    entropy_within_tolerance,
)
from model_collapse.pipeline import make_enterprise_dataset, _simulate_collapse_step
from model_collapse.mitigation import mitigated_pipeline_step

# ---------------------------------------------------------------------------
# Thresholds (from paper findings)
# ---------------------------------------------------------------------------

CONSTRAINT_VIOLATION_THRESHOLD = 0.02   # 2% — P0 fix if exceeded
MIA_AUC_THRESHOLD_HIPAA = 0.67          # HIPAA balanced tier
MIA_AUC_THRESHOLD_STRICT = 0.55         # GDPR strict tier
TSTR_GAP_THRESHOLD_CLASSIFICATION = 0.05   # 5% gap → flag
TSTR_GAP_THRESHOLD_ANOMALY = 0.15           # 15% gap → flag (anomaly is harder)
COLLAPSE_TAIL_THRESHOLD = 0.20           # 20% drop after 3 iterations → flag


# ---------------------------------------------------------------------------
# Simulate product output
# Replace these functions with real Anote API calls.
# ---------------------------------------------------------------------------

def _simulate_product_hr_rows(n: int = 1000, violation_rate: float = 0.03, seed: int = 42) -> list[dict]:
    """Simulate Anote-generated HR rows with a realistic violation rate."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        birth_year = rng.randint(1960, 1998)
        hire_year = birth_year + rng.randint(22, 32)
        has_term = rng.random() < 0.20
        term_year = hire_year + rng.randint(1, 12) if has_term else None
        age = 2026 - birth_year
        salary = max(0, rng.gauss(80000, 20000))
        row: dict = {
            "hire_date": f"{hire_year}-06-15",
            "salary": round(salary, 2),
            "birth_year": birth_year,
            "age": age,
        }
        if term_year:
            row["termination_date"] = f"{term_year}-12-31"
        # Inject violations at the specified rate
        if rng.random() < violation_rate:
            violation_type = rng.choice(["date_order", "salary_negative", "age_mismatch"])
            if violation_type == "date_order" and "termination_date" in row:
                row["hire_date"] = f"{hire_year + 20}-01-01"  # hire after termination
            elif violation_type == "salary_negative":
                row["salary"] = -abs(row["salary"])
            elif violation_type == "age_mismatch":
                row["age"] = age + rng.randint(5, 15)  # age inconsistent with birth_year
        rows.append(row)
    return rows


def _simulate_mia_auc(epsilon: float) -> float:
    """Simulate MIA AUC for a given ε (from benchmark Appendix B)."""
    # Realistic curve: AUC rises as ε increases (weaker privacy)
    base = 0.52 + 0.30 * (1 - 1 / (1 + epsilon / 2))
    return min(0.95, base + random.gauss(0, 0.01))


def _simulate_tstr_scores(epsilon: float, task: str) -> tuple[float, float]:
    """Return (tstr_score, real_baseline_score) for a given task and ε."""
    rng = random.Random(int(epsilon * 100))
    baselines = {"classification": 0.92, "anomaly_detection": 0.78, "churn_prediction": 0.85}
    baseline = baselines.get(task, 0.88)
    # TSTR gap increases at lower ε; anomaly degrades fastest
    degradation = {"classification": 0.08, "anomaly_detection": 0.18, "churn_prediction": 0.10}
    deg = degradation.get(task, 0.10)
    scale = deg * (1 - math.log1p(epsilon) / math.log1p(10))
    tstr = max(0.40, baseline - scale + rng.gauss(0, 0.01))
    return round(tstr, 4), baseline


# ---------------------------------------------------------------------------
# Investigation 1: Constraint Violation Rate
# ---------------------------------------------------------------------------

SCHEMA_CONFIGS = {
    "hr_records":             {"schema": HR_SCHEMA, "expected_violation_rate": 0.03},
    "financial_transactions": {"schema": None,      "expected_violation_rate": 0.01},
    "customer_crm":           {"schema": None,      "expected_violation_rate": 0.02},
    "inventory":              {"schema": None,      "expected_violation_rate": 0.01},
    "event_logs":             {"schema": None,      "expected_violation_rate": 0.015},
}


def run_investigation_1(n_rows: int, verbose: bool) -> dict:
    """Measure constraint violation rate across 5 enterprise schemas."""
    results = {}

    for schema_name, config in SCHEMA_CONFIGS.items():
        schema = config["schema"]
        sim_rate = config["expected_violation_rate"]

        if schema is None:
            # Schemas not yet wired up — report estimated rate from domain expert review
            violation_rate = sim_rate
            source = "estimated"
        else:
            rows = _simulate_product_hr_rows(n_rows, violation_rate=sim_rate)
            eval_result = evaluate_dataset(rows, schema)
            violation_rate = eval_result["violation_rate"]
            source = "measured"

        status = "PASS" if violation_rate <= CONSTRAINT_VIOLATION_THRESHOLD else "FAIL"
        results[schema_name] = {
            "violation_rate": round(violation_rate, 4),
            "threshold": CONSTRAINT_VIOLATION_THRESHOLD,
            "status": status,
            "source": source,
        }

    if verbose:
        print("\n" + "=" * 65)
        print("Investigation 1: Constraint Violation Rate")
        print(f"Threshold: < {CONSTRAINT_VIOLATION_THRESHOLD:.0%} per schema")
        print("=" * 65)
        print(f"  {'Schema':<28}  {'Violation Rate':>14}  {'Source':>10}  {'Status':>6}")
        print(f"  {'-'*28}  {'-'*14}  {'-'*10}  {'-'*6}")
        for schema_name, r in results.items():
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(
                f"  {schema_name:<28}  {r['violation_rate']:>13.2%}  "
                f"{r['source']:>10}  {icon} {r['status']}"
            )

        failing = [s for s, r in results.items() if r["status"] == "FAIL"]
        if failing:
            print(f"\n  ❌ P0 action required for: {', '.join(failing)}")
            print("  Recommendation: add post-generation constraint validator")
            print("  Implementation: src/consistency/evaluator.py + src/consistency/rules.py")
        else:
            print("\n  ✅ All schemas pass the constraint violation threshold")

    return results


# ---------------------------------------------------------------------------
# Investigation 2: DP Implementation Audit
# ---------------------------------------------------------------------------

PRODUCT_EPSILON_CONTEXTS = {
    "healthcare_hipaa":      {"epsilon": 1.5,  "tier": "balanced",        "threshold": MIA_AUC_THRESHOLD_HIPAA},
    "finance_sox":           {"epsilon": 7.0,  "tier": "utility_focused", "threshold": 0.82},
    "gdpr_research":         {"epsilon": 0.5,  "tier": "strict",          "threshold": MIA_AUC_THRESHOLD_STRICT},
    "internal_analytics":    {"epsilon": 10.0, "tier": "utility_focused", "threshold": 0.90},
}


def run_investigation_2(verbose: bool) -> dict:
    """Audit DP implementation: compare observed MIA AUC vs. benchmark thresholds."""
    results = {}

    for context_name, config in PRODUCT_EPSILON_CONTEXTS.items():
        eps = config["epsilon"]
        threshold = config["threshold"]
        mia_auc = _simulate_mia_auc(eps)
        status = "PASS" if mia_auc <= threshold else "FAIL"

        # Recommended ε from benchmark
        tier_epsilons = COMPLIANCE_TIERS.get(config["tier"], [eps])
        recommended_eps = min(tier_epsilons)

        evaluated = evaluate_configuration(
            epsilon=eps,
            auc=mia_auc,
            tstr_score=0.88,  # placeholder
            fidelity=0.85,
        )

        results[context_name] = {
            "epsilon": eps,
            "mia_auc": round(mia_auc, 4),
            "privacy_score": round(evaluated["privacy_score"], 4),
            "auc_threshold": threshold,
            "tier": config["tier"],
            "recommended_epsilon": recommended_eps,
            "status": status,
        }

    if verbose:
        print("\n" + "=" * 75)
        print("Investigation 2: DP Implementation Audit")
        print("=" * 75)
        print(f"  {'Context':<24}  {'ε':>5}  {'MIA AUC':>8}  {'Threshold':>10}  {'Privacy':>8}  {'Status':>6}")
        print(f"  {'-'*24}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")
        for context_name, r in results.items():
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(
                f"  {context_name:<24}  {r['epsilon']:>5.1f}  {r['mia_auc']:>8.4f}  "
                f"{r['auc_threshold']:>10.2f}  {r['privacy_score']:>8.4f}  {icon} {r['status']}"
            )

        failing = {c: r for c, r in results.items() if r["status"] == "FAIL"}
        if failing:
            print(f"\n  ❌ P0 action required:")
            for ctx, r in failing.items():
                print(f"    {ctx}: AUC {r['mia_auc']:.4f} > threshold {r['auc_threshold']:.2f}")
                print(f"    Recommendation: lower ε to {r['recommended_epsilon']} for {r['tier']} tier")
                print(f"    UI update: show 'Recommended: ε = {r['recommended_epsilon']}' for {ctx} context")
        else:
            print("\n  ✅ All DP contexts pass MIA AUC threshold")
            print("  Recommendation: update UI to show ε recommendations from benchmark")

    return results


# ---------------------------------------------------------------------------
# Investigation 3: TSTR Utility Measurement
# ---------------------------------------------------------------------------

CUSTOMER_TASKS = {
    "fraud_detection":   {"epsilon": 2.0, "task": "classification",   "gap_threshold": TSTR_GAP_THRESHOLD_CLASSIFICATION},
    "anomaly_detection": {"epsilon": 2.0, "task": "anomaly_detection", "gap_threshold": TSTR_GAP_THRESHOLD_ANOMALY},
    "churn_prediction":  {"epsilon": 2.0, "task": "churn_prediction",  "gap_threshold": TSTR_GAP_THRESHOLD_CLASSIFICATION},
}


def run_investigation_3(epsilon: float | None, verbose: bool) -> dict:
    """Measure TSTR utility gap for 3 representative customer tasks."""
    results = {}

    for task_name, config in CUSTOMER_TASKS.items():
        eps = epsilon if epsilon is not None else config["epsilon"]
        tstr_score, real_baseline = _simulate_tstr_scores(eps, config["task"])
        tstr_gap = (real_baseline - tstr_score) / real_baseline
        threshold = config["gap_threshold"]
        status = "PASS" if tstr_gap <= threshold else "FAIL"

        results[task_name] = {
            "epsilon": eps,
            "tstr_score": tstr_score,
            "real_baseline": real_baseline,
            "tstr_gap": round(tstr_gap, 4),
            "gap_threshold": threshold,
            "status": status,
        }

    if verbose:
        print("\n" + "=" * 70)
        print("Investigation 3: TSTR Utility Measurement")
        print(f"Threshold: < {TSTR_GAP_THRESHOLD_CLASSIFICATION:.0%} gap for classification; "
              f"< {TSTR_GAP_THRESHOLD_ANOMALY:.0%} for anomaly detection")
        print("=" * 70)
        print(f"  {'Task':<22}  {'ε':>5}  {'Real base':>10}  {'TSTR':>7}  {'Gap':>8}  {'Threshold':>10}  {'Status':>6}")
        print(f"  {'-'*22}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*6}")
        for task_name, r in results.items():
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(
                f"  {task_name:<22}  {r['epsilon']:>5.1f}  {r['real_baseline']:>10.4f}  "
                f"{r['tstr_score']:>7.4f}  {r['tstr_gap']:>7.1%}  "
                f"{r['gap_threshold']:>9.0%}  {icon} {r['status']}"
            )

        failing = [t for t, r in results.items() if r["status"] == "FAIL"]
        if failing:
            print(f"\n  ❌ P1 action required for: {', '.join(failing)}")
            print("  Recommendation: add TSTR quality gate to product export pipeline")
            print("  For anomaly detection: recommend ε ≥ 5 in product UI for this use case")
        else:
            print("\n  ✅ All customer tasks pass TSTR utility threshold")
            print("  Recommendation: add TSTR score to product quality badge")

    return results


# ---------------------------------------------------------------------------
# Investigation 4: Model Collapse Detection
# ---------------------------------------------------------------------------

def _run_iterative_pipeline(n_records: int, n_iterations: int, drift_rate: float, seed: int) -> list[dict]:
    """Simulate 5-iteration synthetic retraining pipeline."""
    rng = random.Random(seed)
    field = "record_type"
    original = make_enterprise_dataset(n_records, field=field, seed=seed)
    current = list(original)
    generations = []

    original_tail_h = tail_coverage_entropy(original, field)
    original_minority = minority_class_representation(
        original, field, ["fraud", "critical_security"]
    )

    for gen in range(n_iterations + 1):
        tail_h = tail_coverage_entropy(current, field)
        minority = minority_class_representation(current, field, ["fraud", "critical_security"])
        generations.append({
            "generation": gen,
            "tail_entropy": round(tail_h, 4),
            "tail_pct": round(tail_h / original_tail_h, 4) if original_tail_h > 0 else 1.0,
            "minority_representation": round(minority, 5),
            "within_20pct_tolerance": entropy_within_tolerance(tail_h, original_tail_h, 0.20),
        })
        if gen < n_iterations:
            current = _simulate_collapse_step(current, field, drift_rate, rng)
            if not current:
                current = [{field: "routine", "value": 0.0}]

    return generations


def run_investigation_4(n_records: int, n_iterations: int, drift_rate: float, verbose: bool) -> dict:
    """Test for model collapse in iterative workflows."""
    generations = _run_iterative_pipeline(n_records, n_iterations, drift_rate, seed=42)

    # Find first generation exceeding the 20% tail drop threshold
    collapse_gen = next(
        (g["generation"] for g in generations if not g["within_20pct_tolerance"]),
        None
    )
    status = "PASS" if collapse_gen is None else "FAIL"
    collapse_at_gen3 = not generations[min(3, n_iterations)]["within_20pct_tolerance"]

    if verbose:
        print("\n" + "=" * 65)
        print(f"Investigation 4: Model Collapse in Iterative Workflows")
        print(f"n_records={n_records}  n_iterations={n_iterations}  drift_rate={drift_rate:.0%}")
        print(f"Threshold: tail entropy must stay within 20% of baseline")
        print("=" * 65)
        print(f"  {'Gen':>4}  {'Tail H':>8}  {'vs. baseline':>13}  {'Minority%':>10}  {'Within 20%?':>12}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*13}  {'-'*10}  {'-'*12}")
        for g in generations:
            within = "✅ Yes" if g["within_20pct_tolerance"] else "❌ NO"
            print(
                f"  {g['generation']:>4}  {g['tail_entropy']:>8.4f}  "
                f"{g['tail_pct']:>12.1%}  "
                f"{g['minority_representation']*100:>9.2f}%  {within}"
            )

        if collapse_gen is not None:
            print(f"\n  ❌ P1 action required: collapse detected at generation {collapse_gen}")
            print(f"  Customers using iterative workflows at {drift_rate:.0%} drift rate")
            print(f"  will experience >20% tail degradation by generation {collapse_gen}")
            print(f"  Recommendation:")
            print(f"    1. Add tail entropy monitor to iterative workflow UI")
            print(f"    2. Add 're-anchor with real data' option (anchor_ratio=0.20)")
            print(f"    3. Show 'Data Diversity Health' indicator (Green/Yellow/Red)")
        else:
            print(f"\n  ✅ No collapse detected across {n_iterations} iterations at {drift_rate:.0%} drift")

    return {
        "drift_rate": drift_rate,
        "n_iterations": n_iterations,
        "collapse_generation": collapse_gen,
        "status": status,
        "collapse_at_generation_3": collapse_at_gen3,
        "per_generation": generations,
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(inv1, inv2, inv3, inv4) -> None:
    print("\n" + "=" * 65)
    print("PRODUCT AUDIT SUMMARY")
    print("=" * 65)

    inv1_status = "PASS" if all(r["status"] == "PASS" for r in inv1.values()) else "FAIL"
    inv2_status = "PASS" if all(r["status"] == "PASS" for r in inv2.values()) else "FAIL"
    inv3_status = "PASS" if all(r["status"] == "PASS" for r in inv3.values()) else "FAIL"
    inv4_status = inv4["status"]

    rows = [
        ("1", "Constraint violation rate",          inv1_status, "P0 — constraint enforcement needed"),
        ("2", "DP implementation (MIA AUC)",         inv2_status, "P0 — update ε calibration in UI"),
        ("3", "TSTR utility gap",                   inv3_status, "P1 — add quality gate to export"),
        ("4", "Model collapse in iterative workflows", inv4_status, "P1 — add collapse monitor + re-anchor"),
    ]

    print(f"\n  {'#':>2}  {'Investigation':<38}  {'Result':>8}  {'Action if FAIL'}")
    print(f"  {'-'*2}  {'-'*38}  {'-'*8}  {'-'*35}")
    for num, name, status, action in rows:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {num:>2}  {name:<38}  {icon} {status:>5}  {action if status == 'FAIL' else '—'}")

    all_pass = all(s == "PASS" for _, _, s, _ in rows)
    print()
    if all_pass:
        print("  ✅ All investigations pass. Product meets benchmark thresholds.")
        print("  Next: add benchmark metrics to product quality badge (TSTR, privacy score).")
    else:
        failing = [n for _, n, s, _ in rows if s == "FAIL"]
        print(f"  ❌ {len(failing)} investigation(s) failed. See details above.")
        print("  File product tickets for P0 issues before next release.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Anote Synthetic-Data product audit (issue #14)")
    parser.add_argument("--investigation", type=int, choices=[1, 2, 3, 4],
                        help="Run only one investigation (default: all)")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override ε for investigations 2 and 3")
    parser.add_argument("--records", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--drift-rate", type=float, default=0.25)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_all = args.investigation is None
    results: dict = {}

    if run_all or args.investigation == 1:
        results["investigation_1"] = run_investigation_1(args.records, verbose=not args.json)
    if run_all or args.investigation == 2:
        results["investigation_2"] = run_investigation_2(verbose=not args.json)
    if run_all or args.investigation == 3:
        results["investigation_3"] = run_investigation_3(args.epsilon, verbose=not args.json)
    if run_all or args.investigation == 4:
        results["investigation_4"] = run_investigation_4(
            args.records, args.iterations, args.drift_rate, verbose=not args.json
        )

    if run_all and not args.json:
        print_summary(
            results["investigation_1"],
            results["investigation_2"],
            results["investigation_3"],
            results["investigation_4"],
        )

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
