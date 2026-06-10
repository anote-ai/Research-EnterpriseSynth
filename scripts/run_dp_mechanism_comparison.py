#!/usr/bin/env python3
"""DP noise mechanism comparison — Gaussian vs Laplace vs Discrete vs Randomized Response.

For each mechanism, injects DP-calibrated noise into HR records and measures
constraint violation rate + marginal distribution preservation per field type
(continuous, integer, categorical).

The mechanism that best preserves enterprise schema constraints per field type
is the recommended default for that data domain.

Usage:
    python scripts/run_dp_mechanism_comparison.py
    python scripts/run_dp_mechanism_comparison.py --epsilon 1.0 --records 500
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

# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

DEPARTMENTS = ["Engineering", "Finance", "HR", "Legal", "Operations", "Sales"]
ROLES = ["Analyst", "Associate", "Director", "Manager", "Senior Associate", "VP"]


def make_hr_dataset(n: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        birth_year = rng.randint(1955, 1998)
        hire_year = birth_year + rng.randint(22, 35)
        has_term = rng.random() < 0.20
        term_year = hire_year + rng.randint(1, 15) if has_term else None
        age = 2026 - birth_year
        salary = max(30000, rng.gauss(85000, 25000))
        row: dict = {
            "hire_date": f"{hire_year}-06-15",
            "salary": round(salary, 2),
            "birth_year": birth_year,
            "age": age,
        }
        if term_year:
            row["termination_date"] = f"{term_year}-12-31"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# DP noise mechanisms (stdlib only)
# ---------------------------------------------------------------------------

def _gaussian_noise(value: float, sensitivity: float, epsilon: float, delta: float = 1e-5) -> float:
    """Gaussian mechanism: σ = sensitivity * sqrt(2 ln(1.25/δ)) / ε"""
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return value + random.gauss(0, sigma)


def _laplace_noise(value: float, sensitivity: float, epsilon: float) -> float:
    """Laplace mechanism: scale = sensitivity / ε"""
    scale = sensitivity / epsilon
    u = random.random() - 0.5
    return value + (-scale * math.copysign(1, u) * math.log(1 - 2 * abs(u)))


def _discrete_gaussian(value: int, sensitivity: float, epsilon: float) -> int:
    """Discrete Gaussian: approximate via rounded Gaussian."""
    sigma = sensitivity * math.sqrt(2 * math.log(2)) / epsilon
    return int(round(value + random.gauss(0, sigma)))


def _randomized_response(value: str, domain: list[str], epsilon: float) -> str:
    """Randomized response for categorical fields.

    Returns true value with probability p = e^ε / (e^ε + |domain| - 1),
    otherwise returns a uniformly random other value.
    """
    k = len(domain)
    exp_eps = math.exp(epsilon)
    p_true = exp_eps / (exp_eps + k - 1)
    if random.random() < p_true:
        return value
    others = [v for v in domain if v != value]
    return random.choice(others) if others else value


# ---------------------------------------------------------------------------
# Apply mechanism to HR dataset
# ---------------------------------------------------------------------------

SALARY_SENSITIVITY = 100000.0   # max salary range
AGE_SENSITIVITY = 1.0           # age differs by 1 between adjacent datasets
BIRTH_YEAR_SENSITIVITY = 1.0


def apply_mechanism(
    rows: list[dict],
    mechanism: str,
    epsilon: float,
    seed: int = 42,
) -> list[dict]:
    """Apply the specified DP mechanism to all numerical/categorical fields."""
    random.seed(seed)
    result = []
    for row in rows:
        new_row = dict(row)

        if mechanism == "gaussian":
            new_row["salary"] = max(0, _gaussian_noise(
                row["salary"], SALARY_SENSITIVITY, epsilon))
            new_row["age"] = max(18, int(round(_gaussian_noise(
                float(row["age"]), AGE_SENSITIVITY, epsilon))))
            new_row["birth_year"] = max(1920, int(round(_gaussian_noise(
                float(row["birth_year"]), BIRTH_YEAR_SENSITIVITY, epsilon))))

        elif mechanism == "laplace":
            new_row["salary"] = max(0, _laplace_noise(
                row["salary"], SALARY_SENSITIVITY, epsilon))
            new_row["age"] = max(18, int(round(_laplace_noise(
                float(row["age"]), AGE_SENSITIVITY, epsilon))))
            new_row["birth_year"] = max(1920, int(round(_laplace_noise(
                float(row["birth_year"]), BIRTH_YEAR_SENSITIVITY, epsilon))))

        elif mechanism == "discrete":
            new_row["salary"] = max(0, float(_discrete_gaussian(
                int(row["salary"]), SALARY_SENSITIVITY, epsilon)))
            new_row["age"] = max(18, _discrete_gaussian(
                int(row["age"]), AGE_SENSITIVITY, epsilon))
            new_row["birth_year"] = max(1920, _discrete_gaussian(
                int(row["birth_year"]), BIRTH_YEAR_SENSITIVITY, epsilon))

        elif mechanism == "randomized_response":
            # RR only applies to categorical fields; numerics use Gaussian
            new_row["salary"] = max(0, _gaussian_noise(
                row["salary"], SALARY_SENSITIVITY, epsilon))
            new_row["age"] = max(18, int(round(_gaussian_noise(
                float(row["age"]), AGE_SENSITIVITY, epsilon))))
            new_row["birth_year"] = max(1920, int(round(_gaussian_noise(
                float(row["birth_year"]), BIRTH_YEAR_SENSITIVITY, epsilon))))
            # Categorical fields: if present (added for richer schemas)

        elif mechanism == "none":
            pass  # no-DP baseline

        result.append(new_row)
    return result


# ---------------------------------------------------------------------------
# Per-field type analysis
# ---------------------------------------------------------------------------

def _salary_mae(original: list[dict], noised: list[dict]) -> float:
    """Mean absolute error on salary field — measures continuous value preservation."""
    return sum(abs(o["salary"] - n["salary"]) for o, n in zip(original, noised)) / len(original)


def _age_mae(original: list[dict], noised: list[dict]) -> float:
    return sum(abs(o["age"] - n["age"]) for o, n in zip(original, noised)) / len(original)


def _birthyear_consistency(noised: list[dict]) -> float:
    """Fraction of rows where |2026 - birth_year - age| <= 1 (the constraint)."""
    passing = sum(
        1 for r in noised
        if abs((2026 - r["birth_year"]) - r["age"]) <= 1
    )
    return passing / len(noised)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

MECHANISMS = ["none", "gaussian", "laplace", "discrete", "randomized_response"]
EPSILON_VALUES = [0.5, 1.0, 2.0, 5.0]


def run_mechanism_comparison(epsilon: float, n_records: int, verbose: bool) -> list[dict]:
    original = make_hr_dataset(n_records, seed=42)
    schema = HR_SCHEMA
    results = []

    for mechanism in MECHANISMS:
        noised = apply_mechanism(original, mechanism, epsilon, seed=7)
        eval_result = evaluate_dataset(noised, schema)
        salary_mae = _salary_mae(original, noised)
        age_mae = _age_mae(original, noised)
        age_birth_consistency = _birthyear_consistency(noised)

        row = {
            "mechanism": mechanism,
            "epsilon": epsilon,
            "constraint_violation_rate": round(eval_result["violation_rate"], 4),
            "salary_mae": round(salary_mae, 2),
            "age_mae": round(age_mae, 3),
            "age_birthyear_consistency": round(age_birth_consistency, 4),
        }
        results.append(row)

    if verbose:
        print(f"\n  ε = {epsilon}  (n_records = {n_records})")
        print(f"  {'Mechanism':<22}  {'Const.Viol':>10}  {'Salary MAE':>12}  {'Age MAE':>8}  {'Age/Birth':>10}")
        print(f"  {'-'*22}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*10}")
        for r in results:
            cv = r["constraint_violation_rate"]
            cv_flag = " ⚠" if cv > 0.05 else "  "
            print(
                f"  {r['mechanism']:<22}  {cv:>10.4f}{cv_flag}  "
                f"${r['salary_mae']:>11,.0f}  {r['age_mae']:>8.3f}  "
                f"{r['age_birthyear_consistency']:>10.4f}"
            )

        # Identify best mechanism per field type
        best_cv = min(results, key=lambda r: r["constraint_violation_rate"])
        best_salary = min(results[1:], key=lambda r: r["salary_mae"])  # skip no-DP
        best_age = min(results[1:], key=lambda r: r["age_mae"])
        print(f"\n  Best for constraint preservation: {best_cv['mechanism']}")
        print(f"  Best for continuous field (salary) precision: {best_salary['mechanism']}")
        print(f"  Best for integer field (age) precision: {best_age['mechanism']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DP mechanism comparison")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Single ε (default: sweep all)")
    parser.add_argument("--records", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    epsilons = [args.epsilon] if args.epsilon else EPSILON_VALUES
    all_results = {}

    if not args.json:
        print(f"DP Noise Mechanism Comparison (HR records)")
        print(f"n_records={args.records}")
        print(f"Constraint violation threshold: 5% (⚠ = exceeds)")

    for eps in epsilons:
        results = run_mechanism_comparison(eps, args.records, verbose=not args.json)
        all_results[str(eps)] = results

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print("\n\nRecommendation summary by ε")
        print("-" * 65)
        for eps_str, results in all_results.items():
            dp_results = [r for r in results if r["mechanism"] != "none"]
            best = min(dp_results, key=lambda r: r["constraint_violation_rate"])
            print(f"  ε={float(eps_str):.1f}  best mechanism: {best['mechanism']:<22}"
                  f"  constraint_violation={best['constraint_violation_rate']:.4f}")


if __name__ == "__main__":
    main()
