#!/usr/bin/env python3
"""Demographic parity audit for synthetic dataset releases.

Checks whether the synthetic data accurately preserves sensitive subgroup
statistics relative to the real data. A large gap (>5%) means the statistics
are well-preserved — which may be intentional (for utility) but must be
disclosed in the dataset card.

Usage:
    python scripts/run_demographic_parity_audit.py
    python scripts/run_demographic_parity_audit.py --json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Simulated HR dataset (replace with real Anote product output)
# ---------------------------------------------------------------------------

SENSITIVE_COLS = ["gender", "age_group", "department", "ethnicity"]
OUTCOME_COL = "promoted"

GENDER_PROMO = {"M": 0.22, "F": 0.14, "NB": 0.17}
AGE_PROMO = {"20-30": 0.13, "31-40": 0.21, "41-50": 0.20, "51+": 0.17}
DEPT_PROMO = {"Engineering": 0.25, "Sales": 0.18, "HR": 0.12, "Finance": 0.20, "Legal": 0.15}
ETHNICITY_PROMO = {"White": 0.21, "Asian": 0.19, "Black": 0.13, "Hispanic": 0.14, "Other": 0.16}


def _make_records(n: int, noise_scale: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        gender = rng.choice(["M", "F", "NB"])
        age_group = rng.choice(["20-30", "31-40", "41-50", "51+"])
        dept = rng.choice(["Engineering", "Sales", "HR", "Finance", "Legal"])
        ethnicity = rng.choice(["White", "Asian", "Black", "Hispanic", "Other"])
        base_rate = (
            GENDER_PROMO[gender]
            + AGE_PROMO[age_group]
            + DEPT_PROMO[dept]
            + ETHNICITY_PROMO[ethnicity]
        ) / 4
        noisy_rate = min(1.0, max(0.0, base_rate + rng.gauss(0, noise_scale)))
        promoted = int(rng.random() < noisy_rate)
        records.append({
            "gender": gender,
            "age_group": age_group,
            "department": dept,
            "ethnicity": ethnicity,
            OUTCOME_COL: promoted,
        })
    return records


def _group_rates(records: list[dict], col: str, outcome: str) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for r in records:
        key = r[col]
        counts.setdefault(key, []).append(r[outcome])
    return {k: sum(v) / len(v) for k, v in counts.items()}


def demographic_parity_audit(
    real_records: list[dict],
    synth_records: list[dict],
    sensitive_cols: list[str],
    outcome_col: str,
    parity_threshold: float = 0.05,
) -> dict[str, object]:
    """Audit whether synthetic data preserves sensitive subgroup statistics.

    Returns a report for each sensitive column with the max parity gap
    between real and synthetic data, and whether the statistics are preserved
    within the threshold.
    """
    report: dict[str, object] = {}
    all_preserved = True

    for col in sensitive_cols:
        real_rates = _group_rates(real_records, col, outcome_col)
        synth_rates = _group_rates(synth_records, col, outcome_col)
        all_groups = set(real_rates) | set(synth_rates)

        gaps: dict[str, float] = {}
        for grp in sorted(all_groups):
            r = real_rates.get(grp, 0.0)
            s = synth_rates.get(grp, 0.0)
            gaps[grp] = round(abs(r - s), 4)

        max_gap = max(gaps.values()) if gaps else 0.0
        preserved = max_gap <= parity_threshold

        if not preserved:
            all_preserved = False

        report[col] = {
            "real_rates": {k: round(v, 4) for k, v in real_rates.items()},
            "synth_rates": {k: round(v, 4) for k, v in synth_rates.items()},
            "group_gaps": gaps,
            "max_gap": round(max_gap, 4),
            "threshold": parity_threshold,
            "preserved_within_threshold": preserved,
            "disclosure_required": True,
        }

    report["summary"] = {
        "all_within_threshold": all_preserved,
        "note": (
            "Statistics preserved within threshold — disclose in dataset card that "
            "synthetic data accurately reflects subgroup outcome rates."
            if all_preserved
            else "Statistics NOT preserved within threshold — synthetic data diverges "
            "from real subgroup rates. Investigate utility vs. privacy tradeoff."
        ),
    }
    return report


# ---------------------------------------------------------------------------
# Privacy washing detection
# ---------------------------------------------------------------------------

MISUSE_PATTERNS = [
    {
        "id": "epsilon_only",
        "description": "Reporting ε without δ (obscures approximate-DP status)",
        "check": "Is (ε, δ) reported as a pair, not just ε?",
        "status": "MANUAL_REVIEW",
    },
    {
        "id": "mia_auc_self_reported",
        "description": "MIA AUC self-reported without shadow model protocol",
        "check": "Was the shadow model protocol from run_product_audit.py used?",
        "status": "MANUAL_REVIEW",
    },
    {
        "id": "high_epsilon_hipaa_claim",
        "description": "Using ε≥5 and claiming HIPAA compliance without expert cert",
        "check": "Does ε<3 for any HIPAA-labeled dataset, or is expert cert provided?",
        "status": "MANUAL_REVIEW",
    },
    {
        "id": "no_dp_deidentification",
        "description": "Claiming 'de-identification' without formal DP guarantee",
        "check": "Is DP explicitly applied? Or only traditional de-identification?",
        "status": "MANUAL_REVIEW",
    },
]


def privacy_washing_checklist(verbose: bool = True) -> list[dict]:
    if verbose:
        print("\n" + "=" * 65)
        print("Privacy Washing Checklist")
        print("=" * 65)
        for item in MISUSE_PATTERNS:
            print(f"\n  [{item['status']}] {item['description']}")
            print(f"  Check: {item['check']}")
        print(
            "\n  All items require manual review before public synthetic data release."
        )
    return MISUSE_PATTERNS


# ---------------------------------------------------------------------------
# CO2 estimator
# ---------------------------------------------------------------------------

def estimate_co2(
    gpu_hours: float = 0.0,
    cpu_hours: float = 0.0,
    tpu_hours: float = 0.0,
    provider: str = "gcp",
) -> dict[str, float]:
    """Estimate CO2e emissions for a compute job.

    Emission factors (kg CO2e per hour):
    - GPU (A100 on GCP): ~0.20 kg/hr (Lottick et al., 2019 + Google 2022 data)
    - CPU (2 vCPU on GCP): ~0.007 kg/hr
    - TPU v4 (GCP): ~0.147 kg/hr (Google Environmental Report 2023)
    """
    factors = {
        "gcp": {"gpu": 0.20, "cpu": 0.007, "tpu": 0.147},
        "aws": {"gpu": 0.23, "cpu": 0.009, "tpu": 0.0},
        "azure": {"gpu": 0.21, "cpu": 0.008, "tpu": 0.0},
    }
    f = factors.get(provider, factors["gcp"])
    gpu_co2 = gpu_hours * f["gpu"]
    cpu_co2 = cpu_hours * f["cpu"]
    tpu_co2 = tpu_hours * f["tpu"]
    total = gpu_co2 + cpu_co2 + tpu_co2
    return {
        "gpu_hours": gpu_hours,
        "cpu_hours": cpu_hours,
        "tpu_hours": tpu_hours,
        "provider": provider,
        "gpu_co2_kg": round(gpu_co2, 4),
        "cpu_co2_kg": round(cpu_co2, 4),
        "tpu_co2_kg": round(tpu_co2, 4),
        "total_co2_kg": round(total, 4),
    }


BENCHMARK_COMPUTE = {
    "tabular_experiments_cpu": {
        "description": "All tabular sweeps (ε sweep, collapse study, mechanism comparison)",
        "gpu_hours": 0,
        "cpu_hours": 4.0,
        "tpu_hours": 0,
    },
    "dp_sgd_single_run": {
        "description": "Single DP-SGD fine-tune run (ε=2, document synthesis)",
        "gpu_hours": 32,
        "cpu_hours": 0,
        "tpu_hours": 0,
    },
    "dp_sgd_full_sweep": {
        "description": "Full DP-SGD sweep (6 ε × 3 seeds × document types)",
        "gpu_hours": 576,
        "cpu_hours": 0,
        "tpu_hours": 0,
    },
    "non_dp_baseline": {
        "description": "Non-DP LLM fine-tune baseline",
        "gpu_hours": 8,
        "cpu_hours": 0,
        "tpu_hours": 0,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Demographic parity audit + CO2 estimator")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    real_records = _make_records(args.records, noise_scale=0.01, seed=42)
    synth_records = _make_records(args.records, noise_scale=0.04, seed=99)  # more noise = more DP distortion

    audit = demographic_parity_audit(
        real_records, synth_records, SENSITIVE_COLS, OUTCOME_COL, args.threshold
    )
    privacy_washing_checklist(verbose=not args.json)

    co2_table: dict[str, dict] = {}
    for key, config in BENCHMARK_COMPUTE.items():
        co2_table[key] = estimate_co2(
            gpu_hours=config["gpu_hours"],
            cpu_hours=config["cpu_hours"],
            tpu_hours=config["tpu_hours"],
            provider="gcp",
        )
        co2_table[key]["description"] = config["description"]

    if args.json:
        print(json.dumps({"parity_audit": audit, "co2_estimates": co2_table}, indent=2))
        return

    # Human-readable output
    print("\n" + "=" * 65)
    print("Demographic Parity Audit")
    print(f"n_real={args.records}  n_synth={args.records}  threshold={args.threshold:.0%}")
    print("=" * 65)
    for col, result in audit.items():
        if col == "summary":
            continue
        assert isinstance(result, dict)
        preserved = result["preserved_within_threshold"]
        icon = "✅" if preserved else "❌"
        print(f"\n  {icon} {col}  (max gap: {result['max_gap']:.1%})")
        for grp in sorted(result["group_gaps"]):
            real_r = result["real_rates"].get(grp, 0.0)
            synth_r = result["synth_rates"].get(grp, 0.0)
            gap = result["group_gaps"][grp]
            print(f"     {grp:<16}: real={real_r:.1%}  synth={synth_r:.1%}  gap={gap:.1%}")

    summary = audit["summary"]
    assert isinstance(summary, dict)
    print(f"\n  {summary['note']}")

    print("\n" + "=" * 65)
    print("CO2 Emissions Estimate (GCP, kg CO2e)")
    print("=" * 65)
    print(f"  {'Experiment':<45} {'GPU-h':>7} {'CO2 kg':>8}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*8}")
    for key, row in co2_table.items():
        print(
            f"  {row['description']:<45} "
            f"{row['gpu_hours']:>7.0f}  "
            f"{row['total_co2_kg']:>8.2f}"
        )
    total_co2 = sum(r["total_co2_kg"] for r in co2_table.values())
    print(f"  {'TOTAL (worst case: full sweep)':<45}  {'':>7}  {total_co2:>8.2f}")
    print("\n  Note: tabular experiments are CPU-only and contribute <0.03 kg CO2e total.")
    print("  The dominant cost is DP-SGD document synthesis (GPU-required).")


if __name__ == "__main__":
    main()
