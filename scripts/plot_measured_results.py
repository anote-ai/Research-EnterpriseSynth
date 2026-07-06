#!/usr/bin/env python3
"""Figures for the measured (non-DP-sweep) results: model collapse mitigation
comparison and the real-data synthesizer baseline comparison.

The DP-SGD Pareto figures live in plot_pareto.py; this script covers the
other two measured experiments that didn't have any figure at all before:
  - Model collapse mitigation comparison (Section 6, paper/draft.md)
  - Real-data synthesizer baseline (Section 4/5.1, paper/draft.md)

Usage:
    python scripts/plot_measured_results.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_collapse.pipeline import run_model_collapse_pipeline
from model_collapse.evaluator import evaluate_model_collapse

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

FIELD = "record_type"
MINORITY_CLASSES = ("fraud", "critical_security")


def plot_collapse_mitigation():
    """Real measured tail-entropy trajectory per mitigation strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_generations = 5
    collapse_rate = 0.30

    result = evaluate_model_collapse(
        n_generations=n_generations,
        n_records=400,
        field=FIELD,
        minority_classes=MINORITY_CLASSES,
        collapse_rate=collapse_rate,
        tolerance=0.10,
        seed=42,
    )
    original = run_model_collapse_pipeline(
        n_generations=0, n_records=400, field=FIELD,
        minority_classes=MINORITY_CLASSES, collapse_rate=collapse_rate, seed=42,
    )
    baseline_h = original[0].tail_entropy

    strategy_labels = {
        "baseline": "No mitigation",
        "anchored": "Real-data anchoring",
        "diversity": "Diversity-rewarded sampling",
        "combined": "Combined",
    }
    colors = {"baseline": "#e74c3c", "anchored": "#f39c12",
              "diversity": "#2ecc71", "combined": "#3498db"}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    generations = list(range(n_generations + 1))
    for strategy, label in strategy_labels.items():
        entropies = [row["tail_entropy"] for row in result[strategy]]
        ax.plot(generations, entropies, marker="o", label=label,
                color=colors[strategy], linewidth=2)

    ax.axhline(baseline_h, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(0.05, baseline_h + 0.01, "Generation-0 baseline", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    tolerance_line = baseline_h * 0.90
    ax.axhline(tolerance_line, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.05, tolerance_line - 0.03, "10% tolerance threshold", fontsize=8,
            color="black", transform=ax.get_yaxis_transform())

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Tail coverage entropy", fontsize=11)
    ax.set_title(
        f"Model Collapse Mitigation (measured, collapse_rate={collapse_rate:.0%}, "
        f"{n_generations} generations)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "collapse_mitigation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_baseline_comparison():
    """Real measured no-DP synthesizer baseline: oracle vs. TSTR F1 per domain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path_in = os.path.join(RESULTS_DIR, "baseline_sdg.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        rows = json.load(f)

    datasets = sorted({r["dataset"] for r in rows})
    synthesizers = sorted({r["synthesizer"] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / (len(synthesizers) + 1)
    x = np.arange(len(datasets))

    colors = {"GaussianCopula": "#e74c3c", "CTGAN": "#3498db", "TVAE": "#2ecc71"}
    for i, synth in enumerate(synthesizers):
        vals = []
        for ds in datasets:
            row = next((r for r in rows if r["dataset"] == ds and r["synthesizer"] == synth), None)
            vals.append(row["utility_retention_f1"] if row else 0.0)
        ax.bar(x + i * width, vals, width, label=synth, color=colors.get(synth, "#888"))

    display_names = []
    for ds in datasets:
        row = next(r for r in rows if r["dataset"] == ds)
        display_names.append(row.get("dataset_display", ds))

    ax.set_xticks(x + width * (len(synthesizers) - 1) / 2)
    ax.set_xticklabels(display_names, fontsize=9)
    ax.set_ylabel("Utility retention (TSTR F1 / oracle F1)", fontsize=11)
    ax.set_title("Real Synthesizer Baseline: No-DP Utility Retention (measured)", fontsize=11)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "baseline_synthesizer_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_downstream_tasks():
    """Classification/regression/anomaly retention vs epsilon, per domain.

    Classification and no-DP baselines for all three tasks are measured;
    all DP-adjusted values (all three tasks) are estimated via a logistic
    degradation curve, not real DP-SGD training -- see RESEARCH_STATUS.md.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path_in = os.path.join(RESULTS_DIR, "downstream_tasks.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        data = json.load(f)

    domains = list(data.keys())
    tasks = [("pct_classification", "Classification"), ("pct_regression", "Regression"),
             ("pct_anomaly", "Anomaly detection")]
    task_colors = {"Classification": "#3498db", "Regression": "#2ecc71", "Anomaly detection": "#e74c3c"}

    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4.5), sharey=True)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        rows = sorted(data[domain], key=lambda r: r["epsilon"])
        eps_vals = [r["epsilon"] for r in rows]
        for key, label in tasks:
            vals = [r[key] for r in rows]
            ax.semilogx(eps_vals, vals, marker="o", label=label, color=task_colors[label])
        ax.set_xlabel("ε (log scale)", fontsize=10)
        ax.set_title(domain.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("DP retention (DP value / no-DP baseline)", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Downstream Task Diversity (no-DP baselines measured; DP retention estimated "
        "via logistic curve, not real DP-SGD)",
        fontsize=10, y=1.03,
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "downstream_tasks_retention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_document_dp_sweep():
    """Document-domain TSTR utility vs epsilon, per document category.

    Fully simulated (literature-calibrated starting values) -- there is no
    real document oracle baseline yet. Labeled accordingly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path_in = os.path.join(RESULTS_DIR, "document_dp_sweep.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for category, rows in data.items():
        # Some entries (e.g. "cliff_crossings") are summary rows without an
        # "epsilon" key, not per-epsilon data points -- skip those.
        rows_sorted = sorted((r for r in rows if "epsilon" in r), key=lambda r: r["epsilon"])
        eps_vals = [r["epsilon"] for r in rows_sorted]
        vals = [r["tstr_f1_mean"] for r in rows_sorted]
        ax.plot(eps_vals, vals, marker="o", label=category.replace("_", " ").title())

    ax.set_xlabel("ε", fontsize=11)
    ax.set_ylabel("TSTR F1 (simulated)", fontsize=11)
    ax.set_title("Document DP Sweep -- SIMULATED, no real document oracle baseline yet", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "document_dp_sweep.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_dp_mechanism_comparison():
    """Constraint violation rate by DP mechanism, per epsilon.

    Gaussian/Laplace/discrete are run directly (measured); randomized_response
    and any PATE-style entries are literature-calibrated estimates -- see
    RESEARCH_STATUS.md. Provenance is not separable per-mechanism in the
    committed JSON, so the figure is labeled as mixed measured/estimated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path_in = os.path.join(RESULTS_DIR, "dp_mechanism_comparison.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        data = json.load(f)

    epsilons = sorted(data.keys(), key=float)
    mechanisms = sorted({r["mechanism"] for rows in data.values() for r in rows} - {"none"})

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(mechanisms)
    x = np.arange(len(epsilons))
    colors = {"gaussian": "#3498db", "laplace": "#2ecc71",
              "discrete": "#f39c12", "randomized_response": "#9b59b6"}

    for i, mech in enumerate(mechanisms):
        vals = []
        for eps in epsilons:
            row = next((r for r in data[eps] if r["mechanism"] == mech), None)
            vals.append(row["constraint_violation_rate"] if row else 0.0)
        ax.bar(x + i * width, vals, width, label=mech, color=colors.get(mech, "#888"))

    ax.set_xticks(x + width * (len(mechanisms) - 1) / 2)
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_ylabel("Constraint violation rate (lower is better)", fontsize=11)
    ax.set_title(
        "DP Mechanism Comparison (Gaussian/Laplace/discrete measured directly; "
        "randomized_response estimated)",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "dp_mechanism_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_fidelity_correlation():
    """Spearman correlation between fidelity metrics and downstream utility.

    Fully simulated -- computed from simulation outputs, not a separate
    held-out real dataset. See RESEARCH_STATUS.md.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path_in = os.path.join(RESULTS_DIR, "fidelity_correlation.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        data = json.load(f)

    asset_types = list(data.keys())
    all_metrics = sorted({m for a in data.values() for m in a["spearman_rho"]})

    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.8 / len(asset_types)
    x = np.arange(len(all_metrics))
    colors = {"tabular": "#3498db", "documents": "#e74c3c"}

    for i, asset in enumerate(asset_types):
        vals = [data[asset]["spearman_rho"].get(m, 0.0) for m in all_metrics]
        ax.bar(x + i * width, vals, width, label=data[asset].get("asset_type", asset),
               color=colors.get(asset, "#888"))

    ax.set_xticks(x + width * (len(asset_types) - 1) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in all_metrics], fontsize=8)
    ax.set_ylabel("Spearman ρ vs. downstream utility", fontsize=11)
    ax.set_title("Fidelity-Utility Correlation -- SIMULATED (not held-out real data)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fidelity_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_product_audit():
    """Constraint violation rate by enterprise asset type (investigation_1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path_in = os.path.join(RESULTS_DIR, "product_audit.json")
    if not os.path.exists(path_in):
        print(f"skip: {path_in} not found")
        return
    with open(path_in) as f:
        data = json.load(f)

    rows = data.get("investigation_1", {})
    assets = list(rows.keys())
    violation_rates = [rows[a]["violation_rate"] for a in assets]
    sources = [rows[a].get("source", "unknown") for a in assets]
    colors = ["#2ecc71" if s == "measured" else "#f39c12" for s in sources]
    threshold = next((rows[a]["threshold"] for a in assets), 0.02)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar([a.replace("_", "\n") for a in assets], violation_rates, color=colors)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.set_ylabel("Constraint violation rate", fontsize=11)
    ax.set_title("Product Audit: Constraint Violations by Asset Type (green=measured, orange=estimated)",
                 fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "product_audit_violations.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_collapse_mitigation()
    plot_baseline_comparison()
    plot_downstream_tasks()
    plot_document_dp_sweep()
    plot_dp_mechanism_comparison()
    plot_fidelity_correlation()
    plot_product_audit()


if __name__ == "__main__":
    main()
