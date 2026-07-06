#!/usr/bin/env python3
"""Pareto frontier visualization for the privacy-utility-fidelity benchmark.

Produces two output modes:
  1. ASCII charts (default, no dependencies) — prints to terminal; suitable for
     CI artifacts and Markdown embedding.
  2. Matplotlib PNG charts (--png flag) — saves figures to paper/figures/.

Also computes the Pareto-efficient configurations (no other point dominates on
all three dimensions simultaneously) and prints a summary table.

Usage:
    python scripts/plot_pareto.py
    python scripts/plot_pareto.py --asset tabular_hr --png
    python scripts/plot_pareto.py --all --png
"""
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from privacy_benchmark.config import EPSILON_VALUES

# ---------------------------------------------------------------------------
# Data source: results/epsilon_sweep.json (produced by run_epsilon_sweep.py)
#
# This used to be a hardcoded BENCHMARK_DATA dict of simulated values,
# disconnected from the real Opacus DP-SGD measurements wired into
# epsilon_sweep.json. That meant these figures never reflected the real,
# messier, domain-dependent measured data, only the old calibrated
# simulation. Loading from epsilon_sweep.json directly keeps the figures in
# sync with whatever's currently measured vs. simulated there, and each
# point's provenance (data_source) is threaded through so the chart can be
# labeled honestly rather than presented as uniformly measured.
# ---------------------------------------------------------------------------

_EPSILON_SWEEP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "epsilon_sweep.json"
)


def _load_epsilon_sweep() -> dict[str, list[dict]]:
    if not os.path.exists(_EPSILON_SWEEP_PATH):
        print(
            f"error: {_EPSILON_SWEEP_PATH} not found. Run "
            f"'python scripts/run_epsilon_sweep.py --json > results/epsilon_sweep.json' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(_EPSILON_SWEEP_PATH) as f:
        return json.load(f)


ALL_ASSETS = ["tabular_hr", "financial_transactions", "healthcare_ehr"]


# ---------------------------------------------------------------------------
# Pareto efficiency computation
# ---------------------------------------------------------------------------

def is_pareto_efficient(points: list[dict]) -> list[bool]:
    """Return a boolean mask of Pareto-efficient points.

    A point is Pareto-efficient if no other point has equal or better values
    on ALL three dimensions (privacy, utility, fidelity) and strictly better
    on at least one.
    """
    n = len(points)
    efficient = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pi = points[i]
            pj = points[j]
            # j dominates i if j ≥ i on all metrics and > i on at least one
            if (pj["privacy_score"] >= pi["privacy_score"] and
                    pj["utility_score"] >= pi["utility_score"] and
                    pj["fidelity_score"] >= pi["fidelity_score"] and
                    (pj["privacy_score"] > pi["privacy_score"] or
                     pj["utility_score"] > pi["utility_score"] or
                     pj["fidelity_score"] > pi["fidelity_score"])):
                efficient[i] = False
                break
    return efficient


def evaluate_asset(asset_type: str, epsilon_sweep: dict[str, list[dict]]) -> list[dict]:
    """Return the (already-evaluated) rows for one asset type from epsilon_sweep.json."""
    rows = epsilon_sweep.get(asset_type)
    if not rows:
        print(f"error: no data for asset_type={asset_type!r} in {_EPSILON_SWEEP_PATH}", file=sys.stderr)
        sys.exit(1)

    results = []
    for row in sorted(rows, key=lambda r: r["epsilon"]):
        evaluated = {
            "epsilon": row["epsilon"],
            "privacy_score": row["privacy_score"],
            "utility_score": row["utility_score"],
            "fidelity_score": row["fidelity_score"],
            "asset_type": asset_type,
            "tier": row.get("compliance_tier"),
            "data_source": row.get("data_source", "unknown"),
        }
        results.append(evaluated)

    pareto_mask = is_pareto_efficient(results)
    for result, is_eff in zip(results, pareto_mask):
        result["pareto_efficient"] = is_eff

    return results


# ---------------------------------------------------------------------------
# ASCII chart rendering
# ---------------------------------------------------------------------------

_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def _to_bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_ascii_pareto(results: list[dict], asset_type: str) -> None:
    print(f"\n{'=' * 75}")
    print(f"  Pareto Frontier: {asset_type.replace('_', ' ').title()}")
    print(f"{'=' * 75}")
    sources = {r.get("data_source", "unknown") for r in results}
    for s in sources:
        print(f"  data source: {s}")
    print(f"  {'ε':>5}  {'Tier':<20} {'Privacy':>8} {'Utility':>8} {'Fidelity':>9}  {'Pareto':>7}  Privacy bar")
    print(f"  {'-'*5}  {'-'*20} {'-'*8} {'-'*8} {'-'*9}  {'-'*7}  {'-'*20}")
    for r in results:
        marker = " ★" if r["pareto_efficient"] else "  "
        bar = _to_bar(r["privacy_score"], 16)
        print(
            f"  {r['epsilon']:>5.1f}  {r.get('tier', '?'):<20}"
            f"  {r['privacy_score']:>7.3f}  {r['utility_score']:>7.3f}  {r['fidelity_score']:>8.3f}"
            f"  {marker}  {bar}"
        )

    efficient = [r for r in results if r["pareto_efficient"]]
    print(f"\n  Pareto-efficient configurations ({len(efficient)} of {len(results)}):")
    for r in efficient:
        print(f"    ε={r['epsilon']:.1f}  privacy={r['privacy_score']:.3f}  "
              f"utility={r['utility_score']:.3f}  fidelity={r['fidelity_score']:.3f}")


def print_ascii_tradeoff_curves(all_results: dict[str, list[dict]]) -> None:
    """Print side-by-side privacy vs. utility tradeoff across asset types."""
    epsilons = EPSILON_VALUES
    print(f"\n{'=' * 75}")
    print("  Privacy vs. Utility Tradeoff (★ = Pareto-efficient)")
    print(f"  ε {'':>4}  " + "  ".join(f"{at.replace('_',' ')[:14]:<14}" for at in all_results))
    print(f"  {'─'*5}  " + "  ".join("─" * 14 for _ in all_results))

    for eps in epsilons:
        cells = []
        for asset, results in all_results.items():
            row = next((r for r in results if r["epsilon"] == eps), None)
            if row:
                marker = "★" if row["pareto_efficient"] else " "
                cells.append(f"{marker}P={row['privacy_score']:.2f} U={row['utility_score']:.2f}")
            else:
                cells.append("-" * 14)
        print(f"  {eps:>5.1f}  " + "  ".join(f"{c:<14}" for c in cells))


def print_compliance_summary(all_results: dict[str, list[dict]]) -> None:
    """Print the compliance tier → expected tradeoff summary table."""
    print(f"\n{'=' * 75}")
    print("  Compliance Tier Summary")
    print(f"  {'Tier':<20} {'ε Range':<12} {'Privacy':>8} {'Utility':>8} {'Fidelity':>9}  {'Regulation'}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*9}  {'-'*20}")

    tier_meta = {
        "strict":          ("0.1–0.5",  "GDPR Art. 89 / external share"),
        "balanced":        ("1–2",       "HIPAA / standard GDPR"),
        "utility_focused": ("5–10",      "SOX / internal analytics"),
    }
    for tier_name, (eps_range, regulation) in tier_meta.items():
        # Average across asset types for this tier
        tier_rows = [
            r for asset_results in all_results.values()
            for r in asset_results
            if r.get("tier") == tier_name
        ]
        if not tier_rows:
            continue
        avg_priv = sum(r["privacy_score"] for r in tier_rows) / len(tier_rows)
        avg_util = sum(r["utility_score"] for r in tier_rows) / len(tier_rows)
        avg_fid = sum(r["fidelity_score"] for r in tier_rows) / len(tier_rows)
        print(
            f"  {tier_name:<20} {eps_range:<12}"
            f"  {avg_priv:>7.3f}  {avg_util:>7.3f}  {avg_fid:>8.3f}  {regulation}"
        )


# ---------------------------------------------------------------------------
# Optional matplotlib output
# ---------------------------------------------------------------------------

def save_png_charts(all_results: dict[str, list[dict]], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed — skipping PNG output. Run: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {"strict": "#e74c3c", "balanced": "#2ecc71", "utility_focused": "#3498db"}
    tier_labels = {"strict": "Strict (ε=0.1–0.5)", "balanced": "Balanced (ε=1–2)",
                   "utility_focused": "Utility-focused (ε=5–10)"}

    # --- Chart 1: Privacy vs Utility per asset type ---
    # sharey=False deliberately: the real measured data has wildly different
    # utility ranges per domain (financial up to ~0.77, HR/healthcare under
    # ~0.2) -- that spread IS the headline finding (5.1.1 in the paper), so
    # forcing a shared y-axis would either clip most domains' data out of
    # view or flatten the very difference the figure exists to show.
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4.5), sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (asset, results) in zip(axes, all_results.items()):
        for r in results:
            tier = r.get("tier", "utility_focused")
            color = colors.get(tier, "#888")
            marker = "*" if r["pareto_efficient"] else "o"
            size = 180 if r["pareto_efficient"] else 80
            ax.scatter(r["privacy_score"], r["utility_score"], c=color,
                       marker=marker, s=size, zorder=5, edgecolors="white", linewidth=0.5)
            ax.annotate(f"ε={r['epsilon']:.0f}" if r["epsilon"] >= 1 else f"ε={r['epsilon']}",
                        (r["privacy_score"], r["utility_score"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)

        # Connect the Pareto-efficient points with a dashed line
        pareto_pts = sorted([r for r in results if r["pareto_efficient"]],
                            key=lambda x: x["privacy_score"])
        if len(pareto_pts) >= 2:
            ax.plot([p["privacy_score"] for p in pareto_pts],
                    [p["utility_score"] for p in pareto_pts],
                    "k--", linewidth=0.8, alpha=0.4, zorder=4)

        ax.set_xlabel("Privacy Score (1 − MIA AUC)", fontsize=10)
        ax.set_ylabel("Utility Score (TSTR F1)", fontsize=10)
        ax.set_title(asset.replace("_", " ").title(), fontsize=11, fontweight="bold")
        # Axis limits are computed from the actual data rather than hardcoded:
        # the real measured DP-SGD data has a much wider utility range (down to
        # 0.0) and a narrower privacy range (~0.49-0.53, since this pilot-scale
        # model shows little exploitable MIA signal at any tested ε) than the
        # old simulated data did.
        all_priv = [r["privacy_score"] for r in results]
        all_util = [r["utility_score"] for r in results]
        priv_pad = max(0.01, (max(all_priv) - min(all_priv)) * 0.2)
        util_pad = max(0.02, (max(all_util) - min(all_util)) * 0.15)
        ax.set_xlim(min(all_priv) - priv_pad, max(all_priv) + priv_pad)
        ax.set_ylim(max(0.0, min(all_util) - util_pad), max(all_util) + util_pad)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    sources = {r.get("data_source", "unknown") for res in all_results.values() for r in res}
    provenance = "measured" if all("measured" in s for s in sources) else "mixed measured/simulated"
    fidelity_sources = {r.get("fidelity_source", "") for res in all_results.values() for r in res}
    fidelity_note = (
        "fidelity also measured" if fidelity_sources and all("simulated" not in s for s in fidelity_sources)
        else "fidelity still simulated" if all("measured" not in s for s in fidelity_sources)
        else "fidelity mixed measured/simulated"
    )
    patches = [mpatches.Patch(color=c, label=tier_labels[k]) for k, c in colors.items()]
    fig.legend(handles=patches, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"EnterpriseSynth: Privacy-Utility Pareto Frontier ({provenance} privacy/utility; "
        f"{fidelity_note})\nNote: y-axes are NOT shared across panels -- "
        f"utility ranges differ by an order of magnitude across domains",
        fontsize=10, y=1.1,
    )
    plt.tight_layout()
    path1 = os.path.join(output_dir, "pareto_privacy_utility.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # --- Chart 2: ε vs all three metrics (line chart) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    line_styles = {"privacy_score": "-", "utility_score": "--", "fidelity_score": ":"}
    metric_colors = {"privacy_score": "#e74c3c", "utility_score": "#2ecc71",
                     "fidelity_score": "#3498db"}
    metric_labels = {"privacy_score": "Privacy (1−AUC)",
                     "utility_score": "Utility (TSTR F1)",
                     "fidelity_score": "Fidelity"}

    for asset, results in all_results.items():
        eps_vals = [r["epsilon"] for r in results]
        for metric, ls in line_styles.items():
            vals = [r[metric] for r in results]
            alpha = 0.9 if asset == "tabular_hr" else 0.45
            label = f"{metric_labels[metric]} ({asset.replace('_',' ')})" if alpha > 0.6 else None
            ax.semilogx(eps_vals, vals, ls, color=metric_colors[metric],
                        alpha=alpha, linewidth=2 if alpha > 0.6 else 1, label=label)

    # y-axis range from actual data: real measured utility can go down to 0.0
    # (e.g. HR/adult at low ε), unlike the old simulated data which never
    # dropped below ~0.55.
    all_vals = [
        r[metric]
        for results in all_results.values()
        for r in results
        for metric in ("privacy_score", "utility_score", "fidelity_score")
    ]
    y_lo = max(0.0, min(all_vals) - 0.05)
    y_hi = min(1.05, max(all_vals) + 0.08)

    # Shade compliance tiers
    ax.axvspan(0.09, 0.55, alpha=0.06, color="#e74c3c", label=None)
    ax.axvspan(0.9, 2.2, alpha=0.06, color="#2ecc71", label=None)
    ax.axvspan(4.5, 11, alpha=0.06, color="#3498db", label=None)
    label_y = y_hi - 0.03
    ax.text(0.15, label_y, "Strict", fontsize=8, color="#c0392b", ha="center")
    ax.text(1.4, label_y, "Balanced", fontsize=8, color="#27ae60", ha="center")
    ax.text(7.0, label_y, "Utility-focused", fontsize=8, color="#2980b9", ha="center")

    fidelity_sources = {
        r.get("fidelity_source", "") for res in all_results.values() for r in res
    }
    fidelity_provenance = (
        "measured" if fidelity_sources and all("simulated" not in s for s in fidelity_sources)
        else "simulated" if all("measured" not in s for s in fidelity_sources)
        else "mixed measured/simulated"
    )

    ax.set_xlabel("ε (privacy budget, log scale)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Privacy, Utility, Fidelity vs. ε (privacy/utility measured, fidelity {fidelity_provenance})",
        fontsize=11,
    )
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "epsilon_tradeoff_curves.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto frontier visualization")
    parser.add_argument("--asset", default="all",
                        choices=ALL_ASSETS + ["all"])
    parser.add_argument("--all", dest="show_all", action="store_true")
    parser.add_argument("--png", action="store_true",
                        help="Save matplotlib PNG charts to paper/figures/")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    assets = ALL_ASSETS if (args.asset == "all" or args.show_all) else [args.asset]
    all_results: dict[str, list[dict]] = {}

    epsilon_sweep = _load_epsilon_sweep()
    for asset in assets:
        all_results[asset] = evaluate_asset(asset, epsilon_sweep)

    if args.json:
        print(json.dumps(all_results, indent=2))
        return

    for asset, results in all_results.items():
        print_ascii_pareto(results, asset)

    if len(all_results) > 1:
        print_ascii_tradeoff_curves(all_results)
        print_compliance_summary(all_results)

    if args.png:
        figures_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
        save_png_charts(all_results, figures_dir)


if __name__ == "__main__":
    main()
