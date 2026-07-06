#!/usr/bin/env python3
"""Quick progress viewer for an in-progress run_opacus_dp_sweep.py run.

Parses the "Dataset:" / "-> mean ..." lines from a sweep's stderr log and
plots whatever's completed so far, so you can check in on a long-running
sweep without waiting for it to finish.

Usage:
    python scripts/plot_sweep_progress.py /tmp/real_dp_sweep_with_fidelity_log.txt
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_log(path: str) -> list[dict]:
    dataset = None
    epsilon = None
    rows = []
    with open(path) as f:
        for line in f:
            m = re.search(r"Dataset:\s*(.+?)\s{2,}Synthesizer", line)
            if m:
                dataset = m.group(1).strip()
                continue
            m = re.search(r"ε\s*=\s*([\d.]+)\s", line)
            if m:
                epsilon = float(m.group(1))
                continue
            m = re.search(
                r"->\s*mean TSTR F1=([\d.]+).*mean MIA AUC=([\d.]+)(?:.*mean fidelity=([\d.]+))?",
                line,
            )
            if m and dataset and epsilon is not None:
                rows.append({
                    "dataset": dataset,
                    "epsilon": epsilon,
                    "tstr_f1_mean": float(m.group(1)),
                    "mia_auc_mean": float(m.group(2)),
                    "fidelity_mean": float(m.group(3)) if m.group(3) else None,
                })
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/plot_sweep_progress.py <log_file>", file=sys.stderr)
        sys.exit(1)
    log_path = sys.argv[1]
    rows = parse_log(log_path)
    if not rows:
        print("No completed (epsilon, seed-group) results found yet in log.", file=sys.stderr)
        sys.exit(0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_rows = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: r["epsilon"])
        eps_vals = [r["epsilon"] for r in ds_rows]
        ax.plot(eps_vals, [r["tstr_f1_mean"] for r in ds_rows], "o-", label="TSTR F1", color="#2ecc71")
        ax.plot(eps_vals, [r["mia_auc_mean"] for r in ds_rows], "o-", label="MIA AUC", color="#e74c3c")
        if any(r["fidelity_mean"] is not None for r in ds_rows):
            ax.plot(eps_vals, [r["fidelity_mean"] or 0 for r in ds_rows], "o-",
                     label="Fidelity", color="#3498db")
        ax.set_xlabel("ε")
        ax.set_title(ds, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    n_done = len(rows)
    fig.suptitle(f"Sweep progress (in-progress): {n_done} (dataset, ε) points completed so far", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "paper", "figures", "_sweep_progress.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  ({n_done} points)")


if __name__ == "__main__":
    main()
