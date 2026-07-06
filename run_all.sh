#!/usr/bin/env bash
# Reproduce every table and figure from the EnterpriseSynth paper.
#
# Usage:
#   bash run_all.sh            # full reproduction (CPU, ~25 min)
#   bash run_all.sh --quick    # quick smoke-test with reduced params (~3 min)
#
# Prerequisites: Python 3.10+, pip
# All outputs land in results/ and paper/figures/.

set -euo pipefail

QUICK=0
for arg in "$@"; do
  [[ "$arg" == "--quick" ]] && QUICK=1
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── 0. Environment ────────────────────────────────────────────────────────────
echo "[0/7] Installing dependencies..."
pip install -e ".[dev]" -q

mkdir -p results paper/figures

# ── 1. ε-Utility tradeoff (Table 2 / Appendix B) ────────────────────────────
echo "[1/7] Running epsilon-utility sweep (Table 2)..."
if [[ $QUICK -eq 1 ]]; then
  python scripts/run_epsilon_sweep.py --seeds 2 --json > results/epsilon_sweep.json
else
  python scripts/run_epsilon_sweep.py --seeds 10 --json > results/epsilon_sweep.json
fi

# ── 2. Model collapse study (Table 3 / Figure 2) ─────────────────────────────
echo "[2/7] Running model collapse study (Table 3)..."
if [[ $QUICK -eq 1 ]]; then
  python scripts/run_collapse_study.py --generations 5 --json > results/collapse_study.json
else
  python scripts/run_collapse_study.py --generations 10 --json > results/collapse_study.json
fi

# ── 3. Fidelity-TSTR correlation (Table 4) ───────────────────────────────────
echo "[3/7] Running fidelity-TSTR correlation (Table 4)..."
python scripts/run_fidelity_correlation.py --json > results/fidelity_correlation.json

# ── 4. DP document sweep (Table 5) ───────────────────────────────────────────
echo "[4/7] Running DP document sweep (Table 5)..."
python scripts/run_document_dp_sweep.py --json > results/document_dp_sweep.json

# ── 5. DP mechanism comparison (Table 6) ─────────────────────────────────────
echo "[5/7] Running DP mechanism comparison (Table 6)..."
python scripts/run_dp_mechanism_comparison.py --json > results/dp_mechanism_comparison.json

# ── 6. Product audit (supplementary) ─────────────────────────────────────────
echo "[6/7] Running product audit (supplementary)..."
python scripts/run_product_audit.py --json > results/product_audit.json

# ── 7. Figures (Pareto frontier + all other measured results) ──────────────
# Runs last since plot_measured_results.py reads product_audit.json from step 6.
echo "[7/7] Generating figures (Pareto frontier + all other measured results)..."
python scripts/plot_pareto.py --all --png
python scripts/plot_measured_results.py

echo ""
echo "Reproduction complete."
echo "  Tables/data -> results/"
echo "  Figures     -> paper/figures/"
echo ""
echo "Key results:"
python - <<'PYEOF'
import json, os

def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

epsilon = _load("results/epsilon_sweep.json")
collapse = _load("results/collapse_study.json")

if epsilon:
    print(f"  epsilon sweep: {len(epsilon)} entries written")

if collapse:
    print(f"  collapse study: {len(collapse)} entries written")
PYEOF
