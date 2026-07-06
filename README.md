# EnterpriseSynth

[![CI](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml/badge.svg)](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

When should an enterprise team use synthetic data instead of real data — and at what privacy budget does the tradeoff stop being worth it?

**EnterpriseSynth** is a research benchmark for evaluating the privacy-utility-fidelity tradeoff of differentially private synthetic data generation across regulated enterprise domains: HR records, healthcare EHR, financial transactions, IoT sensor logs, and e-commerce relational data. The repository supports both in-repo benchmarking (deterministic scripts producing JSON results) and paper-oriented workflows (draft and appendix tables that cite the same result files).

---

## What This Repository Contributes

EnterpriseSynth focuses on four research questions:

**DP budget selection:** How much utility (TSTR F1) do you sacrifice at each privacy level (ε ∈ {0.1, 0.5, 1, 2, 5, 10}, δ=1e-5) — and which ε maps to GDPR, HIPAA, and SOX compliance requirements?

**Domain sensitivity:** Does the same DP budget cost more utility in financial time-series data than in tabular HR records? Which synthesizer (CTGAN / TVAE / GaussianCopula) performs best per domain?

**Fidelity-utility correlation:** Which fidelity metric (Wasserstein distance, constraint violation rate, BERTScore) best predicts downstream model utility across enterprise data types?

**Iterative retraining safety:** What happens to rare records (fraud, security incidents, edge medical cases) when you retrain on synthetic outputs across multiple generations — and which mitigation strategies contain the collapse?

The current repository includes:

- A three-dimensional evaluation framework: Privacy (MIA AUC) × Utility (TSTR F1) × Fidelity (Wasserstein + constraint violation rate)
- Real measured baselines: CTGAN, TVAE, and GaussianCopula trained on Adult Income, Credit-G, and Diabetes PIMA (UCI) with measured wall-clock times
- Domain-specific sensitivity multipliers (tabular_hr=1.0 → ecommerce_relational=1.70) that produce distinct Pareto curves per domain
- A multi-generation model collapse study with tail-coverage entropy metrics and two validated mitigations
- Bootstrap confidence intervals and Wilcoxon signed-rank tests with Bonferroni correction throughout
- `scripts/run_opacus_dp_sweep.py` — end-to-end real DP-SGD sweep via Opacus (runs when `pip install opacus torch` is available)

---

## Current Status

| Experiment | Current status | Best entrypoint |
| --- | --- | --- |
| Real baseline (no-DP) | Measured and reproducible in-repo | `python scripts/run_baseline_sdg.py` |
| ε sweep — Pareto curves | **Measured (pilot scale)** — real Opacus DP-SGD, 5-seed mean; fidelity column still simulated | `python scripts/run_epsilon_sweep.py` |
| DP real integration | Oracle + no-DP TSTR measured; DP F1 estimated | `python scripts/run_dp_real_integration.py` |
| Downstream task diversity | Classification/regression/anomaly no-DP baselines measured; DP values estimated | `python scripts/run_downstream_tasks.py` |
| Model collapse study | Measured in controlled simulation pipeline | `python scripts/run_collapse_study.py` |
| DP mechanism comparison | Gaussian / Laplace / Discrete run directly | `python scripts/run_dp_mechanism_comparison.py` |
| Real DP-SGD sweep | Opacus-backed DPVAE, full 3-domain × 6-ε × 5-seed sweep measured and committed | `python scripts/run_opacus_dp_sweep.py` |

Useful repository documents:

- [ONE_PAGER.md](ONE_PAGER.md) — 2-minute summary: core findings, provenance at a glance, and what's still open
- [DESIGN_DOC.md](DESIGN_DOC.md)
- [RESEARCH_STATUS.md](RESEARCH_STATUS.md) — full real vs. simulated provenance for every result file
- [paper/draft.md](paper/draft.md) — full paper draft (Abstract through Appendices)

---

## Current In-Repo Findings

These are the current repo-verified findings, not a claim that every paper-ready external measurement has been finalized:

**Real baseline (measured):** TVAE retains 94.3% of oracle F1 on HR data; CTGAN retains 98.3% on financial data; GaussianCopula retains 91.4% on healthcare EHR. These establish the utility ceiling before DP is applied.

**ε sweep (measured, pilot-scale Opacus DP-SGD):** At ε=2 (HIPAA-compatible tier, δ=1e-5), measured DP utility retention is domain-dependent: 18% (HR), 91% (financial), 20% (healthcare) — real DP-SGD training on small real datasets shows far more variance and less uniformity than the earlier calibrated simulation suggested. Notably, financial transactions is the *most* DP-robust domain measured here, the opposite of the design doc's original "financial degrades fastest" hypothesis. See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) and paper/draft.md Section 5.1.1 for the full discussion, including known training-instability limitations of the pilot-scale DPVAE synthesizer.

**Model collapse (measured in controlled pipeline):** Tail-record entropy drops to warning threshold by generation 2–3 and critical threshold by generation 5 at a 30% collapse rate. Fraud and security-class records fall below 0.5% representation by generation 7 without mitigation. Real-data anchoring and diversity-rewarded sampling both keep tail diversity within 10% of generation-0 baseline.

**Fidelity-utility correlation (simulated pipeline):** Constraint violation rate is the dominant fidelity predictor for tabular assets (|ρ|=1.0); BERTScore is dominant for document assets (|ρ|=1.0). Multiple fidelity metrics are required — no single metric is sufficient for both asset types.

---

## Result Provenance Matters

This repository intentionally tracks more than one kind of result:

| Result type | Meaning |
| --- | --- |
| **Measured** | Real SDV training runs on UCI public datasets with wall-clock times and actual F1 scores |
| **Estimated** | Calibrated retention curves applied to real measured baselines (e.g. DP F1 = oracle × retention(ε)) |
| **Simulated** | DomainSpec logistic model outputs — no real DP-SGD training; domain-ordered and monotone by design |

The ε sweep and Table 2 DP columns are now **measured** from a real, pilot-scale Opacus DP-SGD training pipeline (`scripts/run_opacus_dp_sweep.py`, committed in `results/real_dp_sweep.json`), not a calibrated simulation. This measured data is noisier and less uniform across domains than the earlier simulation — some domains show non-monotonic behavior at extreme ε values, which we report as-is rather than smoothing over. Fidelity remains simulated (no real fidelity metric has been computed on the DPVAE output yet). See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for the full accounting, including known limitations of the pilot-scale synthesizer.

---

## Benchmark Pipeline

```
Real Enterprise Dataset (HR / EHR / Financial / IoT / E-commerce)
        |
        v
  [DP-SGD Synthesizer]  ←── ε, δ=1e-5 budget
  (Opacus-backed DPVAE; non-DP baselines use CTGAN / TVAE / GaussianCopula)
        |
        v
  Synthetic Dataset
        |
        +──────────────────────────────────────+
        |                                      |
        v                                      v
  [TSTR Evaluation]                   [Privacy Audit]
  Train classifier on synthetic,      Membership Inference Attack
  test F1 on held-out real data       → AUC → privacy_score = 1−AUC
        |                                      |
        v                                      v
  utility_score                       privacy_score
        |                                      |
        +──────────────────────────────────────+
        |
        v
  [Fidelity Metrics]
  Wasserstein distance, column correlations,
  constraint violation rate (logical consistency)
        |
        v
  Pareto Results  →  results/epsilon_sweep.json
  (privacy × utility × fidelity per ε, per domain)
```

---

## Quick Start

```bash
pip install -e ".[dev]"

# Run the full ε sweep across all three domains
python scripts/run_epsilon_sweep.py

# Real-data baselines (Adult Income, Credit-G, Diabetes PIMA) — takes ~20 min
python scripts/run_baseline_sdg.py

# Real DP-SGD sweep via Opacus-backed DPVAE — requires pip install opacus torch
python scripts/run_opacus_dp_sweep.py --dry-run   # check deps
python scripts/run_opacus_dp_sweep.py --json > results/real_dp_sweep.json
```

```python
from privacy_benchmark.evaluator import evaluate_with_ci
from privacy_benchmark.domains import get_domain

domain = get_domain("financial_transactions")

result = evaluate_with_ci(
    epsilon=2.0,
    auc=0.58,
    tstr_scores=[0.61, 0.63, 0.60, 0.62, 0.64],
    fidelity=0.81,
)
print(result)
# {
#   "epsilon": 2.0,
#   "privacy_score": 0.42,
#   "utility_score": 0.62,
#   "tstr_ci_lower": 0.595,
#   "tstr_ci_upper": 0.645,
#   "compliance_tier": "balanced",
#   "below_utility_cliff": false
# }
```

---

## Reproduce All Results

```bash
bash run_all.sh          # full reproduction (~25 min, CPU only)
bash run_all.sh --quick  # smoke test (~3 min)
python -m pytest tests/ -v  # 296 tests
```

---

## Figures

All figures in `paper/figures/` are generated directly from committed `results/*.json` files (`scripts/plot_pareto.py` and `scripts/plot_measured_results.py`), so they always match whatever is currently measured vs. simulated in those files:

| Figure | Experiment | Provenance |
| --- | --- | --- |
| `pareto_privacy_utility.png`, `epsilon_tradeoff_curves.png` | ε sweep — Pareto curves | Privacy/utility measured (real DP-SGD); fidelity simulated |
| `baseline_synthesizer_comparison.png` | Real no-DP synthesizer baseline | Measured |
| `collapse_mitigation_comparison.png` | Model collapse mitigation | Measured |
| `downstream_tasks_retention.png` | Downstream task diversity | No-DP baselines measured; DP retention estimated |
| `document_dp_sweep.png` | Document DP sweep | Simulated |
| `dp_mechanism_comparison.png` | DP mechanism comparison | Gaussian/Laplace/discrete measured; randomized_response estimated |
| `fidelity_correlation.png` | Fidelity-utility correlation | Simulated |
| `product_audit_violations.png` | Product audit | Mixed measured/estimated (color-coded in the figure) |

Regenerate all of them with:

```bash
python scripts/plot_pareto.py --all --png
python scripts/plot_measured_results.py
```

---

## Demo Runbook (~8 minutes, fully offline)

Every headline result runs from committed data — no API key required.
Only Step 4 (real DP training) needs `pip install opacus torch`; it has an offline fallback.

### 0. Setup

```bash
python3.12 -m pip install -e ".[dev]"
python3.12 -m pytest -q          # expect: 296 passed
```

### 1. Real baseline — which synthesizer wins per domain? (~30 sec)

```bash
python3.12 scripts/run_dp_real_integration.py
```

Shows the compliance-tier table with real measured F1 on Adult Income, Credit-G, and Diabetes PIMA. TVAE retains 94.3% on HR; CTGAN 98.3% on financial — the utility ceiling before any DP is applied.

### 2. Privacy-utility Pareto curves — now from real DP-SGD training (~5 sec to reload committed results)

```bash
python3.12 scripts/run_epsilon_sweep.py
```

Loads the real, pilot-scale Opacus DP-SGD sweep (`results/real_dp_sweep.json`) per domain/ε. Financial transactions is the most DP-robust domain measured (91% retention at ε=2) — the opposite of the original per-domain sensitivity multiplier hypothesis, which predicted financial data would degrade fastest. HR and healthcare show much lower, noisier retention. See paper/draft.md Section 5.1.1 for why this contradicts the earlier simulated narrative and what it does/doesn't tell us given the small pilot scale.

### 3. Model collapse — what happens after 5 generations? (~10 sec)

```bash
python3.12 scripts/run_collapse_study.py
```

At a 30% iterative reuse rate, tail entropy hits critical threshold by generation 5; fraud/security records fall below 0.5% by generation 7. Real-data anchoring and diversity-rewarded sampling both keep tail diversity within 10% of baseline indefinitely.

### 4. (Optional) Real DP training via Opacus-backed DPVAE

```bash
# Real run — DP-SGD with Opacus privacy accounting.
# Writes to results/real_dp_sweep.json — never overwrites epsilon_sweep.json.
python3.12 scripts/run_opacus_dp_sweep.py --synthesizer DPVAE \
    --json > results/real_dp_sweep.json

# No time? Offline fallback (calibrated simulation, identical interface):
python3.12 scripts/run_epsilon_sweep.py
```

**If something fails:**

| Problem | Fix |
|---|---|
| `No module named 'sdv'` | `python3.12 -m pip install sdv` — must use Python 3.12 specifically |
| `No module named 'opacus'` | Skip Step 4; use `run_epsilon_sweep.py` fallback. Steps 1–3 fully offline |
| `results/pareto_study.json` empty | `python3.12 scripts/run_pareto_study.py > results/pareto_study.json` |

**Never overwrite committed data** — any real DP run writes to `results/real_dp_sweep.json`, never to `results/epsilon_sweep.json`. Steps 1–3 read only committed JSON files and reproduce identically on every machine.

---

## Output Format

Each row in `results/epsilon_sweep.json` contains:

| Field | Type | Description |
| --- | --- | --- |
| `epsilon` | `float` | DP budget ε |
| `delta` | `float` | Fixed δ=1e-5 throughout |
| `asset_type` | `str` | Domain name (e.g. `financial_transactions`) |
| `privacy_score` | `float` | 1 − AUC_MIA (higher = stronger privacy) |
| `utility_score` | `float` | Mean TSTR F1 across seeds |
| `tstr_ci_lower` | `float` | 95% bootstrap CI lower bound |
| `tstr_ci_upper` | `float` | 95% bootstrap CI upper bound |
| `fidelity_score` | `float` | Composite fidelity (Wasserstein + correlation) |
| `compliance_tier` | `str` | `strict` / `balanced` / `utility_focused` |
| `below_utility_cliff` | `bool` | True if utility < 90% of no-DP baseline |
| `domain_sensitivity_multiplier` | `float` | Per-domain DP sensitivity (1.0–1.70) |

---

## Note on Scope

The `src/enterprisesynth/` module contains an OpenAPI/SFT trace generation prototype from an earlier phase of this project. The primary research contribution — the DP benchmark — lives in `src/privacy_benchmark/`, `src/consistency/`, and `src/tstr_eval/`. Both threads are described in [DESIGN_DOC.md](DESIGN_DOC.md).

---

## Target Venues

- MLinPL 2026 — deadline Aug 1, 2026
- AAAI 2027 Workshop on Enterprise AI Evaluation — deadline Jul 28, 2026

---

## Citation

```bibtex
@misc{enterprisesynth2026,
  title        = {EnterpriseSynth: A Benchmark for Differentially Private Synthetic Data
                  in Regulated Enterprise Domains},
  author       = {Thimmaraju, Rashmi},
  year         = {2026},
  howpublished = {\url{https://github.com/anote-ai/Research-EnterpriseSynth}},
  note         = {Preprint}
}
```
