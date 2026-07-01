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
| ε sweep — Pareto curves | Calibrated simulation; domain-varying curves | `python scripts/run_epsilon_sweep.py` |
| DP real integration | Oracle + no-DP TSTR measured; DP F1 estimated | `python scripts/run_dp_real_integration.py` |
| Downstream task diversity | Classification F1 measured; regression/anomaly estimated | `python scripts/run_downstream_tasks.py` |
| Model collapse study | Measured in controlled simulation pipeline | `python scripts/run_collapse_study.py` |
| DP mechanism comparison | Gaussian / Laplace / Discrete run directly | `python scripts/run_dp_mechanism_comparison.py` |
| Real DP-SGD sweep | Pending — requires `pip install opacus torch` | `python scripts/run_opacus_dp_sweep.py` |

Useful repository documents:

- [DESIGN_DOC.md](DESIGN_DOC.md)
- [RESEARCH_STATUS.md](RESEARCH_STATUS.md) — full real vs. simulated provenance for every result file
- [paper/draft.md](paper/draft.md) — full paper draft (Abstract through Appendices)

---

## Current In-Repo Findings

These are the current repo-verified findings, not a claim that every paper-ready external measurement has been finalized:

**Real baseline (measured):** TVAE retains 94.3% of oracle F1 on HR data; CTGAN retains 98.3% on financial data; GaussianCopula retains 91.4% on healthcare EHR. These establish the utility ceiling before DP is applied.

**ε sweep (calibrated simulation, domain-varying):** At ε=2 (HIPAA-compatible tier, δ=1e-5), estimated DP utility retention is 79–81% across tabular enterprise domains. Utility cliff (below 90% baseline) persists for all domains at ε ≤ 5. Financial transactions degrade fastest — need ε=5 to match HR utility at ε=2.

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

For the ε sweep and Table 2 DP columns, the default scripts produce **calibrated simulation results**. These are directionally sound, domain-varying, and reproducible, but they should not be cited as measured DP-SGD training numbers until `scripts/run_opacus_dp_sweep.py` has been run and `results/real_dp_sweep.json` committed. See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for the full accounting.

---

## Benchmark Pipeline

```
Real Enterprise Dataset (HR / EHR / Financial / IoT / E-commerce)
        |
        v
  [DP-SGD Synthesizer]  ←── ε, δ=1e-5 budget
  (CTGAN / TVAE / GaussianCopula + Opacus)
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

# Real DP-SGD sweep via Opacus — requires pip install opacus torch
python scripts/run_opacus_dp_sweep.py --dry-run   # check deps
python scripts/run_opacus_dp_sweep.py             # full run (~2h)
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
