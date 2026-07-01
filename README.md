# EnterpriseSynth

[![CI](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml/badge.svg)](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Status

> See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for a full breakdown of which results are directly measured on real data vs. produced by calibrated simulation models.

**EnterpriseSynth** is a benchmark for differentially private synthetic tabular data generation in regulated enterprise domains. It measures the Privacy × Utility × Fidelity tradeoff across six DP budgets (ε ∈ {0.1, 0.5, 1, 2, 5, 10}, δ=1e-5) and six regulated data domains (HR, Healthcare EHR, Financial, IoT, E-commerce, Legal/DevOps).

**Key findings:**

- At ε=2, δ=1e-5, DP synthetic data retains 79–81% of real-data oracle F1 across tabular enterprise domains
- Financial time-series degrades fastest under DP (needs ε=5 to match HR utility at ε=2)
- Model collapse drops tail-record diversity 51% by generation 5 without mitigation; diversity-rewarded sampling keeps it within 10% of generation-0

**Note on scope:** The `src/enterprisesynth/` module includes an OpenAPI/SFT trace generation prototype from an earlier phase of the project. The primary research contribution — the DP benchmark — lives in `src/privacy_benchmark/`, `src/consistency/`, and `src/tstr_eval/`. See [DESIGN_DOC.md](DESIGN_DOC.md) and [paper/draft.md](paper/draft.md) for the full research framing.

## Research Significance

EnterpriseSynth is the first benchmark to simultaneously measure Privacy × Utility × Fidelity for differentially private synthetic data across six regulated enterprise domains. Prior work (SDGym, CTAB-GAN+) evaluates on generic tabular datasets without enterprise constraint verification; EnterpriseSynth adds constraint violation rate, domain-specific sensitivity multipliers, and a multi-generation model collapse study.

The framework is especially relevant for:

- Enterprise data teams evaluating DP synthesizers for GDPR/HIPAA/SOX compliance
- ML researchers studying the privacy-utility tradeoff in structured tabular data
- Compliance engineers mapping ε budgets to regulatory tiers
- Practitioners building synthetic data pipelines that require iterative retraining safety

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

## Quick Start

```bash
pip install -e ".[dev]"

# Run the full ε sweep across all three domains
python scripts/run_epsilon_sweep.py

# Downstream task diversity (classification / regression / anomaly detection)
python scripts/run_downstream_tasks.py

# Real-data baselines on public datasets (Adult Income, Credit-G, Diabetes PIMA)
python scripts/run_dp_real_integration.py
```

```python
from privacy_benchmark.evaluator import evaluate_with_ci
from privacy_benchmark.domains import get_domain

domain = get_domain("financial_transactions")

# Evaluate one DP configuration (ε=2, δ=1e-5)
result = evaluate_with_ci(
    epsilon=2.0,
    auc=0.58,           # MIA AUC from shadow-model attack
    tstr_scores=[0.61, 0.63, 0.60, 0.62, 0.64],  # per-seed TSTR F1
    fidelity=0.81,
)
print(result)
# {
#   "epsilon": 2.0,
#   "privacy_score": 0.42,   # 1 − AUC_MIA
#   "utility_score": 0.62,   # mean TSTR F1
#   "fidelity_score": 0.81,
#   "tstr_ci_lower": 0.595,
#   "tstr_ci_upper": 0.645,
#   ...
# }
```

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

## Logical Consistency Benchmarking

The `src/consistency/` module evaluates inter-column logical consistency in synthetic enterprise tabular data. Constraint violation rate is one of the three primary fidelity metrics.

Example constraints checked:

- `hire_date <= termination_date`
- `salary >= 0`
- `age` consistent with `birth_year`

Violations spike below ε=3 in relational schemas (FK constraint breaks from DP noise) — see `results/product_audit.json`.

## Target Venues

- MLinPL 2026 (Machine Learning in Poland Conference) — deadline Aug 1, 2026
- AAAI 2027 Workshop on Enterprise AI Evaluation — deadline Jul 28, 2026

## Citation

```bibtex
@misc{enterprisesynth2026,
  title        = {EnterpriseSynth: A Benchmark for Differentially Private Synthetic Data
                  in Regulated Enterprise Domains},
  author       = {Anote AI Research},
  year         = {2026},
  howpublished = {\url{https://github.com/anote-ai/Research-EnterpriseSynth}},
  note         = {Preprint}
}
```
