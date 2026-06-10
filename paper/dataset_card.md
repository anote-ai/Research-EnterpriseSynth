# EnterpriseSynth Dataset Card

**HuggingFace repository**: `anote-ai/enterprisesynth`
**Version**: 1.0.0
**License**: Apache 2.0
**Task category**: Synthetic data evaluation / differential privacy benchmarking

---

## Dataset Description

EnterpriseSynth is a benchmark dataset for evaluating differential privacy (DP) synthetic data
generation across regulated enterprise data types. It contains:

1. **Reference corpora** — public-domain originals used to calibrate the benchmark
2. **Pre-generated synthetic samples** — synthetic datasets at multiple ε values (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
3. **Evaluation splits** — train/test splits for TSTR (Train-Synthetic-Test-Real) evaluation

---

## Domain Categories and Schema Types

| Domain | Schema type | Record count | Source dataset | License |
|---|---|---|---|---|
| Human Resources | Tabular (HR records) | 5,000 | Synthetic (generated from public HR statistics) | Apache 2.0 |
| Financial Transactions | Tabular (transactions) | 10,000 | UCI ML Repository — Credit Card Fraud Detection | CC BY 4.0 |
| Healthcare / EHR | Tabular (patient records) | 3,000 | MIMIC-III demo (public subset) | PhysioNet CC BY 4.0 |
| Legal / Contracts | Document (free text) | 500 | CUAD (Contract Understanding Atticus Dataset) | CC BY 4.0 |
| IT Support Tickets | Document (free text) | 1,000 | Kaggle IT Service Management dataset | CC0 Public Domain |
| Compliance Reports | Document (free text) | 300 | Synthetic (from public SEC filing templates) | Apache 2.0 |

**Critical note**: This dataset contains ONLY public-domain originals or synthetically generated
data. No proprietary enterprise data has been included. No real patient records, employee records,
or financial records from any individual company are included.

---

## Privacy-Sensitive Field Taxonomy

Fields that required DP treatment in the tabular schemas:

| Field | Schema | Sensitivity | DP mechanism used |
|---|---|---|---|
| `salary` | HR records | Continuous, high | Gaussian (σ calibrated per ε) |
| `age` / `birth_year` | HR records | Integer, medium | Discrete Gaussian |
| `account_balance` | Financial | Continuous, high | Gaussian |
| `transaction_amount` | Financial | Continuous, medium | Laplace |
| `diagnosis_code` | Healthcare | Categorical, high | Randomized Response |
| `department` | HR records | Categorical, low | Randomized Response |

For document schemas, DP is applied at the token level during LLM generation (DP-SGD).
See `paper/dp_privacy_analysis.md` for the full threat model.

---

## Evaluation Splits

Each schema type is split as follows for TSTR evaluation:

```
Real data (70% train / 30% test)
  ├── train split  →  fed to DP synthetic generator
  │     └── synthetic dataset at ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}
  └── test split   →  used as evaluation target for TSTR
```

Pre-generated synthetic samples at all ε values are included in the HuggingFace repository
so reviewers can run TSTR evaluation without re-running DP training.

---

## File Structure (HuggingFace)

```
anote-ai/enterprisesynth/
├── data/
│   ├── hr_records/
│   │   ├── real_train.csv
│   │   ├── real_test.csv
│   │   └── synthetic_eps_{0.1,0.5,1.0,2.0,5.0,10.0}.csv
│   ├── financial_transactions/
│   │   └── ...
│   ├── healthcare_ehr/
│   │   └── ...
│   └── documents/
│       ├── contracts/
│       ├── support_tickets/
│       └── compliance_reports/
├── evaluation/
│   └── tstr_baselines.json   # real-data model scores for each schema
└── README.md                 # this dataset card
```

---

## How to Load

```python
from datasets import load_dataset

# Load HR records real train split
ds = load_dataset("anote-ai/enterprisesynth", "hr_records")
real_train = ds["real_train"]

# Load synthetic HR records at epsilon=2.0
ds_synth = load_dataset("anote-ai/enterprisesynth", "hr_records_synthetic_eps2")
synthetic = ds_synth["train"]
```

---

## Intended Use and Restrictions

**Intended use**:
- Benchmarking differential privacy synthetic data tools
- Reproducing EnterpriseSynth paper results
- Evaluating TSTR utility metrics without GPU retraining

**Out-of-scope use**:
- Training production models on synthetic healthcare or financial data without further validation
- Any use that requires re-identifying individuals (not possible given DP guarantees)

**Do not submit**: Any synthetic data derived from non-public enterprise datasets to this repository.

---

## Citation

```bibtex
@dataset{enterprisesynth2026,
  title     = {EnterpriseSynth: A Benchmark for Differential Privacy Synthetic Data
               in Regulated Enterprise Settings},
  author    = {Thimmaraju, Rashmi and {Anote AI}},
  year      = {2026},
  publisher = {HuggingFace Datasets},
  url       = {https://huggingface.co/datasets/anote-ai/enterprisesynth}
}
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-Q3 | Initial release with HR, financial, healthcare tabular; contracts and support-ticket documents |
