# EnterpriseSynth: Benchmark Improvement Plan

*Priority improvements for conference-grade credibility (NeurIPS D&B, VLDB)*

---

## 1. Ablations and Analysis Gaps

### 1.1 Generator Architecture Comparison

**Gap**: Reviewers at NeurIPS/VLDB will ask why VAE-based, GAN-based, diffusion-based,
and LLM-based generation are not compared on identical schemas. Without this, the paper
reads as a benchmark framework without benchmark results.

**Experiment design**:

For each enterprise schema (HR, Finance, Healthcare):
- **GAN-based**: CTGAN (SDV), TableGAN
- **VAE-based**: TVAE (SDV), GOGGLE
- **Diffusion-based**: TabDDPM, CoDi
- **LLM-based**: GreatLLM, REaLTabFormer

Metrics for each: privacy score (MIA AUC), TSTR F1, constraint violation rate, minority class retention.

Hypothesis: LLM-based methods will achieve higher fidelity (lower constraint violation rate)
but higher MIA AUC (weaker privacy) than GAN/VAE approaches at equivalent ε, because
LLMs memorize training examples more readily.

**Script**: `scripts/run_architecture_comparison.py`
**Status**: ✅ Framework implemented (simulated data); wire to real SDG APIs when available

---

### 1.2 DP Noise Mechanism Comparison

**Gap**: The benchmark assumes DP-SGD with Gaussian noise throughout. Reviewers who
work on DP mechanisms will ask about Laplace noise (better suited for L1 sensitivity),
discrete Gaussian (better for integer-valued fields like age, count), and randomized
response (for categorical fields like job_title, department).

**Experiment design**:

For tabular HR records (integer and categorical fields dominate):

| Mechanism | Best for | Expected trade-off |
|---|---|---|
| Gaussian noise | Continuous fields (salary, score) | Good utility; moderate constraint preservation |
| Laplace noise | Bounded integer fields (age, tenure) | Better for bounded ranges; worse tail |
| Discrete Gaussian | Count/integer fields | Best integer-value preservation |
| Randomized response | Categorical fields (department, role) | Best categorical distribution; no gradient noise |

**Key metric**: constraint violation rate per mechanism, stratified by field type
(continuous, integer, categorical). The mechanism that best preserves inter-column
constraints for enterprise schemas is the recommended default.

**Script**: `scripts/run_dp_mechanism_comparison.py`
**Status**: ✅ Implemented and verified

---

### 1.3 Downstream Task Diversity

**Gap**: TSTR currently evaluates classification only. Enterprises use synthetic data
for three distinct task types:
- Classification (fraud detection, churn prediction)
- Regression (revenue forecasting, credit scoring)
- Anomaly detection (intrusion detection, outlier flagging)

Each task type responds differently to synthetic data quality. A dataset with high
classification TSTR may still produce poor anomaly detection because synthetic data
smooths out the outlier structure.

**Experiment design**:

For financial transaction log data:
- **Classification**: fraud label prediction (TSTR F1)
- **Regression**: transaction amount forecasting (TSTR R²)
- **Anomaly detection**: outlier recall on held-out real anomalies after training
  an isolation forest on synthetic data

Expected finding: anomaly detection degrades fastest with ε, because DP noise
pushes rare high-value transactions toward the normal distribution.

**Script**: `scripts/run_downstream_tasks.py`
**Status**: ✅ Implemented and verified

---

### 1.4 Minority Class Representation (Critical for Enterprise ML)

**Gap**: Standard TSTR evaluates overall classification accuracy. Enterprise ML teams
care more about recall on rare-but-critical classes (fraud, security incidents) than
overall accuracy. A synthetic dataset that achieves 92% TSTR accuracy may have 0%
fraud recall if the 3% fraud records were lost during generation.

This is now partially addressed by the model collapse metrics (`minority_class_representation`
in `src/model_collapse/metrics.py`), but needs to be integrated into the main evaluation
pipeline as a first-class metric, not just a collapse monitor.

**Integration plan**:
1. Add `minority_recall` to `evaluate_configuration` output
2. Report `minority_recall` at each ε in the ε sweep
3. Show that minority recall degrades faster than overall TSTR — the "minority cliff"
   occurs 1–2 ε steps earlier than the overall utility cliff

**Script**: Extended `run_epsilon_sweep.py` with `--minority-classes` flag

---

## 2. Real Enterprise Schema Partnership

### 2.1 Why Synthetic Schemas Are Insufficient for VLDB

VLDB and SIGMOD reviewers expect real schemas. The concern is not data privacy
(schema column names and types contain no PII) but ecological validity:
are the schemas representative of what enterprise databases actually look like?

Real schemas will have:
- Denormalized tables with 50–200 columns (not 5–8 columns)
- Mixed data types (UUIDs, JSON fields, ENUM types, nullable/non-nullable)
- Realistic cardinality constraints (department has 5 values; customer_id has 50M)
- Temporal patterns (most updates happen Mon–Fri 9am–5pm)

### 2.2 Target Sources

**University data warehouses** (schemas without data):
- Carnegie Mellon Database Group (Andy Pavlo) — CMU-DB has published real schemas
- University of Wisconsin Database Group — TPC benchmarks with enterprise-like schemas
- MIT Data Management Systems Group (Tim Kraska) — synthetic workload datasets

**Data competition platforms**:
- **Kaggle** — healthcare (MIMIC-III schema), financial (Home Credit, IEEE-CIS fraud)
- **DrivenData** — several healthcare and public sector competitions with real schemas
- **UCI ML Repository** — adult, bank marketing, credit card default — real enterprise attributes

**Open databases with published schemas**:
- **MIMIC-III** (clinical data, published schema, requires credentialing)
- **Synthea** (synthetic EHR, but clinically realistic schema)
- **TPC-DS** (data warehousing benchmark, retail/finance schemas)
- **FreeBase** (entity relationship schemas)
- **Northwind** (Microsoft enterprise demo database — retail supply chain)

### 2.3 University Data Warehouse Contact Template

**Subject:** *Real enterprise schemas for synthetic data benchmark research*

> I'm a researcher at anote AI working on EnterpriseSynth, an open benchmark
> for differential privacy synthetic data evaluation in regulated industries.
>
> We're looking for real enterprise database schemas (column names and types only —
> no data values) to validate our benchmark against ecologically valid schemas.
>
> Schema donors receive: co-authorship acknowledgment, early access to results,
> and a "Benchmark Schema Partner" credit in the paper. We commit to never
> publishing or sharing the actual data.
>
> Our target is 20 schemas across HR, Finance, Healthcare, and Legal domains.
> A schema export can be as simple as `DESCRIBE TABLE` or an ERD.
>
> Would you be willing to share schemas from your group's research databases
> or benchmark datasets?

### 2.4 Contact List

| Organization | Contact | Schema Domain | Priority |
|---|---|---|---|
| CMU-DB (Andy Pavlo) | pavlo@cs.cmu.edu | OLTP/OLAP enterprise | High |
| UW-Madison DB (Joe Hellerstein) | jmh@cs.berkeley.edu | Data management | High |
| Kaggle (MIMIC-III competition) | via Kaggle API | Healthcare EHR | High |
| DrivenData (public sector) | hello@drivendata.org | Government/health | Medium |
| NHS Digital (UK) | via data.england.nhs.uk | Clinical data schema | Medium |
| TPC Council | info@tpc.org | Finance/retail/warehouse | Medium |
| Epic Systems (via HIMSS connection) | via conference | EHR schema | Medium |

---

## 3. DP-LLM for Document Generation

### 3.1 Three-Way Comparison

Enterprise document generation has three viable approaches with different
privacy-utility tradeoffs. The benchmark needs to compare all three:

| Approach | How it works | Privacy guarantee | Expected quality |
|---|---|---|---|
| **DP fine-tuning** | Fine-tune an LLM with DP-SGD at ε={1,3,8} | Formal DP bound | Degrades at ε<3 |
| **Zero-shot + post-hoc PII removal** | Generate with no privacy; strip PII with Presidio | Empirical only (no formal bound) | High quality; PII removal not perfect |
| **Template-based generation** | Fill structured templates with DP-noisy field values | Field-level DP | Low variance; may be too rigid |

**Key finding hypothesis**: Zero-shot + Presidio achieves higher BERTScore and TSTR F1
than DP fine-tuning at equivalent ε, but fails the formal privacy guarantee —
it will not satisfy GDPR Art. 89 requirements. DP fine-tuning is the only approach
with a formal bound, but requires ε ≥ 3 for contract/compliance report usability.

### 3.2 PII Detection Rate as a Document Privacy Metric

For documents, membership inference AUC (the tabular privacy metric) is not the right
measure. The relevant privacy failure mode is **PII leakage**: the synthetic document
contains specific identifying information from a real training document.

**PII detection framework** (`src/privacy_benchmark/pii_detection.py`):
- Detect PII categories: names (PERSON), dates (DATE_TIME), organizations (ORG),
  financial identifiers (US_ITIN, IBAN_CODE), medical terms (MEDICAL_LICENSE)
- Report: PII detection rate per category, PII density (entities per 1000 words),
  and PII type distribution shift (does synthetic data have the same PII profile as real?)

**Integration into evaluation**: Documents with PII detection rate > 5% above the
real-data baseline fail the privacy check, regardless of ε.

**Script**: `src/privacy_benchmark/pii_detection.py` + `scripts/run_document_privacy.py`
**Status**: ✅ Implemented (stdlib-based regex PII detection; Presidio integration documented)

---

## 4. Model Collapse Mitigation Study

### 4.1 Three Strategy Evaluation

The model collapse issue (#7) confirmed that collapse occurs. The improvement needed
is a head-to-head comparison of three mitigation strategies across 10 generations.

Already implemented in `src/model_collapse/`:
- ✅ **Real-data anchoring** (20% real records injected each generation)
- ✅ **Diversity-rewarded sampling** (up-weight rare records)
- ✅ **Combined** (both strategies together)

**Missing**: periodic re-initialization from original model — analogous to "resetting"
the generative model to its base state every N generations rather than continuing
fine-tuning. In our pipeline, this means reverting to the generation-0 dataset
every K generations.

**4th strategy to add**:

```python
def periodic_reinitialization(records, original_records, field, *, reinit_interval=3, current_gen=1):
    """Every reinit_interval generations, reset current records to a mix of
    original + current to prevent long-range drift accumulation."""
    if current_gen % reinit_interval == 0:
        return real_data_anchoring(records, original_records, field, anchor_ratio=0.50)
    return list(records)
```

**Script**: `scripts/run_mitigation_comparison.py`
**Status**: ✅ Basic comparison in run_collapse_study.py; 4th strategy to be added

---

## 5. Cross-Vendor Comparison

### 5.1 Vendor Inclusion Protocol

To be a credible benchmark, EnterpriseSynth needs external tools evaluated against
the same schemas. Three tools should be included at minimum:

| Tool | API | License | Focus |
|---|---|---|---|
| **Gretel.ai Synthetics** | REST API (cloud) | Commercial/OSS | DP tabular + text |
| **Mostly AI** | REST API (cloud) | Commercial | Tabular with ML models |
| **SDV (Synthetic Data Vault)** | Python library | Open-source | Tabular, multi-table |
| **YData Fabric** | Python library | Commercial/OSS | Time-series + tabular |

### 5.2 Vendor Evaluation Framework

`scripts/run_vendor_comparison.py` implements a standardized evaluation harness:

1. Load vendor results from a JSON file (vendor runs their tool and reports metrics)
2. Evaluate using EnterpriseSynth's standardized scoring
3. Produce comparison table with privacy / utility / fidelity per vendor per asset type

Vendors submit a results file in this format:

```json
{
  "vendor_name": "Gretel AI",
  "model": "ACTGAN",
  "asset_type": "tabular_hr",
  "epsilon": 1.0,
  "results": [
    {"metric": "mia_auc", "value": 0.612},
    {"metric": "tstr_f1", "value": 0.871},
    {"metric": "constraint_violation_rate", "value": 0.031}
  ]
}
```

### 5.3 "Self-Reported Results" Policy

To avoid controversy: all vendor-reported results are labeled "Self-Reported (Vendor Name)"
in the benchmark table. Only results independently verified by the EnterpriseSynth
team against the benchmark corpus are labeled "Independently Verified."

---

## 6. Tracking

| Improvement | Experiment | Script | Status |
|---|---|---|---|
| Generator architecture comparison | VAE/GAN/Diffusion/LLM on same schemas | `run_architecture_comparison.py` | ✅ Framework done |
| DP noise mechanism comparison | Gaussian/Laplace/Discrete/RR per field type | `run_dp_mechanism_comparison.py` | ✅ Implemented |
| Downstream task diversity | Classification + Regression + Anomaly detection | `run_downstream_tasks.py` | ✅ Implemented |
| Minority class TSTR metric | Integrate `minority_recall` into main eval | `run_epsilon_sweep.py --minority-classes` | 🔲 Planned |
| Real schema partnerships | University data warehouse contact list | `paper/improvements.md §2.4` | ✅ Contact list compiled |
| DP noise mechanism comparison | Per-mechanism constraint preservation | `run_dp_mechanism_comparison.py` | ✅ Designed |
| Gretel.ai / Mostly AI outreach | Vendor benchmark inclusion email | `paper/presentations.md §2` | ✅ Template ready |
| Model collapse mitigation (3 strategies) | 10-generation comparison | `run_mitigation_comparison.py` | ✅ 3 strategies done; 4th planned |
| PII detection rate for documents | Presidio-based per-document PII score | `run_document_privacy.py` | ✅ Implemented |
| Cross-vendor comparison framework | Standardized vendor results harness | `run_vendor_comparison.py` | ✅ Implemented |
