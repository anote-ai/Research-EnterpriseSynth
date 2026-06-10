# EnterpriseSynth → Anote Synthetic-Data: Product Integration Plan

*Maps every research finding to a concrete product feature or investigation.*

---

## Overview: Research ↔ Product Mapping

| Research Finding | Product Gap | Required Action | Priority |
|---|---|---|---|
| Constraint violation rate > 2% on enterprise schemas | Product generates invalid rows | Add constraint validator layer to generation pipeline | P0 |
| MIA AUC > 0.6 possible at default ε | Product privacy claims may be weaker than stated | Add MIA audit to ε selection UI + reporting | P0 |
| TSTR utility gap > 5% vs. real data | No utility measurement in product today | Add TSTR measurement as a quality gate on generated datasets | P1 |
| Model collapse within 5 iterative cycles | Customers doing iterative workflows have no warning | Add tail entropy monitoring + re-anchor recommendation to product | P1 |

---

## Investigation 1: Constraint Violation Rate Audit

### Methodology

**Schemas to test** (5 representative enterprise types):

| Schema | Key constraints | Violation threshold |
|---|---|---|
| HR records | hire_date ≤ termination_date; salary ≥ 0; age ↔ birth_year ±1 | < 2% |
| Financial transactions | amount ≥ 0; transaction_date order; debit ≠ credit | < 2% |
| Customer / CRM | account creation ≤ first purchase; loyalty_points ≥ 0 | < 2% |
| Inventory | quantity_on_hand ≥ 0; reorder_point ≤ max_stock | < 2% |
| Event logs | event_start ≤ event_end; session_id foreign key integrity | < 2% |

**Protocol**:
1. Take each of the 5 customer schema types
2. Generate 1,000 synthetic rows using the current Anote Synthetic-Data product
3. Run automated constraint checker (see `scripts/run_product_audit.py`)
4. For violations that pass automated checks, have a domain expert review a random 100-row sample

**Pass / Fail threshold**: < 2% violation rate per schema.
If any schema exceeds 2%, the product needs post-generation constraint enforcement.

**Automated checker** (from `src/consistency/`):
```python
from consistency.evaluator import evaluate_dataset
from consistency.schemas import HR_SCHEMA

# Replace with actual product-generated data
result = evaluate_dataset(product_generated_rows, HR_SCHEMA)
print(f"Violation rate: {result['violation_rate']:.2%}")
# FAIL if result['violation_rate'] > 0.02
```

**Recommended fix if FAIL**: Add a post-generation filtering step that rejects any
row violating defined constraints, then resamples until n_valid = n_requested.
For non-trivially constrained schemas (e.g., hire/termination date ordering), a
constraint-aware generator (CTGAN with constraint module) may be required.

---

## Investigation 2: DP Implementation Audit

### Methodology

**Goal**: Verify that Anote Synthetic-Data's privacy guarantee matches what the product UI claims.
The theoretical ε is necessary but not sufficient — empirical MIA AUC may exceed 0.6 even
at ε values the UI labels as "HIPAA-compliant."

**Protocol**:
1. Generate 1,000 synthetic rows from a 1,000-row real dataset (50/50 train/test split)
2. Run a shadow-model membership inference attack:
   - Train 50 shadow models, each on a bootstrap of the training data
   - For each shadow model, generate a shadow synthetic dataset
   - Train a binary classifier (MIA classifier): given a synthetic row, predict training/non-training
   - Evaluate MIA classifier AUC on the real test set
3. Compare observed MIA AUC vs. the benchmark's tier thresholds

**Pass / Fail thresholds** (from benchmark, `src/privacy_benchmark/`):

| Compliance claim | Max acceptable MIA AUC | Action if exceeded |
|---|---|---|
| GDPR research (strict tier) | 0.55 | Drop ε to 0.5; warn users |
| HIPAA (balanced tier) | 0.67 | Drop ε to 2; update UI labeling |
| SOX/internal (utility tier) | 0.82 | No action required |

**Lightweight proxy** (without shadow models — for rapid CI check):
```python
from privacy_benchmark.evaluator import evaluate_configuration

# If using reported/estimated AUC from your privacy accountant:
result = evaluate_configuration(
    epsilon=current_product_epsilon,
    auc=estimated_mia_auc,
    tstr_score=0.90,  # placeholder
    fidelity=0.88,    # placeholder
)
print(f"Privacy score: {result['privacy_score']:.3f}")
# Flag if result['auc'] > 0.67 for HIPAA context
```

**ε calibration update**: If MIA AUC exceeds thresholds, update the default ε
in the product UI per compliance context:

| Context | Current product default (estimate) | Research recommendation |
|---|---|---|
| "Healthcare / HIPAA" | Unknown | ε = 1–2 |
| "Finance / SOX" | Unknown | ε = 5–10 |
| "GDPR / Research" | Unknown | ε = 0.5 |
| "Internal / analytics" | Unknown | ε = 5–10 or no-DP |

**Multi-seed validation**: DP training variance is highest at small ε. Run the
product at ε ≤ 1 with ≥ 5 seeds; report mean ± std MIA AUC, not just a point estimate.

---

## Investigation 3: TSTR Utility Measurement

### Methodology

**Goal**: Measure how much downstream ML task quality degrades when training on
Anote-generated synthetic data vs. real data.

**Three representative customer tasks**:

| Task | Dataset type | Metric | Pass threshold |
|---|---|---|---|
| Fraud detection | Financial transactions | F1 on fraud class | TSTR gap < 5% |
| Churn prediction | CRM / customer records | AUC-ROC | TSTR gap < 10% |
| Anomaly detection | Transaction/event logs | Outlier recall | TSTR gap < 15% |

*Note: anomaly detection has a wider threshold because DP noise affects outlier structure most
(per `scripts/run_downstream_tasks.py` findings).*

**Protocol**:
```
Real data (train split)  →  Anote product  →  Synthetic dataset
                                   ↓
                             Train ML model on synthetic
                                   ↓
                             Evaluate on real test split
                                   ↓
                             Report TSTR F1 / AUC / Recall
```

Compare vs. a model trained directly on the real train split (the TSTR baseline).

**Key risk to flag**: If anomaly detection TSTR gap > 15%, customers using Anote
synthetic data for fraud model training are getting significantly degraded models.
This requires either a higher ε recommendation for fraud use cases or explicit
minority-class preservation in the generation pipeline.

**Running the measurement** (from benchmark):
```python
from tstr_eval.evaluator import evaluate_document_category
# For tabular: use run_downstream_tasks.py protocol
# TSTR gap = (real_model_metric - synthetic_model_metric) / real_model_metric
```

---

## Investigation 4: Model Collapse in Iterative Workflows

### Methodology

**Goal**: Determine if customers who use Anote Synthetic-Data in iterative retraining
workflows are experiencing model collapse. This is the highest-severity finding from
the benchmark because it causes silent degradation with no visible warning in standard
product metrics.

**Protocol** (5-generation iterative pipeline):
1. Generate 500 synthetic rows from real data using Anote product
2. Train a lightweight ML model on synthetic rows
3. Score unlabeled data using the model → generate "annotated" synthetic data
4. Use scored data as the new training set for the next generation
5. Repeat 5 times
6. Track tail class frequency (rarest 10% of label values) at each generation

**Pass / Fail threshold**: < 20% drop in tail class frequency after 3 iterations.
If tail class frequency drops > 20%, the workflow needs a re-anchoring intervention.

**Monitoring code** (from benchmark):
```python
from model_collapse.metrics import tail_coverage_entropy, minority_class_representation, entropy_within_tolerance

# Check at each generation before proceeding
tail_h = tail_coverage_entropy(current_dataset, field="label")
if not entropy_within_tolerance(tail_h, original_tail_h, tolerance=0.20):
    print("⚠️  Model collapse warning: tail class frequency dropped >20%")
    print("   Recommendation: re-anchor with original real data before next iteration")
    # Apply mitigation:
    from model_collapse.mitigation import mitigated_pipeline_step
    current_dataset = mitigated_pipeline_step(
        current_dataset, original_real_data, field="label", strategy="both"
    )
```

**Expected finding** (from issue #7 results):
- At 20% per-generation collapse rate: warning threshold crossed at generation 3
- At 30% per-generation collapse rate: warning threshold crossed at generation 2

Most Anote iterative workflows likely fall in the 15–25% drift rate range,
meaning collapse becomes significant within 3–4 iterations.

**Product action if collapse detected**: Add a "Re-anchor with real data" step
to iterative workflow documentation and optionally to the product UI as a
configurable option (anchor_ratio parameter, default 20%).

---

## Recommended Product Feature Additions

### P0: Constraint Validator

**Where**: Post-generation pipeline, before synthetic data is delivered to user

**How**:
1. Accept user-defined constraints (via UI: "hire_date must be before termination_date")
2. After generation, run constraint checker
3. Replace violating rows by resampling from valid rows
4. Report: "Generated 1,000 valid rows (47 rows replaced due to constraint violations)"

**Implementation pointer**: `src/consistency/rules.py` + `src/consistency/evaluator.py`

---

### P0: DP Calibration UI Update

**Where**: Privacy settings screen in Synthetic-Data product

**How**:
1. Show compliance tier alongside the ε slider: "HIPAA → recommended ε = 1–2"
2. Show expected utility retention: "At ε=1, expect 88% of no-DP model quality"
3. Warn when ε is set below the compliance tier's recommendation

**Implementation pointer**: `src/privacy_benchmark/config.py` (COMPLIANCE_TIERS, EPSILON_VALUES)

---

### P1: TSTR Quality Gate

**Where**: Dataset quality report, generated after each synthetic dataset export

**How**:
1. Run a 3-metric quality check: constraint violation rate, TSTR F1 (proxy), fidelity score
2. Show as a "Synthetic Data Quality Score" badge in the product
3. Alert if TSTR gap > 10% vs. the expected baseline for the user's ε setting

---

### P1: Iterative Workflow Collapse Monitor

**Where**: New "Iterative Workflows" section of product documentation + optional UI feature

**How**:
1. If user is generating multiple rounds of synthetic data, compute tail entropy at each round
2. Display a "Data Diversity Health" indicator: Green (≥90% of baseline) / Yellow (75–90%) / Red (<75%)
3. Trigger re-anchor recommendation when Yellow threshold crossed

**Implementation pointer**: `src/model_collapse/metrics.py`, `src/model_collapse/mitigation.py`

---

## Handoff: Research → Product Team

The benchmark infrastructure is production-ready for evaluation purposes:

| Research artifact | Product usage |
|---|---|
| `src/consistency/` | Drop-in constraint validator (add rules per schema) |
| `src/privacy_benchmark/config.py` | Copy COMPLIANCE_TIERS to product config |
| `src/privacy_benchmark/evaluator.py` | Use `evaluate_configuration` for ε calibration UI |
| `src/model_collapse/metrics.py` | Use `tail_coverage_entropy` as the monitoring metric |
| `src/model_collapse/mitigation.py` | Use `mitigated_pipeline_step` for re-anchoring |
| `scripts/run_product_audit.py` | Run as part of product QA pipeline |

All modules are pure Python (no external ML dependencies) and tested on Python 3.10–3.11.
