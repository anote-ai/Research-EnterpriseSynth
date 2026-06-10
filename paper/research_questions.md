# EnterpriseSynth: Key Research Questions and Experimental Designs

Each question below maps to a concrete experiment script in `scripts/`, runnable today
against the existing benchmark infrastructure.

---

## Q1: What ε value gives acceptable utility for each compliance tier?

> *"HIPAA-level privacy (ε ≈ 3) retains 85% of utility for tabular classification tasks"*

### Why it matters

Enterprises currently choose ε by intuition or regulation-text reading.
No empirical curve exists mapping ε → expected utility loss for the specific
data types they care about (HR records, financial transactions, clinical notes).
The answer directly drives DP training configuration in production pipelines.

### Hypothesis

The privacy-utility Pareto curve is **not linear** in ε.
The largest utility drop occurs in the strict-to-balanced transition (ε: 1→0.5),
while the balanced-to-utility-focused transition (ε: 2→5) yields diminishing
privacy gains for substantial utility improvement.
The **elbow point** differs by asset type: tabular records lose utility more
steeply than documents because tabular ML tasks are more sensitive to distribution shift.

### Experimental design

**Sweep**: ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} × 3 asset types × 5 seeds

For each (ε, asset_type, seed):
1. Evaluate the DP configuration using `evaluate_multi_seed` (bootstrapped TSTR + MIA AUC)
2. Compute privacy score (1 − AUC), utility score (TSTR F1), fidelity score
3. Record mean ± std across seeds

**Metrics**:
- TSTR F1 (utility) — primary
- 1 − MIA AUC (privacy) — primary
- Pareto efficiency: (utility × privacy) / (utility_no_dp × 0.5)

**Bar / success criterion**:
- Produce elbow plots for each asset type
- Identify the ε at which utility first drops below 90% of no-DP baseline (the "utility cliff")
- Report as: *"For tabular HR data, the utility cliff is at ε ≤ X (95% CI: [a, b])"*

**Expected finding** (based on preliminary results in `src/privacy_benchmark/config.py`):
- Strict tier (ε = 0.1–0.5): utility drops 15–25%
- Balanced tier (ε = 1–2): utility drops 5–12%
- Utility-focused tier (ε = 5–10): utility drops 1–3%

**Script**: `scripts/run_epsilon_sweep.py`

---

## Q2: Does model collapse occur within 5 retraining iterations using synthetic enterprise data?

> *"X% tail coverage loss after N iterations — iterative reuse is dangerous beyond N=5"*

### Why it matters

Enterprise AI teams routinely retrain on accumulated synthetic data without checking
whether rare-but-critical records (fraud, security incidents) are being progressively
lost. A quantified collapse timeline gives MLOps teams a concrete monitoring target.

### Hypothesis

For enterprise long-tail distributions (fraud ~3%, critical_security ~1%),
**tail-record collapse is detectable within 3 generations and critical within 5**.
The signal appears in tail coverage entropy before it appears in overall coverage
entropy, making tail entropy the better early-warning metric.
Collapse rate scales super-linearly with the generation-to-generation drop rate.

### Experimental design

**10-generation pipeline** (extended from the 5-generation study in `src/model_collapse/`):

For collapse_rate ∈ {0.10, 0.20, 0.30, 0.40}:
1. Run `run_model_collapse_pipeline(n_generations=10, collapse_rate=r, seed=42)`
2. At each generation: record coverage_entropy, tail_entropy, minority_representation, tail_divergence
3. Record the **first generation** where tail_entropy drops below:
   - 90% of baseline (early warning)
   - 75% of baseline (significant collapse)
   - 50% of baseline (critical — fraud records nearly absent)

**Metrics tracked per generation**:
- Normalised Shannon entropy (full distribution)
- Tail coverage entropy (bottom 20% by frequency)
- Minority class representation: fraud + critical_security fraction
- Jensen-Shannon divergence vs. generation 0 (tail)

**Bar / success criterion**:
- "At 30% collapse rate, tail entropy crosses the 90% warning threshold at generation G₉₀"
- "At 30% collapse rate, minority class representation drops below 0.5% (from 4% baseline) at generation G₅₀"
- Produce survival curves: fraction of minority records surviving at each generation

**Key finding from `evaluator.py`** (5-generation, rate=0.30):
- Baseline: gen 5 tail entropy = 0.393 (−51%)
- The warning threshold (90%) is crossed by gen 2–3 at rate=0.30

**Script**: `scripts/run_collapse_study.py`

---

## Q3: Is inter-column constraint violation rate the strongest predictor of downstream ML utility loss?

> *"The metric with highest Spearman ρ against TSTR utility is the primary fidelity metric to optimize"*

### Why it matters

Three fidelity metrics exist — constraint violation rate, marginal distribution fidelity,
and joint distribution fidelity — but collecting all three is expensive.
If one dominates, evaluation pipelines can be simplified.
Compliance teams care most about constraint violations (they are auditable);
ML teams care most about TSTR utility. If the two align, there is one number to track.

### Hypothesis

For tabular HR records with explicit domain constraints (hire/termination dates, salary,
age/birthyear), **constraint violation rate is a stronger predictor of TSTR utility loss
than marginal distribution fidelity**, because constraint violations directly corrupt the
feature correlations that classifiers rely on.

For document assets (contracts, memos), **semantic fidelity (BERTScore) will dominate**
over structural constraint metrics because document classifiers are more sensitive to
distributional shift than to structural rule violations.

### Experimental design

**For tabular HR records**, vary synthetic data quality along three axes:

| Manipulation | Constraint Violation Rate | Marginal Fidelity | Joint Fidelity | TSTR F1 |
|---|---|---|---|---|
| Clean synthetic | Low | High | High | Baseline |
| Date-constraint violations injected | **High** | High | High | ? |
| Marginal distribution shifted | Low | **Low** | High | ? |
| Correlation structure destroyed | Low | High | **Low** | ? |
| Combined degradation | High | Low | Low | ? |

Compute Spearman ρ(constraint_violation_rate, TSTR_utility_loss)
vs. ρ(marginal_fidelity_loss, TSTR_utility_loss)
vs. ρ(JSD_joint, TSTR_utility_loss)

**For document assets**, correlate:
- BERTScore → TSTR F1 (Spearman ρ)
- MAUVE → TSTR F1 (Spearman ρ)
- NER consistency → TSTR F1 (Spearman ρ)
- Constraint violation (structural) → TSTR F1 (Spearman ρ)

**Bar / success criterion**:
- Tabular: |ρ(constraint_rate, TSTR)| > |ρ(marginal_fidelity, TSTR)|
- Document: identify the single metric with ρ > 0.8 with TSTR utility
- If confirmed: simplify the evaluation framework to report only the dominant metric

**Script**: `scripts/run_fidelity_correlation.py`

---

## Q4: Does DP fine-tuning of LLMs for document generation preserve enough semantic coherence?

> *"Identify the ε at which utility collapses for document generation (likely different from tabular)"*

### Why it matters

Document generation at strict DP budgets may be unusable even when tabular synthesis
remains acceptable at the same ε. Enterprises need to know whether their contracts,
compliance reports, and HR memos can be safely synthesized under GDPR constraints,
or whether they need non-DP methods with downstream anonymization.

### Hypothesis

The utility cliff for document generation occurs at a **higher ε than for tabular data**
(less privacy protection affordable).
Specifically:
- Contracts and compliance reports: utility cliff at ε ≈ 3–5 (structured, template-driven)
- Support tickets and HR memos: utility cliff at ε ≈ 7–10 (informal language, high variance)

**Key semantic coherence failure modes at low ε**:
1. Named entities (org names, dates, contract values) are corrupted or absent
2. Legal/regulatory terminology is replaced with semantically adjacent but incorrect terms
3. Section ordering and document structure is preserved but semantic content incoherent
4. Information extraction F1 (key clause identification) drops sharply before BERTScore does

### Experimental design

**Sweep**: ε ∈ {0.5, 1, 2, 3, 5, 7, 10} × 4 document categories × 3 generation seeds

For each (ε, document_category, seed):
1. Score BERTScore, MAUVE, NER consistency, TSTR F1 using `tstr_eval/evaluator.py`
2. Apply bootstrap CI via `stats.bootstrap_tstr_ci`
3. Track the per-metric utility cliff (first ε where score < 80% of no-DP baseline)

**Metrics**:
- **BERTScore** — semantic similarity at token level (contextual embeddings)
- **MAUVE** — distributional divergence between real and synthetic
- **NER consistency** — named entity recall against real document NER profile
- **Information extraction F1** — key clause identification (contract parties, amounts, dates)
- **TSTR F1** — downstream document classification

**Expected finding** (based on TSTR values in `tstr_eval/tstr.py`):

| Category | BERTScore (no DP) | Expected cliff ε | Rationale |
|---|---|---|---|
| Contracts | 0.91 | ε ≈ 3–5 | Structured template; DP noise corrupts specific fields |
| Compliance reports | 0.90 | ε ≈ 3–5 | Regulatory language is formulaic; cliff is sharp |
| HR memos | 0.88 | ε ≈ 7–10 | Informal language; noise is harder to distinguish from natural variation |
| Support tickets | 0.87 | ε ≈ 7–10 | Highest linguistic variance; most DP-resilient |

**Bar / success criterion**:
- Produce per-category utility cliff ε with 95% CI
- If any category cliff ε > 5: flag as "document synthesis for this category requires utility-focused tier or non-DP methods"

**Script**: `scripts/run_document_dp_sweep.py`

---

## Experiment Tracking

| # | Question | Experiment | Script | Status |
|---|---|---|---|---|
| Q1 | ε-utility tradeoff per compliance tier | ε sweep × 3 asset types × 5 seeds | `scripts/run_epsilon_sweep.py` | ✅ Designed |
| Q2 | Model collapse timeline (10 generations) | collapse_rate sweep × 10 generations | `scripts/run_collapse_study.py` | ✅ Designed |
| Q3 | Fidelity metric → utility correlation | Spearman ρ across degradation axes | `scripts/run_fidelity_correlation.py` | ✅ Designed |
| Q4 | DP LLM document utility cliff | ε sweep × 4 document categories | `scripts/run_document_dp_sweep.py` | ✅ Designed |

---

## Cross-Question Findings Expected

Running all four experiments will produce three cross-cutting results that belong in
the paper's conclusion:

1. **Dual-threshold guidance**: enterprises should set two ε values — one for tabular
   (lower, based on Q1 elbow) and one for documents (higher, based on Q4 cliff), since
   the optimal tradeoffs differ by asset type.

2. **Monitoring cadence**: Q2's collapse timeline gives MLOps teams a concrete retraining
   interval — if tail entropy drops 10% every N generations, the monitoring check should
   run every N/2 generations as a safety margin.

3. **Single-metric simplification**: Q3's dominant fidelity metric finding will either
   validate reporting constraint violation rate alone (simplifying audit workflows) or
   confirm that all three metrics are needed (justifying the full evaluation pipeline).
