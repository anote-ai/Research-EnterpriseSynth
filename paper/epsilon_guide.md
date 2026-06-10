# Choosing Your Privacy Budget: A Practical Guide for Enterprise Synthetic Data

**EnterpriseSynth Benchmark | anote AI**
*For compliance officers, data science leads, and CDOs in regulated industries*

---

## Why Your ε Choice Matters More Than You Think

If your organization generates synthetic data for AI model training, you have already
made an implicit privacy decision — even if nobody called it that. Whether you use
Gretel, Mostly AI, an in-house GAN, or any other SDG tool with differential privacy
enabled, there is a number called **ε (epsilon)** controlling the fundamental tradeoff:

> **Smaller ε = stronger privacy guarantee = more noise = lower data quality**

Most teams either skip DP entirely ("our data is already anonymized") or copy a value
from a paper without understanding what it means for their specific use case.
This guide gives you concrete numbers grounded in benchmarks across real enterprise
data types.

---

## The One Table Every Data Science Lead Needs

| Compliance Context | Recommended ε | Expected Utility Retention | What You're Protecting |
|---|---|---|---|
| GDPR Art. 89 (research/scientific) | **0.1 – 0.5** | 74 – 81% | Direct identifiers; pseudonymous research data |
| HIPAA-covered entity (PHI) | **1 – 2** | 88 – 92% | Protected Health Information; EHR records |
| GDPR standard (internal analytics) | **1 – 3** | 88 – 95% | Employee data; customer records |
| SOX audit data (internal) | **5 – 10** | 96 – 99% | Financial transaction logs; audit trails |
| Internal R&D, low-sensitivity | **10+** or no-DP | 99 – 100% | Operational data with no PII |

*Utility retention = TSTR F1 as fraction of no-DP baseline. Based on EnterpriseSynth
benchmark results across HR, financial, and healthcare data assets.*

---

## What "Utility Retention" Means in Practice

An 88% utility retention (ε = 1) means: a fraud detection classifier trained on
your ε=1 synthetic data will achieve approximately 88% of the accuracy it would
achieve if trained on a no-privacy-constraint synthetic baseline.

For most enterprise use cases, this is acceptable. For a fraud detection model where
every percentage point of F1 matters, it may not be. That is why the table gives
ranges — your tolerance for utility loss is a business decision, not a technical one.

**The utility cliff**: Our benchmark identifies a sharp degradation zone at ε ≤ 0.5
for tabular data. Moving from ε = 1 to ε = 0.5 costs roughly 7 points of F1.
Moving from ε = 0.5 to ε = 0.1 costs another 7 points. If your compliance
requirement permits ε = 1, do not use ε = 0.1 "to be safe" — you are trading
real utility for a marginal privacy improvement.

---

## Compliance Tier Deep Dive

### Tier 1: Strict (ε = 0.1 – 0.5)

**Who needs this:** Clinical trial sponsors under GDPR Art. 89, organizations with
sensitive PII (exact birth dates, diagnosis codes, SSNs) generating synthetic data
for external sharing or publication.

**What to expect:**
- Utility retention: 74–81% for tabular; 45–55% for documents at ε = 0.5
- High training variance: run at least 5 seeds and report mean ± std
- Document generation at this tier is generally not viable — semantic coherence
  collapses below ε = 2 for contracts and compliance reports
- Recommended verification: run membership inference attack after training; confirm
  AUC ≤ 0.55

**Compliance note:** ε = 1 is commonly cited as "GDPR-compatible" in the academic
literature (Dwork & Roth, 2014). The strict tier (ε ≤ 0.5) provides additional
margin for regulatory uncertainty and audit readiness.

---

### Tier 2: Balanced (ε = 1 – 2)

**Who needs this:** HIPAA-covered entities, most GDPR use cases involving employee
or customer data, organizations generating synthetic training data for internal AI
development.

**What to expect:**
- Utility retention: 88–92% for tabular; 73–77% for documents at ε = 2
- Moderate training variance: 2–3 seeds sufficient; differences are small
- Downstream ML performance: classification, regression, and forecasting tasks
  retain enough fidelity for production model training
- The Pareto-optimal tier for most enterprise use cases

**This is the recommended default.** If you are unsure which tier applies, start
at ε = 2 and move stricter only if a compliance review requires it.

---

### Tier 3: Utility-Focused (ε = 5 – 10)

**Who needs this:** SOX audit trail simulation, internal analytics on operational
data, organizations where data subjects have consented to synthetic data use.

**What to expect:**
- Utility retention: 96–99%
- Very low training variance: results are highly reproducible
- Privacy guarantee is weaker: AUC on membership inference climbs to 0.70–0.82
  at ε = 10, meaning a determined adversary with the right auxiliary information
  could distinguish some training set members

**Caveat:** At ε = 10, the DP guarantee provides weaker formal protection than at ε = 1.
If your threat model includes insider attacks or the synthetic data will be shared
externally, use the balanced tier instead.

---

## Document vs. Tabular: Different Rules Apply

A critical finding from EnterpriseSynth: **document generation requires higher ε
than tabular synthesis to achieve the same utility level.**

| Asset Type | ε for 80%+ utility | ε for 90%+ utility |
|---|---|---|
| Tabular HR / CRM records | ε ≥ 1.0 | ε ≥ 5.0 |
| Financial transaction logs | ε ≥ 1.0 | ε ≥ 5.0 |
| Healthcare EHR (tabular) | ε ≥ 1.0 | ε ≥ 5.0 |
| Legal contracts | ε ≥ 2.0 | needs ε ≥ 5–7 |
| Compliance reports | ε ≥ 2.0 | needs ε ≥ 5–7 |
| Support tickets / HR memos | ε ≥ 5.0 | ε ≥ 7–10 |

**Practical implication:** If your pipeline generates both tabular records and
associated documents (e.g., a patient EHR record + clinical notes), you may need
to use different ε values for each modality, or accept lower utility on the document side.

---

## The Model Collapse Warning

One finding that every team running iterative synthetic data retraining must understand:

> **Without mitigation, iterative synthetic data retraining causes your rare-record
> types to disappear within 5 generations.**

In a dataset with 4% fraud and 0.6% critical security incidents, our benchmark shows:

| Generations | Tail Record Entropy | Fraud + Security Records |
|---|---|---|
| 0 (baseline) | 0.854 (100%) | 4.6% of dataset |
| 2 | 0.722 (85%) | 2.9% — 37% depleted |
| 5 | 0.391 (46%) | 1.75% — 62% depleted |
| 10 | 0.000 (0%) | 0.0% — completely absent |

**Fraud detection models trained on generation-5+ synthetic data will have
near-zero fraud recall regardless of ε.** This is not a privacy issue —
it is a synthetic data generation issue that affects every team retraining on
synthetic outputs, with or without DP.

**Mitigation**: Apply diversity-rewarded sampling at each generation.
This maintains fraud/security records at 97% of baseline entropy across 10+ generations.
See `src/model_collapse/` for the implementation.

---

## Checklist: Before You Configure DP Synthetic Data

```
Before choosing ε:
  [ ] Identify the data type (tabular / document / time-series)
  [ ] Identify the regulatory context (GDPR / HIPAA / SOX / internal)
  [ ] Identify the downstream ML task and acceptable utility loss
  [ ] Check whether data contains direct identifiers → if yes, strict tier
  [ ] Check whether output will be shared externally → if yes, minimum balanced tier

When running training:
  [ ] Run ≥ 3 seeds if using ε ≤ 2; ≥ 5 seeds if using ε ≤ 0.5
  [ ] Verify ε with privacy accountant (report computed_epsilon, not just requested)
  [ ] Run membership inference attack baseline after training

If iterative retraining is planned:
  [ ] Instrument tail entropy monitoring before each generation
  [ ] Enable diversity-rewarded sampling in the generation pipeline
  [ ] Set a retraining interval budget based on collapse rate (see benchmark)
```

---

## Quick Reference Card

```
                 STRICT           BALANCED       UTILITY-FOCUSED
                 ε = 0.1–0.5      ε = 1–2        ε = 5–10
                 ───────────      ───────────    ───────────────
Utility          74–81%           88–92%         96–99%
Doc utility      45–55%           73–77%         90–96%
Privacy (AUC)    0.52–0.55        0.61–0.67      0.72–0.82
Training seeds   5+               2–3            1
Best for         GDPR research    HIPAA/GDPR     SOX/internal
                 External share   Internal AI    Analytics
```

---

*EnterpriseSynth is an open benchmark. Results, data, and code:
https://github.com/anote-ai/Research-EnterpriseSynth*

*For questions, corrections, or to include your SDG tool in the benchmark:
contact the EnterpriseSynth team via the GitHub repository.*
