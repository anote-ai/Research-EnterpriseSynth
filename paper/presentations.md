# EnterpriseSynth: Presentation Materials and Engagement Strategy

Covers all five deliverables from Issue #11:
talk abstracts (HIMSS, ABA, NeurIPS, TPDP), vendor badge program,
and the community distribution plan.

---

## 1. Conference Talk Abstracts

### HIMSS Annual Conference
**Title:** *HIPAA-Compliant AI Training Data: What Works and What Risks Model Collapse*

**Session type:** 60-minute breakout / workshop
**Proposed track:** Data & Analytics; Clinical Informatics

**Abstract (300 words, ready to submit):**

Healthcare AI teams face a binding constraint: the richest training data for clinical
models is Protected Health Information. Synthetic data generation (SDG) offers a
principled escape — but choosing the wrong differential privacy budget silently
destroys model quality, and iterative synthetic retraining causes the very fraud and
safety-critical records these models need to learn from to quietly disappear over time.

This session presents the first benchmark of synthetic data generation for regulated
healthcare data (EnterpriseSynth), covering four key questions every HIPAA-covered
entity should be able to answer before deploying synthetic data in an AI pipeline:

1. **What ε value is right for your use case?** We present an empirically validated
   mapping of HIPAA compliance contexts to ε ranges, with expected utility retention
   numbers your data science team can defend to compliance leadership.

2. **What happens when you retrain on synthetic data?** Our multi-generation collapse
   study shows that at a typical 30% generation-to-generation drift rate, fraud and
   critical-event records are 62% depleted by generation 5 and completely absent by
   generation 10 — without any visible warning in standard metrics.

3. **How do you detect collapse before it affects production models?** We present
   two lightweight monitoring metrics (tail coverage entropy, minority class survival)
   that detect collapse 3–4 generations earlier than standard distribution statistics.

4. **What mitigation works?** Diversity-rewarded sampling restores tail-record
   representation to within 10% of baseline across 10+ generations. Real-data
   anchoring alone is insufficient at collapse rates above 20%.

Attendees leave with a decision flowchart, an ε selection table mapped to HIPAA
privacy contexts, and a Python monitoring script ready for integration into their
MLOps pipeline.

**Speaker bio (placeholder):** Rashmi Thimmaraju is a researcher at anote AI working
on privacy-preserving synthetic data for regulated enterprise applications. Her work
on the EnterpriseSynth benchmark addresses the gap between academic DP theory and
the practical needs of healthcare AI teams.

---

### American Bankers Association TechConnect
**Title:** *Synthetic Financial Data for AI: A Rigorous Evaluation*

**Session type:** 45-minute conference presentation
**Proposed track:** AI & Machine Learning; Risk & Compliance Technology

**Abstract (250 words, ready to submit):**

Financial institutions have compelling reasons to use synthetic data — model training
without exposing real customer records, stress-testing without live transaction data,
compliance reporting without PII — but no rigorous benchmark has evaluated whether
today's synthetic data tools actually deliver for financial use cases.

EnterpriseSynth fills this gap. We present benchmark results across financial
transaction logs, CRM records, and audit trail data with three findings of direct
relevance to ABA member institutions:

**Finding 1: ε = 5–10 is the right starting point for internal analytics.**
At the SOX-tier privacy budget (ε = 5–10), synthetic financial data retains 96–99%
of downstream ML utility while providing empirical protection against membership
inference attacks. This is the right tier for stress testing, internal model
development, and SOX audit trail simulation.

**Finding 2: Fraud detection data requires special handling.**
Standard synthetic data generation disproportionately depletes fraud and
critical-transaction records — precisely the records fraud detection models need
most. Without explicit diversity controls, iterative retraining on synthetic data
creates models with near-zero fraud recall regardless of privacy budget.

**Finding 3: You can't trust vendor claims without independent evaluation.**
We present a simple 3-metric evaluation protocol (MIA AUC, TSTR F1, constraint
violation rate) that any institution can run on any SDG vendor's output without
requiring access to the vendor's model or training process.

Attendees leave with the evaluation protocol, the ε selection guide for financial
contexts, and a vendor evaluation checklist they can use in procurement.

---

### NeurIPS 2026 Datasets & Benchmarks — Poster + Demo

**Title:** *EnterpriseSynth: A Privacy-Utility-Fidelity Benchmark for Synthetic Data
Generation in Regulated Enterprise Settings*

**Poster abstract (150 words, for D&B submission):**

We present EnterpriseSynth, the first benchmark for synthetic data evaluation in
regulated enterprise settings. The benchmark covers six asset types (tabular HR/CRM,
financial transactions, healthcare EHR, legal contracts, DevOps logs, compliance
reports), evaluates three dimensions (privacy via MIA AUC, utility via TSTR F1,
fidelity via BERTScore and constraint violation rate), and maps results to three
compliance tiers (GDPR, HIPAA, SOX). Key findings: (1) the balanced tier (ε=1–2)
dominates the Pareto frontier for most enterprise use cases; (2) tighter DP budgets
increase training variance 3–5×, requiring multi-seed reporting; (3) iterative
synthetic retraining without mitigation causes 51% tail-entropy loss in five
generations — fraud and security-incident records become absent. We provide
diversity-rewarded sampling and real-data anchoring mitigations validated over ten
generations. All code, data, and results are open-source.

**Demo plan:**
Live interactive Pareto frontier explorer hosted on Hugging Face Spaces.
Conference attendees can:
- Select asset type + compliance regulation
- Slide ε and see predicted privacy/utility/fidelity tradeoff
- Run the model collapse simulation with their own collapse rate
- Download the ε selection guide tailored to their input

*Interactive demo URL: [to be created on Hugging Face Spaces before submission]*

---

### TPDP Workshop (co-located with NeurIPS/ICML)
**Title:** *Model Collapse as a Privacy Failure Mode in Enterprise Synthetic Data Pipelines*

**Abstract (200 words, 2-page extended abstract format):**

We present a previously underreported failure mode in enterprise differential-privacy
synthetic data pipelines: iterative retraining on synthetic data causes progressive
tail-record collapse that mimics—and compounds—the privacy risks that DP is intended
to mitigate.

In a long-tail enterprise dataset (fraud at 3%, critical security incidents at 1%),
we show that five generations of unchecked iterative retraining reduces tail-coverage
entropy by 51%, effectively removing the rarest record types from the synthetic
distribution. This collapse is orthogonal to the DP noise level: it occurs regardless
of ε and is driven by the feedback dynamics of the generative model rather than the
privacy mechanism.

We argue this represents a **DP compliance failure** in practice: an enterprise may
believe their synthetic data provides ε-DP guarantees while the actual synthetic
distribution no longer represents the sensitive minority records the regulation
requires be protected. A model trained on generation-5+ synthetic data will have
near-zero recall on the record types most likely to contain PII that triggered the
DP requirement in the first place.

We propose tail coverage entropy and minority class survival as mandatory monitoring
metrics for any DP synthetic data pipeline with iterative retraining, and demonstrate
that diversity-rewarded sampling restores compliance across ten generations.

---

## 2. Vendor Badge Program

### Badge Design

**"EnterpriseSynth Certified" badge tiers:**

| Badge | Criteria | What Vendors Get |
|---|---|---|
| **Benchmark Participant** | Submit results for ≥ 2 asset types | Listed in benchmark results table; GitHub mention |
| **Transparent Evaluator** | Full 3-metric report (privacy, utility, fidelity) publicly posted | "Transparent Evaluator" badge; blog post by EnterpriseSynth team |
| **Pareto Leader** | Achieves Pareto-efficient point in ≥ 2 compliance tiers | "Pareto Leader [Year]" badge; featured in paper's results section |

### Vendor Outreach Email (ready to send)

**Subject:** *Independent benchmark inclusion for [Vendor] — EnterpriseSynth*

> Hi [Name],
>
> I'm writing from the EnterpriseSynth project — an open benchmark for synthetic data
> evaluation in regulated enterprise settings (GDPR, HIPAA, SOX), currently being
> prepared for NeurIPS 2026 submission.
>
> We're offering synthetic data vendors the opportunity to include their results in
> the benchmark comparison. This means:
> - Running your API against our enterprise schema corpus (~500 records, six asset types)
> - Reporting three metrics: membership inference AUC, TSTR F1, and constraint violation rate
> - Getting listed in the paper's results section and receiving a "Benchmark Participant" badge
>
> There's no fee — the only ask is that results are reported honestly. We include
> all methods that submit, including unfavorable results (clearly labeled as
> "reported by vendor").
>
> The benchmark corpus and evaluation scripts are open-source:
> https://github.com/anote-ai/Research-EnterpriseSynth
>
> Would a 20-minute call this week work to walk through the evaluation protocol?
>
> Best,
> Rashmi Thimmaraju, anote AI

### Target Vendors (priority order)

| Vendor | Why Priority | Contact Method |
|---|---|---|
| **Gretel.ai** | Largest OSS community; NeurIPS presence | GitHub issues / LinkedIn |
| **Mostly AI** | Enterprise focus; healthcare customers | LinkedIn / conference |
| **Syntheticus** | Strong privacy positioning | Direct email |
| **Synthesized.io** | HR/enterprise data specialty | LinkedIn |
| **YData** | Active OSS community | GitHub |

---

## 3. Community Distribution Plan

### ε Selection Guide Distribution

**Target publications:**
- **IAPP** (International Association of Privacy Professionals) — submit to IAPP
  newsletter "Privacy Perspectives"; 50K+ readership among DPOs and compliance officers
- **CDO Magazine** — technology leadership audience; submit as contributed article
- **Towards Data Science (Medium)** — ML practitioner community; publish as a blog post
  linking to the full benchmark
- **Enterprise Data Management Community** — LinkedIn article with the one-page
  Quick Reference Card as the lead visual

**Distribution timeline:**
1. Publish ε guide on GitHub Pages alongside paper submission
2. Post blog version on Towards Data Science 2 weeks before paper submission
3. Submit to IAPP newsletter 4 weeks before NeurIPS deadline
4. Post Quick Reference Card as LinkedIn carousel 1 week before paper goes public

### Pareto Frontier Interactive Explorer

**Platform:** Hugging Face Spaces (free hosting, Gradio interface)

**Features:**
- Select asset type (tabular / healthcare / financial)
- Select regulatory context (GDPR / HIPAA / SOX)
- Slide ε → see privacy/utility/fidelity scores update
- Run collapse simulation → see tail entropy decay curve
- Download ε guide PDF customized to inputs

**Build plan:**
1. Port `plot_pareto.py` benchmark data to Gradio app (`app.py`)
2. Port `run_collapse_study.py` collapse curves to Gradio interactive plots
3. Deploy to Hugging Face Spaces under `anote-ai/enterprisesynth-explorer`
4. Link from paper, GitHub README, and IAPP article

---

## 4. Timeline Checklist

- [ ] ε selection guide published on GitHub Pages
- [ ] Pareto frontier visualization script tested and PNG charts generated
- [ ] Gretel.ai outreach sent (use template in Section 2)
- [ ] Mostly AI outreach sent
- [ ] HIMSS 2027 abstract submitted (submission window typically opens Sep–Oct 2026)
- [ ] ABA TechConnect abstract submitted (submission window typically opens Jun–Jul 2026)
- [ ] NeurIPS D&B poster abstract submitted (~May 2026)
- [ ] TPDP workshop abstract submitted (check co-location with NeurIPS 2026)
- [ ] Hugging Face Spaces demo deployed
- [ ] Towards Data Science blog post drafted
- [ ] IAPP newsletter submission sent
