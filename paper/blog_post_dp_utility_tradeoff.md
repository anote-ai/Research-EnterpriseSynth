# How Much Utility Do You Actually Lose for GDPR-Grade Privacy? We Measured It.

*Rashmi Thimmaraju · anote AI · June 2026*

---

Every enterprise data team building AI on sensitive data eventually hits the same wall: your legal team says you need differential privacy, your ML team says differentially private data is unusable, and nobody has numbers to settle the argument.

We built **EnterpriseSynth** to produce those numbers. Here's what we found.

---

## The question nobody was answering

Differential privacy (DP) gives you a formal, mathematical guarantee that no adversary — even one with unlimited compute and access to all other records in the world — can be more than ε-confident that any specific individual's data was used to train your model. Smaller ε = stronger privacy. The catch: smaller ε also means more noise, which means lower-quality synthetic data.

The open question for practitioners has always been: *how much lower?* The academic literature is full of DP-SGD papers reporting AUC on MNIST. Enterprise data teams need to know what happens to their fraud detection F1 when they apply GDPR-grade privacy to their transaction logs.

We measured it across three public enterprise-proxy datasets (HR/CRM, Financial, Healthcare EHR), six DP budgets (ε ∈ {0.1, 0.5, 1, 2, 5, 10}, all at δ=1e-5). Non-DP baselines use SDV (CTGAN, TVAE, GaussianCopula); the DP numbers below come from a real Opacus-backed DP-SGD run (a small tabular VAE, "DPVAE"), averaged over 5 random seeds per (domain, ε) — not a calibrated formula.

---

## The short answer: it depends enormously on your domain, more than we expected

At **ε=2, δ=1e-5** — the HIPAA-compatible tier — measured DP utility retention is **18% for HR, 91% for financial, 20% for healthcare**. There is no single "X% retained" number that honestly describes this tier across domains — that's the headline finding, and it's messier than we originally reported.

Here's the full picture — real DP-SGD training, mean ± std over 5 seeds per cell:

| Dataset (Domain) | Oracle F1 | No-DP TSTR | DP ε=1 | DP ε=2 | DP ε=5 | DP ε=10 |
| --- | --- | --- | --- | --- | --- | --- |
| Adult Income (HR/CRM) | 0.69 | 0.620 (94%) | 0.025 ± 0.04 | 0.127 ± 0.07 | 0.173 ± 0.07 | 0.205 ± 0.11 |
| Credit-G (Financial) | 0.845 | 0.783 (98%) | 0.581 ± 0.32 | 0.770 ± 0.07 | 0.661 ± 0.33 | 0.165 ± 0.33 |
| Diabetes PIMA (EHR) | 0.484 | 0.512 (91%*) | 0.133 ± 0.19 | 0.096 ± 0.19 | 0.00 ± 0.00 | 0.00 ± 0.00 |

*All DP values at δ=1e-5, measured via `scripts/run_opacus_dp_sweep.py`. Oracle = train and test on real data. No-DP TSTR = SDV synthesizer, no privacy. \*The DPVAE pipeline computes its own oracle F1 independently of the SDV pipeline (0.484 vs. 0.560 for diabetes); retention % here uses the SDV oracle for the no-DP column and the DPVAE oracle for DP columns — see `paper/draft.md` Table 2 footnote for the exact split.*

Two things jump out that a single "utility cliff at ε<1" story would hide:

- **HR utility never gets above 30% retention**, even at the loosest budget tested (ε=10). We believe this reflects the limits of our small pilot DP-SGD synthesizer (a lightweight custom VAE, not a production DP-CTGAN) more than a fundamental DP limit — see "What this pilot can't tell you yet" below.
- **Financial and healthcare are noisy and non-monotonic at the extremes** — financial actually *drops* from 78% retention at ε=5 to 20% at ε=10, and healthcare collapses to 0% at ε≥5. Both are small real datasets (800 and 614 rows), and training a DP model on that little data in a handful of gradient steps has real run-to-run variance — that's exactly what the ± std columns are showing you, not hidden away in a single mean.

---

## Domain type changes the curve — but not the way we expected

This is the finding that surprised us most, and it cuts against our own original hypothesis. We expected financial time-series data (transaction logs) to degrade *faster* under DP than tabular HR records, because DP noise should destroy the short-lag autocorrelations that fraud models rely on. The measured pilot shows the opposite: **financial transactions was the most DP-robust domain we tested**, retaining 91% of oracle utility at ε=2 versus 18% for HR and 20% for healthcare.

We don't think this overturns the general intuition that time-series/relational data is harder to DP-ify — it's more likely a dataset-scale confound. Credit-G (our financial proxy) is a small, low-cardinality dataset that a tiny model can fit easily; Adult Income (HR) is 50x larger with more feature diversity, which may be straining our small pilot synthesizer's capacity more than it's straining the DP mechanism itself. Either way, **we can't currently back the "financial degrades fastest" claim with data**, so we're retracting it here rather than repeating it.

We do not have measured data for IoT sensor or e-commerce relational domains. `results/pareto_study.json` (from `scripts/run_pareto_study.py`) is no longer empty, but it is a **simulated** 5-domain sweep (same calibrated `DomainSpec` model used elsewhere before real DP-SGD training existed), not real measurements — there is no UCI dataset backing either domain the way Adult Income, Credit-G, and Diabetes PIMA back the three domains in Table 2. No relational-schema or FK-violation claims should be attributed to this benchmark as *measured* until real data exists for those two domains.

---

## Which synthesizer to use — for the no-DP baseline

The synthesizer choice matters almost as much as the DP budget, at least without privacy:

| Domain | Best synthesizer | Why |
| --- | --- | --- |
| HR/CRM tabular | TVAE | Best categorical handling; 94% oracle retention |
| Financial tabular | CTGAN | Mode-specific normalization; 98% oracle retention |
| Healthcare EHR | GaussianCopula | Fast convergence; 91% oracle retention |

GaussianCopula fails badly on HR records (30% oracle retention) because its copula modeling breaks down on high-cardinality categoricals like job title and department. CTGAN and TVAE are robust across domains.

**Important**: this table is about the *no-DP* baseline only. The DP numbers earlier in this post come from a different, smaller pipeline (a custom Opacus-backed VAE, "DPVAE") — we don't yet have DP-SGD versions of CTGAN/TVAE/GaussianCopula to compare against, so don't read "TVAE gets 94% oracle retention" and "HR gets 18% DP retention" as a contradiction about the same model; they're two different synthesizers answering two different questions (best no-DP fidelity vs. best real DP-SGD result we've measured so far).

---

## Three tiers, one decision table

The ε ranges and regulatory mapping below still reflect our design reasoning, but — unlike our first draft of this post — we're not going to give you a single cross-domain "% retained" per tier. The measured data doesn't support one. Check the domain-specific numbers in the table above before committing to a budget.

**Strict (ε=0.1–0.5, δ=1e-5)**

- Utility retained: highly domain-dependent — measured 0% (HR) to 48% (financial) to 23% (healthcare) at ε=0.5
- Use when: clinical trial data, external data sharing, Art. 89 GDPR research, data with direct identifiers (name, SSN, MRN)
- Warning: training variance is high at *every* ε for small real datasets, not just strict ones (see the ± std columns above); always run ≥5 seeds and report mean ± std

**Balanced (ε=1–2, δ=1e-5)**

- Utility retained: measured 18% (HR) to 91% (financial) to 20% (healthcare) at ε=2 — a huge spread within one nominal tier
- Use when: standard HIPAA-covered analytics, HR and financial reporting, GDPR-compliant model training
- Only financial transactions approaches non-DP parity here in our pilot; don't assume this tier is "good enough" for your domain without checking

**Utility-focused (ε=5–10, δ=1e-5)**

- Utility retained: still domain-dependent and, for two of three domains, *non-monotonic* here — financial drops from 78% (ε=5) to 20% (ε=10), and healthcare hits 0% at both ε=5 and ε=10
- Use when: SOX audit trail simulation, internal model development, low-sensitivity operational analytics
- Don't assume "looser ε always means better utility" — our measured pilot has real counterexamples, most likely from small-dataset training instability rather than a genuine property of loose ε

---

## What this pilot can't tell you yet

Three honest limits on how far to trust these numbers:

- **Our DP synthesizer is a pilot, not a product.** DPVAE is a small 2-layer, 128-hidden-unit VAE trained for 15 epochs — nowhere near the capacity or training budget of a production DP-CTGAN or DP-TVAE. Treat the absolute retention numbers above as a floor on what real DP-SGD synthesis can do on this data, not a ceiling.
- **No real fidelity measurement yet.** Wasserstein distance and other statistical fidelity metrics for the DPVAE's synthetic output are still simulated, not measured — we only have real numbers for privacy (MIA AUC) and utility (TSTR F1) so far.
- **One dataset per domain isn't enough to settle the domain-ordering question.** We can't yet tell whether financial transactions is genuinely more DP-robust than HR, or whether Credit-G just happens to be an easier dataset for our small pilot model than Adult Income. That needs a matched-scale study with multiple datasets per domain.

---

## The part people miss: iterative retraining collapse

One finding from our research that gets less attention than the DP tradeoff: **if you retrain your synthesizer on its own outputs repeatedly, your data degrades catastrophically** — regardless of DP budget.

In a 5-generation retraining pipeline, tail-record diversity (fraud cases, security incidents, rare diagnoses) drops by **51% without any mitigation**. These are precisely the records your compliance and fraud teams care most about.

Two mitigation strategies work:
- **Diversity-rewarded sampling** (up-weight rare records during training) keeps tail entropy within 10% of generation-0 across all 5 generations. ✅
- **Real-data anchoring** alone (inject 20% original records each generation) is insufficient at high collapse rates. ❌

The practical implication: **every enterprise synthetic data pipeline needs collapse monitoring**. The three metrics to track are tail coverage entropy, minority class representation rate, and Jensen-Shannon divergence vs. the original distribution. All three are implemented in `src/model_collapse/metrics.py`.

---

## What we're releasing

**EnterpriseSynth** is open source at `github.com/anote-ai/Research-EnterpriseSynth`. It includes:

- A 6-domain enterprise schema corpus with formal constraint verification (3 of the 6 domains have real DP-SGD measurements so far; see caveats above)
- Real-data TSTR baselines on 3 public datasets (Adult Income, Credit-G, Diabetes PIMA)
- Real, measured DP-SGD Pareto data for 3 domains (tabular HR, healthcare EHR, financial transactions) — genuinely domain-varying, though noisier and less uniform than we originally expected
- A multi-generation collapse detection and mitigation framework
- 296 tests, CI-green on Python 3.10 and 3.11

The benchmark is designed so compliance officers can plug in their own ε constraints and get back a synthesizer recommendation and utility estimate grounded in real measurements where we have them, clearly labeled where we don't.

---

## The one-sentence summary

Measured DP-SGD retention at ε=2, δ=1e-5 ranges from 18% to 91% depending on domain — there is no single "X% retained" number that honestly describes the balanced tier, and your domain type matters as much as your privacy budget (though not in the direction we originally hypothesized).

---

*The full technical paper is available at `paper/draft.md` in the repository. Questions and benchmark contributions welcome via GitHub issues.*
