# Mock Peer Review — EnterpriseSynth

*Internal review conducted before NeurIPS 2026 D&B track and TPDP workshop submission.*
*Three mock reviewers: DP theory, ML practitioner, NeurIPS/TPDP culture.*
*Date: 2026-06-23. Paper version reviewed: current `paper/draft.md`.*

---

## Reviewer 1 — DP / Privacy Theory Expert

**Profile**: Researcher with 5+ years working on differential privacy mechanisms, privacy accountants, and membership inference. Has reviewed for TPDP, CCS, and IEEE S&P. Will read every ε claim and check the math.

**Overall score**: 6/10 — Good setup, correctness concerns on composition and document DP.

---

### R1.1 STRENGTHS

- Privacy accounting unit tests (`tests/test_privacy_accountant.py`) are thorough and verifiable in under 5 minutes. This is exactly what TPDP reviewers want.
- The threat model in `dp_privacy_analysis.md` is well-scoped (black-box, single-release, shadow-model MIA).
- Discrete Gaussian mechanism for integer fields is technically correct and often overlooked — citing Canonne et al. (2020) is right.
- Explicit DP-guarantee scope table ("What DP Does and Does Not Guarantee") in the ethics statement is commendable and rarely included. Keep this in the paper itself, not just in supplementary.

---

### R1.2 CONCERNS — POTENTIAL PAPER-KILLERS

**[BLOCKER] The document synthesis DP guarantee is described as "approximate" without quantification.**

> Section 2.4 of `dp_privacy_analysis.md`: "the DP guarantee for document synthesis is approximate and discussed separately in Section 2.4."

*Problem*: TPDP reviewers will demand either (a) a formal (ε, δ) guarantee for document synthesis, or (b) a clear statement that document synthesis is NOT covered by the DP guarantee and an explanation of what privacy protection is actually provided. Calling it "approximate" without bounding the approximation error is not acceptable. Reviewers may reject the paper on this point alone, arguing the privacy claims are overclaimed for 2 of the 6 data types.

*Fix*: Pick one of:
- **Option A**: State explicitly that document synthesis (contracts, HR memos, EHR clinical notes) does NOT carry a formal (ε, δ)-DP guarantee. The DP guarantee covers only the fine-tuning step via DP-SGD; composition across tokens introduces approximation that is not bounded in the paper. Characterize the document types as "DP-SGD fine-tuned (ε_training reported; full per-token composition not claimed)."
- **Option B**: Implement and report PRV accountant (Gopi et al., 2021) bounds for the LLM fine-tuning step, and state explicitly which composition is used (per-token, per-document, or per-user). Requires additional implementation.

Recommended: **Option A** with a clear "Limitations" callout in Section 8.

---

**[MAJOR] Basic vs. advanced composition must be stated explicitly for each reported ε value.**

The paper states: "We use basic composition in all reported results to give conservative (guaranteed) bounds." This is fine, but is inconsistent with the claim in Section 4.4 "DP epsilon accounting is verified to within a tolerance of 0.01 between reported and privacy-accountant-computed ε values." If basic composition is used, the accountant always matches by definition (it is the accountant). The tolerance statement implies a comparison against something else — clarify what that "something else" is.

*Fix*: Add a footnote to every Table reporting ε: "All ε values computed via basic (ε₁+ε₂+...+εₖ) composition across k fields; δ_total = k × δ_per_field. PRV-accountant cross-check tolerance: ±0.01 on the tabular synthesis budget."

---

**[MAJOR] RDP → (ε, δ)-DP conversion is not described.**

The privacy accountant tests (`test_privacy_accountant.py`) test `compute_epsilon` with Gaussian, Laplace, and RR mechanisms. DP-SGD training (for the LLM-based document generator) uses Rényi DP (RDP) and then converts to (ε, δ)-DP via the standard conversion (Mironov 2017, Proposition 3). This conversion is not described in the paper or the `dp_privacy_analysis.md`. Reviewers will ask: "For DP-SGD, how many gradient steps, what noise multiplier, what batch size, and what is the resulting RDP → (ε, δ) conversion?" These numbers must appear in the paper or appendix.

*Fix*: Add an appendix table: "DP-SGD hyperparameters for document synthesis fine-tuning: noise_multiplier, batch_size, n_epochs, RDP order α used for conversion, resulting (ε, δ) at δ=1e-5."

---

**[MINOR] Composition across multiple ε tiers in the benchmark tables is unclear.**

Table 2 reports ε ∈ {0.1, 0.5, 1, 2, 5, 10} for the same model. Are these separate training runs (each using its own budget), or is each ε achieved by early-stopping the same training run? If the latter, these are not independent DP mechanisms — the reported ε for each row may be understated because the model has "seen" the data at prior ε values.

*Fix*: Clarify in the experimental protocol: "Each ε value corresponds to an independent training run with budget ε from scratch, not early-stopping of a single training run."

---

**[MINOR] δ values are not reported in the main paper.**

Every (ε, δ)-DP result should report both ε and δ. The paper consistently reports only ε. For tabular experiments, δ = 10⁻⁵ or 10⁻⁶ is standard but must be stated. Add a single footnote: "All tabular DP results use δ = 1e-5 (per field). Total δ reported as k × 1e-5 per the basic composition."

---

### R1.3 VERIFICATION STEPS

Run these before submission to confirm DP correctness:

```bash
# 1. Privacy accountant unit tests — must all pass
pytest tests/test_privacy_accountant.py -v

# 2. Spot-check: Gaussian at sigma=calibrated must recover target epsilon
python - <<'EOF'
import math
from privacy_benchmark.accountant import compute_epsilon, gaussian_sigma_for_epsilon
sigma = gaussian_sigma_for_epsilon(1.0, 1e-5)
eps = compute_epsilon("gaussian", sigma=sigma, delta=1e-5, sensitivity=1.0)
assert abs(eps - 1.0) < 1e-6, f"Accountant error: got {eps}"
print(f"OK: Gaussian eps={eps:.8f}")
EOF

# 3. Composition cross-check: 5 fields at eps=0.2 must give eps_total=1.0
python - <<'EOF'
from privacy_benchmark.accountant import compose_basic
eps_total, delta_total = compose_basic([(0.2, 1e-6)] * 5)
assert abs(eps_total - 1.0) < 1e-9
assert abs(delta_total - 5e-6) < 1e-12
print(f"OK: Composition eps={eps_total}, delta={delta_total}")
EOF
```

---

## Reviewer 2 — ML Practitioner (Synthetic Data / Utility Evaluation)

**Profile**: Applied ML researcher who has used and evaluated synthetic data for model training in production. Will scrutinize whether the TSTR evaluation is realistic, whether baselines are appropriate, and whether the statistical methodology holds.

**Overall score**: 7/10 — Solid methodology, some evaluation realism concerns.

---

### R2.1 STRENGTHS

- TSTR F1 with bootstrap CIs (1,000 bootstrap samples, percentile method) is the right methodology for finite-sample robustness. Good.
- Wilcoxon signed-rank test with Bonferroni correction is appropriate for paired comparisons across ε values and data types.
- Multi-seed (5 seeds) evaluation and reporting mean ± std is correct.
- The document evaluation suite (BERTScore, MAUVE, NER consistency, TSTR F1) is more comprehensive than any prior DP synthetic data paper.

---

### R2.2 CONCERNS

**[MAJOR] TSTR document scores (F1 0.83–0.91) are very high — reviewers will demand ablation or baseline context.**

The document TSTR F1 values in Table 4 (Contracts: 0.89, Support Tickets: 0.83, Compliance Reports: 0.88, HR Memos: 0.84) are high enough that a reviewer will immediately ask: "What is the real-data TSTR baseline (training on real data, testing on real data)?" Without this comparison, the scores are uninterpretable. Is 0.89 good or bad if real-data TSTR is 0.97? Or 0.79?

*Fix*: Add a "Real-Data Oracle" row to Table 4: train on real data, test on held-out real data. This gives the utility ceiling. Then the DP synthetic data rows can be interpreted as percentage of oracle performance.

---

**[MAJOR] Which ε value do the document scores in Table 4 correspond to?**

Table 4 as described reports a single set of document metrics without specifying which ε value they correspond to. Each row should either (a) report metrics across all 6 ε values, or (b) clearly state "ε=2 (Balanced tier)."

*Fix*: Add an ε column to Table 4, or split into Table 4a (strict tier ε=1) and Table 4b (balanced tier ε=2).

---

**[MAJOR] MST/AIM baselines are missing from the tabular evaluation.**

The paper compares CTGAN, TVAE, and Gaussian Copula (plus DP-SGD variants). McKenna et al.'s MST and AIM are the current state-of-art for tabular DP synthesis and beat CTGAN+DP-SGD on most NIST benchmark tasks. NeurIPS reviewers will ask why AIM is not included as a baseline. Excluding it is a significant gap.

*Fix*: Add AIM (McKenna et al., NeurIPS 2022) as a tabular baseline at the same ε values. If runtime is prohibitive, include a note explaining why (e.g., "AIM is designed for workload query accuracy, not ML dataset generation; we include DPCTGAN as the closest neural-synthesis baseline").

---

**[MINOR] The 5-seed variance analysis is mentioned in Section 5.3 but the data is not shown in the paper.**

> "Multi-seed variance analysis (Section 5.3) reveals that tighter privacy budgets increase training variance."

The actual variance data (mean ± std per ε value and data type) is not in a table. Add Table 5: "TSTR F1 mean ± std across 5 seeds by ε value" for at least one data type (e.g., HR Records). This is the empirical claim that supports "ε=0.1 has high variance" — it needs numbers.

---

**[MINOR] Bootstrap CI methodology: what is the CI on?**

"95% bootstrap confidence intervals (1,000 bootstrap samples, percentile method)" — CI on what? The TSTR F1 score, the mean across 5 seeds, or a single run? Clarify: "95% bootstrap CI computed over 1,000 resamples of the test set, applied to TSTR F1 point estimate from a single synthetic dataset generated with a fixed seed."

---

**[MINOR] Train/test split for TSTR is not specified.**

For each data type, what is the real data size, and how is it split between synthetic training target and real test set? With small enterprise datasets, the TSTR F1 will have high variance from the test-set size alone. Specify: "All datasets use 80/20 real-data train/test split; synthetic data is generated to match the training set size."

---

### R2.3 QUICK VERIFICATION

```bash
# Confirm TSTR baseline tests pass
pytest tests/test_tstr.py -v

# Confirm statistical rigor tests pass
pytest tests/test_statistical_rigor.py -v

# Spot check constraint violation on HR schema
pytest tests/test_constraints.py -v -k "hr"
```

---

## Reviewer 3 — NeurIPS / TPDP Reviewing Culture Expert

**Profile**: Has published at NeurIPS D&B track and TPDP workshop. Knows what meta-reviewers look for, how to read area chair decisions, and which objections lead to rejection vs. revision. Has been an APC for TPDP.

**Overall score**: 6/10 — Ambitious scope; needs sharpening and scope reduction for TPDP.

---

### R3.1 STRENGTHS

- The problem is well-motivated: compliance teams genuinely need ε selection guidance, and no prior work provides it empirically.
- The model collapse contribution is novel and timely — Shumailov et al. (2023) is a high-impact paper and extending it to structured tabular data is a natural and publishable direction.
- The ethics statement and regulatory caveat (DP ≠ GDPR/HIPAA compliance) are refreshingly honest and will be positively received by TPDP reviewers who are tired of overclaiming.
- Open-source implementation with reproducibility tests is NeurIPS D&B track's explicit requirement — this is well-addressed.

---

### R3.2 CONCERNS

**[BLOCKER — TPDP SPECIFIC] The paper has four separate contributions. TPDP expects one tight contribution with a strong theoretical component.**

The current paper contributes (1) an enterprise schema benchmark, (2) a three-dimensional evaluation framework, (3) a compliance-tier mapping, and (4) a model collapse study. For a full NeurIPS paper, this scope is appropriate. For a TPDP workshop paper (typically 6–8 pages), this is too broad and will be perceived as "four thin contributions bundled together."

*Fix for TPDP*: Submit only the **model collapse contribution** to TPDP — it has the cleanest privacy-theory hook ("does DP budget accumulation accelerate collapse?"). Contribute the full benchmark to NeurIPS D&B. Do not try to fit both into one TPDP submission.

---

**[MAJOR] "First benchmark" is a strong claim that will face scrutiny.**

The paper makes four "first" claims in the introduction. NeurIPS reviewers will try to falsify each. The claim "first enterprise-schema benchmark corpus" is vulnerable to the objection "SDGym covers tabular enterprise data." The claim "first multi-generation model collapse study on DP tabular data" is likely defensible.

*Fix*: Replace every "first" with "to our knowledge, the first." And add a sentence in the related-work section for each claim explicitly ruling out the nearest prior work: "SDGym uses generic tabular datasets (adult, credit) without domain-specific constraints — our enterprise schema corpus is the first to include inter-column constraint validation as a first-class metric."

---

**[MAJOR] The paper does not include negative results or failure modes of the benchmark.**

NeurIPS D&B track reviewers look for calibrated claims. The paper does not discuss:
- Cases where the benchmark gives misleading results
- Schema types where TSTR F1 is not a valid utility proxy
- Data types where the mitigation strategies fail

*Fix*: Add a "Limitations" subsection (currently Section 8 is generic) with at least two concrete failure modes: (1) "For EHR clinical notes with highly specialized medical terminology, BERTScore underestimates semantic divergence — NER consistency is the more informative metric," and (2) "For financial time-series with strong autocorrelation structure, TSTR F1 with an iid classifier underestimates the utility gap."

---

**[MAJOR] Blind submission check: identifiers in the paper.**

The current draft header includes:
```
**Rashmi Thimmaraju**
anote AI
rashmithimmaraju14@gmail.com
```

NeurIPS and most TPDP submissions require double-blind review. The author name, institution, and email must be removed from the submission PDF. The GitHub repo URL (`https://github.com/anote-ai/Research-EnterpriseSynth`) in the dp_privacy_analysis.md reviewer guide also de-anonymizes the submission.

*Fix*: For the submission version, replace author block with "Anonymous Authors" and replace the GitHub URL with "Anonymous GitHub repo (provided to reviewers upon request)." Maintain a separate de-anonymized version for camera-ready.

---

**[MINOR] Page count: the current draft is approximately 14 pages (excluding references and appendix) — NeurIPS D&B limit is 9 pages + references.**

The current paper has Sections 1–9 plus two appendices. To fit within 9 pages:
- Move Appendix A (Implementation Details) and Appendix B (Compliance Tier Mapping) to supplementary
- Condense Section 7 (Practical Guidance) to a single decision table; remove prose
- Merge Sections 4.1–4.3 into a single "Evaluation Protocol" section
- Cut the enterprise decision guide to 1 page using a compact table format

---

**[MINOR] The conclusion oversells the impact.**

> "A key finding is that unchecked iterative retraining on synthetic data causes tail-record collapse (−51% tail entropy over five generations...)"

This finding is compelling but the paper should not claim it applies universally. The −51% figure comes from a specific schema (fraud records at ~1% prevalence) and a specific mitigation scenario. State: "In our HR-records experiment with a 1% fraud rate, tail entropy fell by 51% over five generations without mitigation."

---

### R3.3 SUBMISSION READINESS CHECKLIST

| Check | Status | Action |
|---|---|---|
| All ε values explicitly stated (no "approximately") | Needs check | Audit every table and figure caption |
| Threat model defined in main paper body | Partial — in supplementary | Move threat model summary to Section 2 |
| Page count ≤ 9 (NeurIPS D&B) | Likely over | Condense Sections 4–7 |
| Blind submission: no author identifiers | FAIL | Remove author block and GitHub URL |
| "First" claims qualified with "to our knowledge" | FAIL | Global find-replace |
| Negative results / failure modes stated | Missing | Add to Section 8 |
| Real-data oracle TSTR baseline in evaluation tables | Missing | Add to Tables 2, 4 |
| δ values reported alongside ε | Missing | Add to all result tables |
| AIM / MST baseline included | Missing | Add or explicitly exclude with justification |
| 5-seed variance table for at least one data type | Missing | Add Table 5 |

---

## Aggregate Assessment: Paper-Killers and Priority Queue

### PAPER-KILLERS (fix before submission or the paper will be rejected)

| ID | Issue | Reviewer | Estimated Fix Time |
|----|-------|----------|--------------------|
| PK-1 | Document synthesis DP guarantee not quantified | R1 | 1 day (choose Option A — just add the disclaimer) |
| PK-2 | Author identifiers in blind submission | R3 | 2 hours (create anonymized version) |
| PK-3 | Missing real-data oracle TSTR baseline | R2 | 1 day (run oracle TSTR; add row to tables) |

### MAJOR ISSUES (fix before submission or expect revision requests)

| ID | Issue | Reviewer | Estimated Fix Time |
|----|-------|----------|--------------------|
| M-1 | δ values not reported in result tables | R1 | 2 hours |
| M-2 | MST/AIM baseline missing from tabular evaluation | R2 | 3 days (run AIM; or write justified exclusion note) |
| M-3 | Basic vs advanced composition ambiguity | R1 | 2 hours (add table footnote) |
| M-4 | "First" claims not qualified | R3 | 1 hour (global replace) |
| M-5 | RDP → (ε,δ) conversion for DP-SGD not described | R1 | 4 hours (add appendix table) |
| M-6 | Which ε do document Table 4 scores correspond to? | R2 | 1 hour (add ε column) |
| M-7 | Page over NeurIPS limit | R3 | 2 days (condensing) |
| M-8 | TPDP scope: 4 contributions too broad | R3 | N/A — different submission strategy |

### MINOR (fix if time permits)

| ID | Issue | Reviewer | Estimated Fix Time |
|----|-------|----------|--------------------|
| m-1 | Train/test split not specified for TSTR | R2 | 30 min |
| m-2 | Bootstrap CI target not defined | R2 | 30 min |
| m-3 | Conclusion claim needs scoping | R3 | 30 min |
| m-4 | 5-seed variance table missing | R2 | 2 hours |
| m-5 | DP experiments: confirm independent training runs per ε | R1 | 1 hour |

---

## Revision Checklist (from issue #19 Step 5 + 6)

### DP Correctness (non-negotiable)

- [ ] **PK-1 resolved**: Document synthesis DP scope explicitly disclaimed (Option A recommended) or formally bounded (Option B)
- [x] **DP accounting unit tests pass**: `pytest tests/test_privacy_accountant.py -v` — 154 tests pass
- [x] **Accountant verified for Gaussian, Laplace, RR, and composition**: see `tests/test_privacy_accountant.py`
- [ ] **δ values added to all result tables**
- [ ] **RDP → (ε,δ) conversion for DP-SGD fine-tuning documented in appendix**
- [ ] **Composition type (basic) stated explicitly in every ε table footnote**
- [ ] **Independent training run per ε value confirmed** (not early-stopping of shared run)

### Utility Evaluation

- [ ] **Real-data oracle TSTR baseline added** to Tables 2 and 4
- [ ] **5-seed variance table** added (at least HR Records across all ε values)
- [x] **Bootstrap CIs** described in Section 4.4 — verify they appear in all tables
- [x] **Wilcoxon + Bonferroni** stated in Section 4.4 — specify n in the text
- [ ] **MST/AIM baseline added** or exclusion justified with a specific sentence in Section 4

### Paper Completeness

- [x] **Ethics statement** covers DP limitation scope and regulatory alignment (`paper/ethics_statement.md`)
- [x] **Regulatory caveat** (DP ≠ GDPR/HIPAA compliance) stated in ethics statement — verify it appears in Section 2 or 8 of the paper
- [ ] **Failure modes** added to Section 8 (EHR BERTScore limitation; time-series TSTR limitation)
- [ ] **"To our knowledge"** added to all "first" claims

### Final Submission Checks

- [ ] **All ε values explicitly stated** — audit every table caption and figure label; no "approximately" language
- [ ] **Threat model** summary added to Section 2 main body (currently only in supplementary)
- [ ] **Page count ≤ 9** for NeurIPS D&B submission (verify with compiled PDF)
- [ ] **Blind submission version**: author block → "Anonymous Authors"; GitHub URL → anonymized
- [ ] **Supplementary**: move Appendix A (Implementation Details) and Appendix B (Compliance Tier Mapping)
- [ ] **DP reviewer verification guide** (`dp_privacy_analysis.md` Section 6) works end-to-end on a fresh clone

---

## Red-Team: Weakest Claims

### Claim: "−51% tail entropy over five generations"
**Risk**: Reviewers ask "is this specific to your synthetic data generator, your fraud prevalence, or is it a general finding?"
**Mitigation**: Report the result for at least two prevalence rates (1% and 3% fraud) and two generators (CTGAN, TVAE) to show the finding generalizes. If it only holds for one setting, frame as a case study.

### Claim: "Mitigation strategies maintain tail diversity within 10%"
**Risk**: "10% relative to what?" — the denominator (original distribution tail entropy) must be defined precisely.
**Mitigation**: Add the formal definition of "within 10%" in Section 6.5: "Tail entropy after 5 generations with mitigation is ≥ 0.90 × tail entropy of the original distribution."

### Claim: "formal DP guarantees are not yet provided by any commercial enterprise synthetic data tool"
**Risk**: Gretel may update their documentation between now and submission. This claim becomes stale quickly.
**Mitigation**: Add a timestamp: "As of June 2026, no commercial tool provides a peer-reviewed, auditable (ε,δ)-DP certificate." Cite Stadler et al. (2022) for empirical support.

### Claim: "first empirical compliance-tier mapping"
**Risk**: NIST Privacy Framework and ISO 29101 have compliance-to-ε guidance, albeit informal.
**Mitigation**: Acknowledge these informal frameworks in Section 5.2 and position EnterpriseSynth as the first **empirical** (experiment-backed, not opinion-based) mapping.
