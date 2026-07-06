# EnterpriseSynth: One-Page Summary

**What this is:** a benchmark for the privacy-utility-fidelity tradeoff of differentially private synthetic data generation across regulated enterprise domains (HR, healthcare EHR, financial transactions).

**Full documents:** [DESIGN_DOC.md](DESIGN_DOC.md) (research questions and hypotheses) · [RESEARCH_STATUS.md](RESEARCH_STATUS.md) (per-result measured/estimated/simulated provenance) · [paper/draft.md](paper/draft.md) (full paper, Abstract through Appendices) · [paper/blog_post_dp_utility_tradeoff.md](paper/blog_post_dp_utility_tradeoff.md) (accessible writeup) · [paper/blog_post_model_collapse.md](paper/blog_post_model_collapse.md) (model collapse side-finding)

---

## Four core findings, and how solid each one is

| Finding | Status | Detail |
| --- | --- | --- |
| Non-DP synthesizer baselines are strong | **Measured** | TVAE retains 94.3% of oracle F1 on HR data (Adult Income); CTGAN retains 98.3% on financial data (Credit-G); GaussianCopula retains 91.4% on healthcare EHR (Diabetes PIMA). Real SDV training runs, not projections. |
| DP-SGD utility retention is highly domain-dependent, not a single number | **Measured (pilot scale)** | At ε=2, δ=1e-5: 18% retention for HR, 91% for financial, 20% for healthcare. This directly contradicts the design doc's original hypothesis that financial time-series data degrades *fastest* under DP; financial was the most DP-robust domain measured here. See [paper/draft.md](paper/draft.md) Section 5.1.1 for why, and why this needs a larger study to confirm. |
| Unmitigated iterative retraining destroys tail-record diversity | **Measured (controlled pipeline)** | Tail coverage entropy drops 51% by generation 5 at a 30% collapse rate; fraud/security records fall below 0.5% representation by generation 7. Diversity-rewarded sampling keeps tail diversity within 10% of baseline; real-data anchoring alone is insufficient. |
| Fidelity metric choice matters, and differs by asset type | **Simulated** | Constraint violation rate is the dominant fidelity predictor for tabular assets; BERTScore is dominant for document assets. No single metric works for both. Not yet validated against a real held-out dataset. |

---

## What's still open (not swept under the rug)

- **`downstream_tasks.json`'s DP values are still estimated**, not measured. Classification's DP values go through real DP-SGD training (via `epsilon_sweep.json`). Regression and anomaly were attempted (`scripts/run_downstream_dp_sweep.py`) and **not resolved**: regression collapses to a constant prediction (R²=0.0 at every ε, including the loosest); the DP-SGD synthetic anomaly detector flags ~100% of the test set as anomalous (precision = base rate, no real discrimination). Neither is wired in.
- **The production-grade DP-TVAE upgrade is unresolved.** A real SDV/ctgan TVAE architecture wrapped in Opacus is a clear improvement over the current custom lightweight DPVAE on credit_g and diabetes (higher utility, much lower variance) — but it completely collapses on the adult (HR) dataset under any real DP noise, even though the same architecture learns adult fine without DP. Several standard mitigations (clip-norm tuning, batch size) didn't fix it. `results/real_dp_sweep.json` currently uses the smaller DPVAE (validated across all 3 domains) until this is resolved.
- **The domain-ordering finding needs a matched-scale study, not just a bigger dataset.** Financial transactions (Credit-G, 800 rows) outperforming HR (Adult Income, 39k rows) under DP could reflect a genuine domain effect, or could just be a dataset-size/complexity confound — can't be told apart with one dataset per domain.
- **Document-domain (contracts, support tickets, compliance reports) has no real oracle baseline** — Table 3 in the paper reports no-DP fidelity/TSTR numbers with no measured "train and test on real documents" ceiling to compare against yet.

## What was cleaned up in this pass

- Removed `paper/PAPER.md`, a stale duplicate paper draft from an earlier branch that still carried the old, now-corrected "79-81% retention across all domains" claim and pre-fix Table 2 numbers — `paper/draft.md` is the single canonical paper document going forward.
- Fixed two real correctness bugs in the DP-SGD training script (`scripts/run_opacus_dp_sweep.py`): the "MIA AUC" privacy metric was actually a real-vs-synthetic discriminator, not a membership-inference attack; the VAE's reconstruction loss let categorical columns collapse to a constant output under DP noise. Both fixed and the fixes are documented in [RESEARCH_STATUS.md](RESEARCH_STATUS.md).
- **Added a real fidelity metric** (Wasserstein-1 distance for numeric columns, total variation distance for categorical columns) to the DP-SGD pipeline — the privacy/utility columns in `epsilon_sweep.json` were already measured, but fidelity was still simulated everywhere until this pass.
- **Fixed `scripts/plot_pareto.py`**, which had been silently disconnected from all the real measured data — it used a hardcoded dict of old simulated numbers instead of loading `results/epsilon_sweep.json`. Also fixed a `sharey=True` bug found while testing the fix that was clipping two of three domains' data completely out of the regenerated figure.
- **Added `scripts/plot_measured_results.py`**, generating real figures for every experiment that didn't have one before: model collapse mitigation, synthesizer baseline comparison, downstream task retention, document DP sweep, DP mechanism comparison, fidelity correlation, and product audit — see the Figures table in [README.md](README.md).
