# Research Status: What Is Real vs. Simulated

This document provides an honest accounting of each experimental result in the repository. Reviewers, collaborators, and code auditors should consult this before treating any result as measured ground truth.

---

## Improvement Plan Status (issue #49)

| Item | Status | Notes |
| --- | --- | --- |
| 1. Real DP-SGD via Opacus | **DONE (pilot scale)** — `results/epsilon_sweep.json` now loads measured data | Opacus-backed DPVAE trained end-to-end at all 6 ε × 3 domains, 5 seeds each, wired into `scripts/run_epsilon_sweep.py`. Two real bugs found and fixed along the way (see below); a genuine training-instability limitation remains and is reported honestly rather than hidden — see "Known limitations of the measured DP-SGD pilot" |
| 2. `run_downstream_tasks.py` with real data | **PARTIAL** — classification, regression, and anomaly no-DP baselines now measured | Classification F1 from `results/baseline_sdg.json`; regression/anomaly no-DP baselines now use real sklearn datasets (diabetes progression, breast cancer) as domain-agnostic proxies — DP-adjusted values for all three tasks are still **estimated** via a logistic degradation curve, not real DP-SGD training |
| 3. README / DESIGN_DOC narrative mismatch | ✅ **RESOLVED** | README rewritten to describe DP benchmark; stale OpenAPI/SFT content removed (PR #50) |

### Known limitations of the measured DP-SGD pilot (item 1)

`scripts/run_opacus_dp_sweep.py` originally had two real correctness bugs, both fixed:

- **MIA metric wasn't a membership-inference attack.** The original `_mia_auc()` trained a real-vs-synthetic discriminator (a "can you tell these apart" test), which is a different question from "did the model memorize its training set." It was also hard-clamped to `[0.5, 0.95]`, so it barely moved with ε. Replaced with a loss-threshold membership-inference attack (Yeom et al. 2018 style): compare the trained VAE's reconstruction error on real training members vs. real held-out non-members. This now responds to ε as expected (e.g. tabular_hr: 0.5045 at ε=0.1 rising to 0.5104 at ε=10).
- **Mixed-type reconstruction loss.** The VAE used a single MSE loss over all columns (numeric + one-hot categorical), which let DP-SGD noise collapse categorical columns to a constant output regardless of ε. Replaced with BCE for categorical column blocks, MSE for numeric ones, plus KL annealing.

After both fixes, a deeper issue remained: the small custom DPVAE (2 hidden layers, 128 units) still shows real training instability on the two smaller real datasets (Credit-G: 800 rows, Diabetes: 614 rows) — high seed-to-seed variance and, at some ε extremes, non-monotonic utility. Ablating the KL term entirely did not fix this, confirming it is not a posterior-collapse bug but inherent instability of fitting a tiny model via DP-SGD in very few gradient steps on tiny datasets. Rather than keep tuning indefinitely or hide the instability, we run 5 seeds per (domain, ε) and report mean ± std — see `results/real_dp_sweep.json` and paper/draft.md Section 5.1.1 for the full discussion, including a domain-ordering finding (financial transactions is the *most* DP-robust domain measured, contradicting the design doc's "financial degrades fastest" hypothesis) that needs a larger, matched-scale study to confirm or refute.

---

## Summary

| Result file | Status | Provenance |
| --- | --- | --- |
| `results/baseline_sdg.json` | **MEASURED** | Full SDV (CTGAN/TVAE/GC) training runs on 3 UCI public datasets |
| `results/dp_real_integration.json` — oracle F1, no-DP TSTR | **MEASURED** | Real training runs via `scripts/run_dp_real_integration.py` |
| `results/dp_real_integration.json` — DP F1 columns | **ESTIMATED** | Calibrated retention curves × real oracle F1 |
| `results/downstream_tasks.json` — classification F1 no-DP | **MEASURED** | From `results/baseline_sdg.json` (Adult/Credit-G/PIMA) |
| `results/downstream_tasks.json` — regression, anomaly no-DP | **MEASURED** | sklearn diabetes-progression / RandomForestRegressor and breast-cancer / IsolationForest, used as domain-agnostic real-data proxies |
| `results/downstream_tasks.json` — all DP values | **ESTIMATED** | Logistic degradation curves applied to measured baselines |
| `results/epsilon_sweep.json` | **MEASURED (pilot scale)** | Loaded from `results/real_dp_sweep.json` — real Opacus DP-SGD training, 5-seed mean, for all 3 domains × 6 ε, including a real fidelity metric (Wasserstein-1/TVD, see below). See limitations above. |
| `results/real_dp_sweep.json` | **MEASURED (pilot scale)** | Opacus-backed DPVAE, real UCI datasets, 5-seed mean ± std per (domain, ε), for privacy (MIA AUC), utility (TSTR F1), and fidelity (Wasserstein-1/TVD). Small-model training instability — see limitations above |
| `results/document_dp_sweep.json` | **SIMULATED** | Literature-calibrated starting values (Zhang 2020, Su 2023, Yue 2023) |
| `results/collapse_study.json` | **SIMULATED** | Controlled pipeline: real seed → CTGAN → re-synthesize × 5 |
| `results/dp_mechanism_comparison.json` | **SIMULATED** | DP-SGD/Laplace run directly; PATE from Papernot et al. 2018 estimates |
| `results/fidelity_correlation.json` | **SIMULATED** | Pearson r from simulation outputs, not held-out real data |
| `results/product_audit.json` | **MEASURED** | Constraint violations measured on generated schema corpus |

---

## Per-Result Detail

### MEASURED — Directly Observed

**`results/baseline_sdg.json`** — generated by `scripts/run_baseline_sdg.py`

- Three synthesizers (CTGAN, TVAE, GaussianCopula) trained on three UCI public datasets
- TSTR F1, oracle F1, Wasserstein distance, correlation delta — all from actual training runs
- Key values:
  - Adult Income / TVAE: tstr_f1 = 0.6202, oracle = 0.658 (94.3% retention)
  - Credit-G / CTGAN: tstr_f1 = 0.7834, oracle = 0.797 (98.3% retention)
  - Diabetes PIMA / GaussianCopula: tstr_f1 = 0.5116, oracle = 0.560 (91.4% retention)

**`results/dp_real_integration.json`** — oracle F1 and no-DP TSTR columns

- oracle\_f1 and nodp\_tstr\_f1 for Adult Income/Credit-G/Diabetes PIMA: **directly measured**
- DP F1 columns (dp\_f1\_eps\*): **estimated** via `oracle_f1 × domain_retention(ε) / baseline_tstr`
- MIA AUC values: **estimated** via `privacy_leakage_at_epsilon(ε)` from DomainSpec model

**`results/product_audit.json`**

- Constraint violation rates measured directly on synthetic outputs from CTGAN/TVAE/GC

---

### MEASURED (pilot scale) — Real Training, Small Custom Model

**`results/real_dp_sweep.json`** and **`results/epsilon_sweep.json`** — generated by `scripts/run_opacus_dp_sweep.py` and `scripts/run_epsilon_sweep.py`

- Real Opacus DP-SGD training of a custom tabular VAE (DPVAE) on the same 3 real UCI datasets as `baseline_sdg.json`, at all 6 ε values, 5 random seeds each (90 total training runs)
- `run_epsilon_sweep.py` now loads `real_dp_sweep.json` per (domain, ε) when a row exists (it does, for all 18 combinations) and reports mean TSTR F1 / mean MIA AUC with bootstrap CIs computed over the 5 real seed values — `data_source` field on each row says `"measured — real DP-SGD training, mean over 5 seeds"`
- **Fidelity is now also measured**: `_fidelity_score()` in `run_opacus_dp_sweep.py` computes 1 − mean(Wasserstein-1 distance normalized by column range for numeric columns, total variation distance for categorical columns) between real training data and DPVAE-synthesized samples, averaged over the same 5 seeds. `run_epsilon_sweep.py` uses this real value (`fidelity_score_mean`) when present rather than the old `DomainSpec.fidelity_at_epsilon()` simulation; each row's `fidelity_source` field reflects which one was used. Notably, measured fidelity is far flatter across ε (~0.75-0.80 for HR) than measured utility is (0.00-0.20) — the DPVAE's statistical similarity to real data barely depends on the privacy budget even though its downstream usefulness depends heavily on it.
- **Real limitations, reported not hidden** (see "Known limitations" above): only the tabular_hr domain shows a clean monotonic utility increase with ε; financial_transactions and healthcare_ehr show high seed-to-seed variance and non-monotonic behavior at some ε extremes, most likely due to training instability of a small model on small datasets (614–800 rows), not a property of DP-SGD itself
- **Domain ordering does not match the design doc's hypothesis**: financial_transactions is the most DP-robust domain measured (91% retention at ε=2), contradicting the simulated model's "financial degrades fastest" assumption — flagged in paper/draft.md Section 5.1.1 and Section 8 rather than silently dropped
- The DPVAE itself is a pilot-scale synthesizer (2-layer, 128-hidden-unit VAE, 15 epochs) — absolute retention numbers (e.g. HR maxing out at 30% even at ε=10) most likely reflect this pilot architecture's ceiling, not a fundamental DP-SGD limit. A production DP-CTGAN/DP-TVAE would be the natural next upgrade to get numbers that upper-bound practitioner expectations.

---

### ESTIMATED — Calibrated Model Applied to Real Baselines

**`results/downstream_tasks.json`** — generated by `scripts/run_downstream_tasks.py`

Classification, regression, and anomaly no-DP baselines are all now **MEASURED**:

- Classification F1 from `baseline_sdg.json`: tabular_hr 0.6202 (Adult/TVAE), financial_transactions 0.7834 (Credit-G/CTGAN), healthcare_ehr 0.5116 (Diabetes PIMA/GaussianCopula)
- Regression R² from sklearn's diabetes-progression dataset / RandomForestRegressor (used as a domain-agnostic real-data proxy, not a domain-specific regression target)
- Anomaly recall from sklearn's breast-cancer dataset / IsolationForest (same caveat — a real-data proxy, not domain-specific)

All DP values are **ESTIMATED** via logistic `_dp_retention(ε, task)` curves applied to
the above baselines — none of the three tasks has been run through real DP-SGD training yet
(unlike `epsilon_sweep.json`'s classification-only real DP-SGD sweep above). The key qualitative
finding — anomaly detection degrades fastest — is established in the literature (Jordon et al. 2022);
the simulation confirms the direction.

**Attempted and not resolved**: `scripts/run_downstream_dp_sweep.py` applies the same DPVAE
training loop to regression (diabetes-progression) and anomaly (breast-cancer) tasks. Both
attempts surfaced genuine problems rather than usable measurements:

- **Regression** collapses to predicting a near-constant target (R² = 0.0 across all 5 seeds
  at *every* tested ε, including ε=10 — ruling out DP noise as the cause). Same class of
  instability as the DPVAE/DPTVAE issues elsewhere in this document, now surfacing for a
  continuous target.
- **Anomaly detection**'s recall metric is degenerate for the DP-SGD synthetic version: the
  IsolationForest trained on DP-SGD-synthesized "normal" data flags ~100% of the real test set
  as anomalous (precision = 0.374, exactly the test set's true anomaly base rate — no genuine
  discrimination). Recall = 1.0 here is trivial, not a working detector. For contrast, the
  **existing, already-committed oracle baseline is legitimate**: real-data recall = 1.0 *with*
  precision = 0.634, a real detector — only the new DP-SGD synthetic attempt is degenerate.

Neither result is wired into `downstream_tasks.json`; both tasks' DP values remain estimated.

---

### SIMULATED — No Direct Measurement

**`results/document_dp_sweep.json`**

- BERTScore/MAUVE/NER consistency values are literature-calibrated estimates
- The DP guarantee for document assets applies to DP-SGD fine-tuning only (not full per-token composition)

**`results/collapse_study.json`**

- The 51% tail entropy drop is from a controlled simulation pipeline (real seed data → CTGAN → re-synthesize × 5 generations)
- Mitigation comparison is measured in the same pipeline; not validated on production-scale enterprise data

**`results/dp_mechanism_comparison.json`**

- DP-SGD and Laplace results are from direct runs; PATE estimated from Papernot et al. 2018

**`results/fidelity_correlation.json`**

- Pearson r values computed from simulation outputs, not from a separate held-out real dataset

---

## Path to Replacing Remaining Simulated/Estimated Results with Measurements

| Step | Script | What it produces | Status |
| --- | --- | --- | --- |
| 1 | `scripts/run_opacus_dp_sweep.py` | `results/real_dp_sweep.json` | ✅ Done — 3 domains × 6 ε × 5 seeds |
| 2 | `run_epsilon_sweep.py` loads from `real_dp_sweep.json` | Measured `epsilon_sweep.json` | ✅ Done |
| 3 | `scripts/run_baseline_sdg.py` (already done) | `baseline_sdg.json` ✅ | None — already run |
| 4 | Extend `run_downstream_tasks.py`'s DP columns to use real DP-SGD (regression/anomaly, not just classification) | Fully measured `downstream_tasks.json` | Attempted (`run_downstream_dp_sweep.py`), not resolved — regression collapses to a constant prediction; anomaly's DP-SGD synthetic detector is degenerate (flags ~100% of test set). Neither wired in. |
| 5 | Replace pilot DPVAE with a production DP-CTGAN/DP-TVAE | Retention numbers that upper-bound practitioner expectations | Not started — current numbers are a pilot-architecture floor (see limitations above) |
| 6 | Compute a real fidelity metric (e.g. Wasserstein distance) on DPVAE synthetic output | Measured fidelity column in `epsilon_sweep.json` | ✅ Done — Wasserstein-1 (numeric) / TVD (categorical), see limitations above |

---

## Reproducibility

`epsilon_sweep.json` now reproduces entirely from real measured training (privacy, utility,
and fidelity) for all 18 (domain, ε) combinations — rerunning it does not retrain the DPVAE, it just
reloads whatever is committed in `results/real_dp_sweep.json`:

```bash
python scripts/run_epsilon_sweep.py --json > results/epsilon_sweep.json
python scripts/run_downstream_tasks.py --json > results/downstream_tasks.json
```

Real-data baselines (already committed in `baseline_sdg.json`):

```bash
pip install sdv pandas scikit-learn
python scripts/run_baseline_sdg.py  # takes ~20 min, downloads UCI datasets
```

Real DP-SGD sweep (already committed in `real_dp_sweep.json`; rerun to reproduce/extend):

```bash
pip install opacus torch
python scripts/run_opacus_dp_sweep.py --dry-run          # checks deps
python scripts/run_opacus_dp_sweep.py --epochs 15 --seeds 5 --json > results/real_dp_sweep.json
# full 3-domain x 6-epsilon x 5-seed sweep takes ~25-30 min on CPU (adult dataset dominates runtime)
```

---

Last updated: 2026-07-01
