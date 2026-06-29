# Research Status (honest snapshot)

This file exists because the README, DESIGN_DOC.md, and paper/draft.md currently
describe three partially-overlapping projects, and not all numbers quoted in the
repo come from a real, reproducible experiment run. This page is the single
source of truth for "what is actually measured vs. projected" as of 2026-06-29.

## What this repo actually contains

There are at least two distinct research threads living in this repository:

1. **EnterpriseSynth (DP synthetic data)** — described in `DESIGN_DOC.md` and
   `paper/draft.md`. Goal: characterize the privacy-utility tradeoff for DP
   synthetic data generation across enterprise schema types.
2. **EnterpriseSynth (cold-start SFT trace generation)** — described in the top
   of `README.md`. Goal: generate verified tool-use SFT traces from OpenAPI
   schemas. This is a different research question with different code
   (`src/enterprisesynth/core.py`, `evaluate.py`) than the DP work
   (`src/privacy_benchmark/`, `src/tstr_eval/`, `src/model_collapse/`).

These two threads are not currently connected by any shared narrative. Anyone
opening this repo for the first time will be confused about which one is "the"
project. **Recommendation: pick one as the primary thesis (the DESIGN_DOC.md /
DP-utility tradeoff thread is the one with an actual design doc, a paper draft,
and the most developed evaluation code, so it should likely be primary) and
either move the OpenAPI/SFT-trace work to a clearly separate section or a
separate repo.**

## What is a real, measured result

| Result | File | Status |
|---|---|---|
| CTGAN / TVAE / GaussianCopula TSTR F1 on Adult / Credit-G / Diabetes-PIMA | `results/baseline_sdg.json` (produced by `scripts/run_baseline_sdg.py`) | **Real.** Has per-run wall-clock times (e.g. CTGAN on Adult: 408.3s), real oracle F1 computed by train/test split on the actual public dataset, and is reproducible by re-running the script. |
| Model collapse multi-generation entropy decay | `results/collapse_study.json` (produced by `scripts/run_collapse_study.py`) | Real in the sense that it is the output of an actual simulation pipeline with seeded randomness — but the underlying "generation" process is itself a synthetic simulation of collapse dynamics, not real generative model retraining. Treat as a methodology demonstration, not an empirical claim about real GAN/LLM retraining. |

## What is NOT a real, measured result (currently)

| Claim | Where it appears | Why it is not real |
|---|---|---|
| Per-domain ε-utility tradeoff curves for `tabular_hr`, `financial_transactions`, `healthcare_ehr` | `results/epsilon_sweep.json`, paper/draft.md Table 2 DP columns, README/DESIGN_DOC "Test Experiment 2" | `scripts/run_epsilon_sweep.py` contains a function literally named `_simulated_scores()` with the comment "Simulated per-ε scores (replace with real SDG runs when available)". The three asset types in `results/epsilon_sweep.json` have **identical numeric values** for every field at every ε — the simulator does not actually vary by domain. No DP-SGD training has been run for this experiment. The design doc's Test Experiment 2 ("Repeat tradeoff experiment on 5 enterprise domain types... Test whether utility-privacy tradeoff differs by domain") has not been executed. |
| "At ε=2, DP synthetic data trains ML models within 4% of real-data accuracy" (DESIGN_DOC.md key finding) | DESIGN_DOC.md, paper draft abstract/conclusion | This was the *expected* result written into the design doc before any DP-SGD experiment existed. The paper draft's Table 2 explicitly says "DP F1 values are estimated from calibrated domain retention curves anchored to the real oracle" — i.e., interpolated/extrapolated, not measured from actual DP-SGD training runs. |
| Downstream model impact by domain and ε (Test Experiment 3 in DESIGN_DOC.md) | DESIGN_DOC.md | No corresponding results file exists; `scripts/run_downstream_tasks.py` exists but its outputs are not in `results/`, so it is unclear whether it has been run end-to-end with real data. |

## Bottom line

The non-DP baseline experiment (CTGAN/TVAE/GaussianCopula utility ceiling) is
real and a legitimate, citable result. The actual DP-SGD tradeoff curve — the
centerpiece claim of the design doc — is currently a calibrated simulation, not
a measured result. Anyone using numbers from `paper/draft.md` Table 2 or
`results/epsilon_sweep.json` for a paper submission or external claim should
re-derive them from a real DP-SGD training run first.
