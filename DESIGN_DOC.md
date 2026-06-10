# EnterpriseSynth — Research Design Document

## Goal

Demonstrate that differential privacy (DP) synthetic data generation can meet enterprise utility requirements while providing formal privacy guarantees, and produce the first systematic characterization of the utility-privacy tradeoff across enterprise schema types.

## Objective

1. Build a DP synthetic data generation pipeline supporting tabular, time-series, and relational data with foreign key preservation
2. Evaluate utility (TSTR accuracy, statistical similarity, downstream model performance) vs. privacy (ε, membership inference resistance) tradeoffs across 5 enterprise domain types
3. Produce a practical ε selection guide: for a given utility requirement and regulatory context, what ε value is appropriate?

## Background / Motivation

Enterprise organizations need to share and train on sensitive data (medical, financial, HR) but face GDPR, HIPAA, and CCPA compliance requirements. Synthetic data is the most promising solution, but organizations don't trust it because: (1) most commercial tools don't provide formal DP guarantees; (2) no one has published a systematic characterization of how utility degrades as ε tightens in enterprise settings.

## Experimental Design

### Baseline Experiment

**Evaluate 3 non-DP synthetic data baselines (CTGAN, TVAE, Gaussian copula) on 3 public tabular datasets using Train-on-Synthetic Test-on-Real (TSTR) accuracy**

- Metric: TSTR accuracy, Wasserstein distance on marginals, column correlation preservation
- Purpose: establish the utility ceiling — what's achievable without DP?
- Expected result: CTGAN TSTR accuracy ≈ 85–90% of real-data training accuracy

### Test Experiment 1: DP Utility-Privacy Tradeoff Curve

Apply DP-SGD training at ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}. For each ε: measure TSTR accuracy, Wasserstein distance, and membership inference attack success. Run 5 seeds; report mean ± std. Plot full utility-privacy Pareto curve.

**Expected result:** dramatic utility drop below ε=1.0 for standard DP-SGD; our improved training procedure reduces utility loss at ε=1.0 to <10% vs. non-DP baseline

### Test Experiment 2: Enterprise Schema Types

Repeat tradeoff experiment on 5 enterprise domain types: healthcare, financial transactions, HR, IoT sensor, e-commerce (relational). Test whether utility-privacy tradeoff differs by domain.

**Expected result:** time-series and relational data lose utility faster under DP; domain-specific ε recommendations emerge

### Test Experiment 3: Downstream Model Impact

For each domain and ε: train a task-specific model on synthetic data, evaluate on real held-out data, compare to model trained on real data.

**Expected result:** at ε=2.0, downstream model accuracy within 3–5% of real-data accuracy — acceptable for most enterprise use cases

## Expected Results

1. A DP synthetic data generation system supporting tabular, time-series, and relational data
2. Full utility-privacy Pareto curves for 5 enterprise domain types
3. Downstream model accuracy impact by domain and ε
4. **Key finding:** "At ε=2.0, DP synthetic data trains ML models within 4% of real-data accuracy"
5. A practical ε selection guide: domain type + acceptable utility loss → recommended ε range

## Why This Matters / Why People Would Care

- **Enterprise data teams:** want to use synthetic data but don't know what ε to choose or whether utility will be acceptable
- **Compliance officers:** GDPR and HIPAA compliance community needs formal privacy guarantees
- **Healthcare and financial institutions:** largest holders of sensitive data and highest-value use cases
- **AI researchers:** enterprise schema support and domain-specific tradeoff characterization are novel contributions

## Timeline

| Month | Milestone |
|---|---|
| 1–2 | System implementation (DP training, relational support, time-series support) |
| 3 | Baseline evaluation on 3 public datasets |
| 4 | DP tradeoff curves across all ε values and domain types |
| 5 | Downstream model impact experiments |
| 6 | Submission to NeurIPS 2026 D&B track |

## Related Issues

- Design doc GitHub issue: #35
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
