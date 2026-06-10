# DP Privacy Analysis: Threat Model, Proof Sketches, and Reviewer Guide

*Supplementary material for EnterpriseSynth — attach to TPDP submission.*

---

## 1. Threat Model

### 1.1 Adversary Definition

We consider a **semi-honest membership inference adversary** with the following capabilities:

| Property | Value |
|---|---|
| Adversary type | Black-box (output only) — cannot inspect model weights or training procedure |
| Auxiliary knowledge | Full knowledge of the synthetic data distribution (worst-case) |
| Attack target | Determine whether a specific record `r` was in the training set used to produce synthetic dataset `S` |
| Attack mechanism | Shadow model membership inference attack (Shokri et al., 2017) |
| Success metric | AUC of binary classifier trained to distinguish training vs. non-training records |

### 1.2 What the Adversary Cannot Do

- Access intermediate computations or noise realizations
- Query the generator with arbitrary inputs (black-box restriction)
- Obtain multiple synthetic datasets generated from the same model (single-release model)

### 1.3 Privacy Guarantee Scope

Our DP guarantees apply to **tabular record release**: each individual record in the training dataset contributes noise before any statistics are computed. The guarantee is:

> For any two adjacent datasets `D` and `D'` differing in exactly one record, and for any measurable output set `S`:
> `Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ`

**What is not covered**: Document synthesis (contracts, HR memos) uses an LLM-based generator; the DP guarantee for document synthesis is approximate and discussed separately in Section 2.4.

---

## 2. Mechanism-Specific Proof Sketches

### 2.1 Gaussian Mechanism

**Claim**: The Gaussian mechanism `M(x) = f(x) + N(0, σ²I)` satisfies `(ε, δ)`-DP for:
```
σ = Δf · √(2 ln(1.25/δ)) / ε
```
where `Δf` is the L2 sensitivity of `f`.

**Proof sketch** (standard result, Dwork & Roth 2014, Theorem A.1):
1. For adjacent `D, D'`: `|f(D) - f(D')| ≤ Δf` by sensitivity definition.
2. The ratio of Gaussian densities satisfies:
   `ln(p(y|D) / p(y|D')) = (2〈f(D) - f(D'), y - f(D')〉 - |f(D) - f(D')|²) / (2σ²)`
3. With probability ≥ 1-δ over the noise draw, this ratio is bounded by ε.
4. Setting σ as above achieves the bound. ∎

**Implementation reference**: `src/privacy_benchmark/mechanisms.py` → `gaussian_mechanism()`
**Unit test**: `tests/test_privacy_accountant.py` → `test_gaussian_sigma_formula()`

---

### 2.2 Laplace Mechanism

**Claim**: The Laplace mechanism `M(x) = f(x) + Lap(0, Δf/ε)` satisfies `ε`-DP (pure DP, δ=0).

**Proof sketch** (standard result, Dwork et al. 2006):
1. `ln(p(y|D) / p(y|D')) = ε/Δf · (|y - f(D')| - |y - f(D)|)`
2. By triangle inequality: `|y - f(D')| - |y - f(D)| ≤ |f(D) - f(D')| ≤ Δf`
3. Therefore `ln(ratio) ≤ ε`. ∎

**Implementation reference**: `src/privacy_benchmark/mechanisms.py` → `laplace_mechanism()`
**Unit test**: `tests/test_privacy_accountant.py` → `test_laplace_pure_dp()`

---

### 2.3 Discrete Gaussian Mechanism

**Claim**: The Discrete Gaussian mechanism `M(x) = f(x) + DG(σ)`, where `DG(σ)` draws from the discrete Gaussian `∝ exp(-z²/(2σ²))` over integers, satisfies `(ε, δ)`-DP.

**Reference**: Canonne, Kamath, Steinke (2020) — "The Discrete Gaussian for Differential Privacy."

**Why we use it for integer fields**: The standard Gaussian mechanism is defined over reals; when applied to integer-valued fields (salary, age, count) and then rounded, the actual DP guarantee degrades. The Discrete Gaussian applies noise directly in the integer domain, recovering the theoretical guarantee without rounding artifacts.

**ε-σ correspondence** (from Canonne et al. Proposition 11):
```
For σ ≥ 1/ε and δ ≥ 0, DG(σ) satisfies (ε, δ)-zCDP with ρ = 1/(2σ²)
```

**Implementation reference**: `src/privacy_benchmark/mechanisms.py` → `discrete_gaussian_mechanism()`
**Unit test**: `tests/test_privacy_accountant.py` → `test_discrete_gaussian_integer_fields()`

---

### 2.4 Randomized Response (for Categorical Fields)

**Claim**: Randomized Response with parameter `p = exp(ε)/(exp(ε) + k-1)` satisfies `ε`-DP for k-ary categorical variables.

**Proof sketch**:
1. For any output value `y` and any pair of adjacent inputs `x, x'`:
   - If `x = y`: `Pr[M(x) = y] = p`
   - If `x ≠ y`: `Pr[M(x) = y] = (1-p)/(k-1)`
2. `Pr[M(x) = y] / Pr[M(x') = y] ≤ p / ((1-p)/(k-1)) = exp(ε)` ∎

**Implementation reference**: `src/privacy_benchmark/mechanisms.py` → `randomized_response()`
**Unit test**: `tests/test_privacy_accountant.py` → `test_randomized_response_ratio()`

---

## 3. Composition Theorem

For the multi-field tabular synthesis pipeline, we apply **basic composition**:

> **Theorem** (Basic Composition): If `M₁` is `(ε₁, δ₁)`-DP and `M₂` is `(ε₂, δ₂)`-DP and they are run on the same dataset, the composed mechanism is `(ε₁+ε₂, δ₁+δ₂)`-DP.

For k fields each with budget `ε_per_field`:
```
ε_total = k × ε_per_field
δ_total = k × δ_per_field
```

**Advanced composition** (Kairouz et al. 2015) improves this to:
```
ε_total ≈ ε_per_field × √(2k ln(1/δ'))  for any δ' > 0
```

We use basic composition in all reported results to give conservative (guaranteed) bounds. The privacy accountant is implemented in `src/privacy_benchmark/accountant.py`.

---

## 4. Privacy Accountant Unit Tests

The file `tests/test_privacy_accountant.py` verifies correctness of the accountant for known inputs.

**Key test cases**:
1. Gaussian σ formula matches Theorem A.1 to within 1e-6
2. Laplace budget for ε=1.0, Δf=1.0 gives scale=1.0
3. Composition of 5 fields at ε=0.2 each gives ε_total=1.0 (basic)
4. Advanced composition is strictly less than basic composition for k≥2
5. RR ratio test: for k=2 categories, `p/(1-p) = exp(ε)` exactly

---

## 5. Limitations and Open Questions

| Limitation | Impact | Mitigation |
|---|---|---|
| Gaussian mechanism applied to bounded integers (salary, age) | Actual ε may be slightly weaker than claimed | Use Discrete Gaussian for all integer fields (recommended) |
| Document synthesis uses LLM with DP fine-tuning (DP-SGD) | Per-token ε accounting is approximate | Report RDP accountant bounds; note "approximate DP" in paper |
| Composition across multiple synthetic releases | Each new release consumes budget | Enforce single-release policy in product; document in threat model |
| Side-channel: schema structure leaks field names | Not covered by DP | Acceptable — schema is assumed public |

---

## 6. Reviewer Verification Guide

A privacy reviewer can verify the DP guarantees in under 2 hours:

```bash
# 1. Clone and install (5 min)
git clone https://github.com/anote-ai/Research-EnterpriseSynth
cd Research-EnterpriseSynth
pip install -e ".[dev]"

# 2. Run privacy accountant unit tests (2 min)
pytest tests/test_privacy_accountant.py -v

# 3. Verify mechanism outputs match ε claims (5 min)
python - <<'EOF'
from privacy_benchmark.mechanisms import gaussian_mechanism
from privacy_benchmark.accountant import compute_epsilon
import math

# Gaussian at sigma=1.0, delta=1e-5 should give epsilon~1.0
sigma = math.sqrt(2 * math.log(1.25 / 1e-5))  # sigma for eps=1.0
eps = compute_epsilon(mechanism="gaussian", sigma=sigma, delta=1e-5, sensitivity=1.0)
print(f"Gaussian epsilon: {eps:.4f} (expected ~1.0)")

# Laplace at scale=1.0 should give epsilon=1.0
eps_lap = compute_epsilon(mechanism="laplace", scale=1.0, sensitivity=1.0)
print(f"Laplace epsilon: {eps_lap:.4f} (expected 1.0)")
EOF

# 4. Run empirical MIA audit for spot-check (10 min)
python scripts/run_product_audit.py --investigation 2

# 5. Reproduce Table 6 (DP mechanism comparison) (5 min)
python scripts/run_dp_mechanism_comparison.py
```

All steps should complete on CPU in under 30 minutes without GPU.
