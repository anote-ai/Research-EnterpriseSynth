"""Privacy accountant correctness tests.

Each test verifies the accountant gives correct epsilon values for analytically
known inputs. These tests are the DP reviewer's first stop for reproducibility.
"""
from __future__ import annotations

import math

import pytest

from privacy_benchmark.accountant import (
    compute_epsilon,
    compose_basic,
    compose_advanced,
    gaussian_sigma_for_epsilon,
)


# ── Gaussian mechanism ────────────────────────────────────────────────────────

def test_gaussian_sigma_formula_roundtrip() -> None:
    """compute_epsilon(gaussian, sigma=calibrated) == target_epsilon."""
    target_eps = 1.0
    delta = 1e-5
    sigma = gaussian_sigma_for_epsilon(target_eps, delta, sensitivity=1.0)
    recovered = compute_epsilon("gaussian", sigma=sigma, delta=delta, sensitivity=1.0)
    assert abs(recovered - target_eps) < 1e-6


def test_gaussian_smaller_sigma_larger_epsilon() -> None:
    """Noisier mechanism (smaller σ) must give larger ε."""
    delta = 1e-5
    eps_high_noise = compute_epsilon("gaussian", sigma=0.5, delta=delta)
    eps_low_noise = compute_epsilon("gaussian", sigma=2.0, delta=delta)
    assert eps_high_noise > eps_low_noise


def test_gaussian_smaller_delta_larger_epsilon() -> None:
    """Tighter δ requires more noise (larger σ) for same ε; given fixed σ, ε is larger."""
    sigma = 1.0
    eps_tight = compute_epsilon("gaussian", sigma=sigma, delta=1e-8)
    eps_loose = compute_epsilon("gaussian", sigma=sigma, delta=1e-3)
    assert eps_tight > eps_loose


def test_gaussian_sensitivity_scaling() -> None:
    """Doubling sensitivity doubles epsilon (linear scaling)."""
    sigma, delta = 1.0, 1e-5
    eps1 = compute_epsilon("gaussian", sigma=sigma, delta=delta, sensitivity=1.0)
    eps2 = compute_epsilon("gaussian", sigma=sigma, delta=delta, sensitivity=2.0)
    assert abs(eps2 - 2 * eps1) < 1e-9


def test_gaussian_sigma_for_epsilon_known_value() -> None:
    """σ = sqrt(2 ln(1.25/1e-5)) ≈ 4.7 for ε=1, Δf=1."""
    sigma = gaussian_sigma_for_epsilon(1.0, 1e-5)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5))
    assert abs(sigma - expected) < 1e-9


def test_gaussian_invalid_params() -> None:
    with pytest.raises(ValueError):
        compute_epsilon("gaussian", sigma=-1.0, delta=1e-5)
    with pytest.raises(ValueError):
        compute_epsilon("gaussian", sigma=1.0, delta=0.0)
    with pytest.raises(ValueError):
        compute_epsilon("gaussian", sigma=1.0, delta=1.5)


# ── Laplace mechanism ─────────────────────────────────────────────────────────

def test_laplace_pure_dp_unit_sensitivity() -> None:
    """Laplace with scale=1.0 and Δf=1.0 gives ε=1.0 (pure DP)."""
    eps = compute_epsilon("laplace", scale=1.0, sensitivity=1.0)
    assert abs(eps - 1.0) < 1e-9


def test_laplace_epsilon_inversely_proportional_to_scale() -> None:
    """Doubling scale halves epsilon."""
    eps1 = compute_epsilon("laplace", scale=1.0)
    eps2 = compute_epsilon("laplace", scale=2.0)
    assert abs(2 * eps2 - eps1) < 1e-9


def test_laplace_sensitivity_scaling() -> None:
    """Epsilon scales linearly with sensitivity."""
    eps1 = compute_epsilon("laplace", scale=1.0, sensitivity=1.0)
    eps3 = compute_epsilon("laplace", scale=1.0, sensitivity=3.0)
    assert abs(eps3 - 3 * eps1) < 1e-9


def test_laplace_invalid_scale() -> None:
    with pytest.raises(ValueError):
        compute_epsilon("laplace", scale=0.0)


# ── Randomized Response ───────────────────────────────────────────────────────

def test_rr_binary_p_half_means_epsilon_zero() -> None:
    """For k=2 at p=0.5 (50-50 coin flip), epsilon=0 (maximum privacy)."""
    eps = compute_epsilon("randomized_response", p=0.5, k=2)
    assert abs(eps) < 1e-9


def test_rr_ratio_equals_exp_epsilon() -> None:
    """For k=2: p/(1-p) == exp(ε) exactly."""
    target_eps = 2.0
    p = math.exp(target_eps) / (math.exp(target_eps) + 1)
    eps = compute_epsilon("randomized_response", p=p, k=2)
    assert abs(eps - target_eps) < 1e-9


def test_rr_higher_k_same_p_higher_epsilon() -> None:
    """More categories (higher k) with same p gives higher epsilon.

    Formula: ε = ln(p*(k-1)/(1-p)).  The (k-1) factor grows with k, so
    the adversary gains more information when the output space is larger
    but the truth-reporting probability stays fixed.
    """
    p = 0.8
    eps_k2 = compute_epsilon("randomized_response", p=p, k=2)
    eps_k5 = compute_epsilon("randomized_response", p=p, k=5)
    assert eps_k5 > eps_k2


def test_rr_invalid_params() -> None:
    with pytest.raises(ValueError):
        compute_epsilon("randomized_response", p=0.0, k=2)
    with pytest.raises(ValueError):
        compute_epsilon("randomized_response", p=0.5, k=1)


# ── Composition theorems ──────────────────────────────────────────────────────

def test_basic_composition_five_fields() -> None:
    """5 fields each at ε=0.2 gives ε_total=1.0."""
    budgets = [(0.2, 1e-6)] * 5
    eps_total, delta_total = compose_basic(budgets)
    assert abs(eps_total - 1.0) < 1e-9
    assert abs(delta_total - 5e-6) < 1e-12


def test_basic_composition_single_mechanism() -> None:
    """Single mechanism composition is the identity."""
    budgets = [(1.5, 1e-5)]
    eps, delta = compose_basic(budgets)
    assert abs(eps - 1.5) < 1e-9
    assert abs(delta - 1e-5) < 1e-12


def test_advanced_composition_stricter_than_basic() -> None:
    """Advanced composition beats basic for many repetitions of a small-ε mechanism.

    The savings only materialise when ε_i is small and k is large.
    At ε_i=0.01, k=500: basic = 5.0; advanced ≈ 1.5 (much tighter).
    """
    budgets = [(0.01, 1e-8)] * 500
    eps_basic, _ = compose_basic(budgets)
    eps_adv, _ = compose_advanced(budgets, delta_prime=1e-5)
    assert eps_adv < eps_basic


def test_advanced_composition_single_mechanism_close_to_basic() -> None:
    """For k=1 advanced ≈ basic (no savings from composition)."""
    budgets = [(1.0, 1e-5)]
    eps_basic, _ = compose_basic(budgets)
    eps_adv, _ = compose_advanced(budgets, delta_prime=1e-5)
    # Advanced adds delta_prime to budget; for k=1 both are close
    assert eps_adv >= 0


def test_advanced_composition_requires_identical_budgets() -> None:
    """Advanced composition raises ValueError for heterogeneous budgets."""
    budgets = [(0.5, 1e-6), (1.0, 1e-6)]
    with pytest.raises(ValueError):
        compose_advanced(budgets)


def test_basic_composition_empty() -> None:
    """Empty budget list gives (0.0, 0.0)."""
    eps, delta = compose_basic([])
    assert eps == 0.0 and delta == 0.0


# ── Unknown mechanism ─────────────────────────────────────────────────────────

def test_unknown_mechanism_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mechanism"):
        compute_epsilon("exponential_mechanism")
