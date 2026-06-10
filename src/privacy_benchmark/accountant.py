"""DP privacy accountant: computes epsilon for standard mechanisms and compositions."""
from __future__ import annotations

import math


def _gaussian_epsilon(sigma: float, delta: float, sensitivity: float) -> float:
    if sigma <= 0 or delta <= 0 or delta >= 1:
        raise ValueError(f"Invalid params: sigma={sigma}, delta={delta}")
    # Invert Theorem A.1 from Dwork & Roth 2014:
    # sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma


def _laplace_epsilon(scale: float, sensitivity: float) -> float:
    if scale <= 0:
        raise ValueError(f"Invalid scale: {scale}")
    return sensitivity / scale


def _randomized_response_epsilon(p: float, k: int) -> float:
    """Compute epsilon from RR probability p for k-ary output.

    p = exp(ε) / (exp(ε) + k - 1)  →  ε = ln(p * (k-1) / (1-p))
    """
    if not (0 < p < 1) or k < 2:
        raise ValueError(f"Invalid params: p={p}, k={k}")
    return math.log(p * (k - 1) / (1 - p))


def compute_epsilon(
    mechanism: str,
    *,
    sensitivity: float = 1.0,
    sigma: float | None = None,
    scale: float | None = None,
    delta: float | None = None,
    p: float | None = None,
    k: int | None = None,
) -> float:
    """Compute ε for a given mechanism and parameter set.

    Supported mechanisms:
    - "gaussian"  : requires sigma, delta (sensitivity optional, default 1.0)
    - "laplace"   : requires scale (sensitivity optional, default 1.0)
    - "randomized_response": requires p, k

    Returns:
        The (ε, δ) privacy budget where ε is the return value and δ is the
        input delta (0 for pure-DP mechanisms).
    """
    if mechanism == "gaussian":
        if sigma is None or delta is None:
            raise ValueError("Gaussian mechanism requires sigma and delta")
        return _gaussian_epsilon(sigma, delta, sensitivity)

    if mechanism == "laplace":
        if scale is None:
            raise ValueError("Laplace mechanism requires scale")
        return _laplace_epsilon(scale, sensitivity)

    if mechanism == "randomized_response":
        if p is None or k is None:
            raise ValueError("Randomized response requires p and k")
        return _randomized_response_epsilon(p, k)

    raise ValueError(f"Unknown mechanism: {mechanism!r}")


def compose_basic(
    budgets: list[tuple[float, float]],
) -> tuple[float, float]:
    """Basic composition theorem.

    Args:
        budgets: List of (epsilon_i, delta_i) pairs for each sub-mechanism.

    Returns:
        (epsilon_total, delta_total) via simple summation.
    """
    eps_total = sum(e for e, _ in budgets)
    delta_total = sum(d for _, d in budgets)
    return eps_total, delta_total


def compose_advanced(
    budgets: list[tuple[float, float]],
    delta_prime: float = 1e-5,
) -> tuple[float, float]:
    """Advanced composition (Kairouz et al., 2015).

    Tighter than basic composition for k repetitions of the same mechanism.
    Requires all mechanisms to share the same (ε_i, δ_i).

    Args:
        budgets: List of (epsilon_i, delta_i) — all must be identical.
        delta_prime: Additional slack for the advanced bound.

    Returns:
        (epsilon_total, delta_total) via advanced composition.
    """
    if not budgets:
        return 0.0, 0.0
    k = len(budgets)
    eps_i = budgets[0][0]
    delta_i = budgets[0][1]
    if any(abs(e - eps_i) > 1e-10 or abs(d - delta_i) > 1e-10 for e, d in budgets):
        raise ValueError(
            "Advanced composition requires all sub-mechanisms to have identical budgets"
        )
    # Kairouz et al. Theorem 3:
    # ε_adv = ε_i * sqrt(2k ln(1/δ')) + k*ε_i*(exp(ε_i)-1)
    # We use the simpler form for small ε_i:
    eps_adv = eps_i * math.sqrt(2 * k * math.log(1 / delta_prime)) + k * eps_i * (
        math.exp(eps_i) - 1
    )
    delta_adv = k * delta_i + delta_prime
    return eps_adv, delta_adv


def gaussian_sigma_for_epsilon(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """Return the Gaussian noise σ needed to achieve (ε, δ)-DP.

    Implements the calibration formula from Dwork & Roth 2014, Theorem A.1:
        σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε
    """
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError(f"Invalid params: epsilon={epsilon}, delta={delta}")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
