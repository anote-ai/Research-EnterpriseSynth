from __future__ import annotations

from typing import Sequence

from privacy_benchmark.metrics import (
    compute_privacy_score,
    compute_utility_score,
    compute_fidelity_score,
)
from privacy_benchmark.stats import (
    bootstrap_tstr_ci,
    dp_training_variance,
    verify_epsilon_accounting,
)


def evaluate_configuration(
    epsilon: float,
    auc: float,
    tstr_score: float,
    fidelity: float,
) -> dict:
    """Evaluate a single DP configuration. Returns scores dict."""
    privacy_score = compute_privacy_score(auc)
    utility_score = compute_utility_score(tstr_score)
    fidelity_score = compute_fidelity_score(fidelity)

    return {
        "epsilon": epsilon,
        "auc": auc,
        "privacy_score": privacy_score,
        "utility_score": utility_score,
        "fidelity_score": fidelity_score,
    }
