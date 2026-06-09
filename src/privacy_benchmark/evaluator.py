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


def evaluate_with_ci(
    epsilon: float,
    auc: float,
    tstr_scores: Sequence[float],
    fidelity: float,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Evaluate a configuration and include 95% bootstrap CIs on TSTR utility.

    *tstr_scores* should contain record-level TSTR scores for resampling.
    """
    ci_result = bootstrap_tstr_ci(
        tstr_scores,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )
    base = evaluate_configuration(epsilon, auc, ci_result["mean"], fidelity)
    base["tstr_ci_lower"] = ci_result["ci_lower"]
    base["tstr_ci_upper"] = ci_result["ci_upper"]
    base["tstr_ci_half_width"] = ci_result["ci_half_width"]
    base["tstr_n_samples"] = float(len(tstr_scores))
    return base


def evaluate_multi_seed(
    epsilon: float,
    seed_configs: Sequence[dict],
    *,
    computed_epsilon: float | None = None,
    epsilon_tolerance: float = 0.01,
) -> dict:
    """Evaluate one ε budget across multiple DP training seeds.

    *seed_configs* is a list of dicts, each with keys: auc, tstr_score, fidelity.
    Optionally runs a privacy accountant check when *computed_epsilon* is given.
    """
    evaluated = [
        evaluate_configuration(
            epsilon,
            cfg["auc"],
            cfg["tstr_score"],
            cfg["fidelity"],
        )
        for cfg in seed_configs
    ]

    score_keys = ["privacy_score", "utility_score", "fidelity_score"]
    seed_results = [{k: e[k] for k in score_keys} for e in evaluated]
    variance = dp_training_variance(seed_results)

    result: dict = {
        "epsilon": epsilon,
        "n_seeds": len(seed_configs),
        "variance": variance,
    }

    if computed_epsilon is not None:
        result["epsilon_check"] = verify_epsilon_accounting(
            epsilon,
            computed_epsilon,
            tolerance=epsilon_tolerance,
        )

    return result
