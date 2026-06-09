from privacy_benchmark.metrics import (
    compute_privacy_score,
    compute_utility_score,
    compute_fidelity_score,
)


def evaluate_configuration(epsilon, auc, tstr_score, fidelity):
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