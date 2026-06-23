"""Pareto frontier computation for the utility-privacy-fidelity tradeoff study."""
from __future__ import annotations

from privacy_benchmark.evaluator import evaluate_configuration


def generate_pareto_frontier(configurations: list[dict]) -> list[dict]:
    """Evaluate each configuration and return a list of result dicts."""
    results = []
    for config in configurations:
        evaluated = evaluate_configuration(
            epsilon=config["epsilon"],
            auc=config["auc"],
            tstr_score=config["tstr_score"],
            fidelity=config["fidelity"],
        )
        results.append(evaluated)
    return results


def is_pareto_efficient(results: list[dict]) -> list[bool]:
    """Return a boolean mask indicating which configurations are Pareto-efficient.

    A configuration is Pareto-efficient if no other configuration dominates it
    on ALL THREE dimensions simultaneously:
      - Higher privacy_score (lower AUC = stronger privacy)
      - Higher utility_score (higher TSTR = better utility)
      - Higher fidelity_score

    Args:
        results: List of dicts with keys privacy_score, utility_score, fidelity_score.

    Returns:
        List of booleans, same length as results.
    """
    n = len(results)
    efficient = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if it is ≥ on all dimensions and > on at least one
            j_dom = (
                results[j]["privacy_score"] >= results[i]["privacy_score"]
                and results[j]["utility_score"] >= results[i]["utility_score"]
                and results[j]["fidelity_score"] >= results[i]["fidelity_score"]
                and (
                    results[j]["privacy_score"] > results[i]["privacy_score"]
                    or results[j]["utility_score"] > results[i]["utility_score"]
                    or results[j]["fidelity_score"] > results[i]["fidelity_score"]
                )
            )
            if j_dom:
                efficient[i] = False
                break
    return efficient


def pareto_efficient_configs(results: list[dict]) -> list[dict]:
    """Return only the Pareto-efficient configurations."""
    mask = is_pareto_efficient(results)
    return [r for r, keep in zip(results, mask) if keep]


def compare_domains(
    domain_results: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Summarise Pareto metrics across domains.

    Args:
        domain_results: Mapping from domain name to list of evaluated configs
            (each config is a dict with epsilon, privacy_score, utility_score,
            fidelity_score).

    Returns:
        Mapping from domain name to summary dict with:
          - n_pareto: number of Pareto-efficient configurations
          - best_utility: max utility_score across all configs
          - best_privacy: max privacy_score across all configs
          - utility_at_eps2: utility_score at ε=2.0 (or nearest)
          - epsilon_for_90pct_utility: minimum ε yielding utility_score ≥ 0.90
    """
    summary: dict[str, dict[str, float]] = {}
    for domain, configs in domain_results.items():
        mask = is_pareto_efficient(configs)
        n_pareto = sum(mask)
        best_utility = max(c["utility_score"] for c in configs)
        best_privacy = max(c["privacy_score"] for c in configs)

        # Utility at ε=2.0
        eps2_configs = [c for c in configs if abs(c["epsilon"] - 2.0) < 1e-6]
        utility_at_eps2 = eps2_configs[0]["utility_score"] if eps2_configs else float("nan")

        # Minimum ε to achieve ≥90% utility
        eps_90 = float("nan")
        for c in sorted(configs, key=lambda x: x["epsilon"]):
            if c["utility_score"] >= 0.90:
                eps_90 = c["epsilon"]
                break

        summary[domain] = {
            "n_pareto": float(n_pareto),
            "best_utility": best_utility,
            "best_privacy": best_privacy,
            "utility_at_eps2": utility_at_eps2,
            "epsilon_for_90pct_utility": eps_90,
        }
    return summary