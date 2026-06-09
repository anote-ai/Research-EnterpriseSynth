from privacy_benchmark.evaluator import evaluate_configuration


def generate_pareto_frontier(configurations):
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