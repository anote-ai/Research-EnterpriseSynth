from model_collapse.metrics import (
    compute_entropy,
    compute_tail_divergence,
    compute_minority_representation,
)


def simulate_generation_iteration(
    iteration,
    entropy,
    tail_divergence,
    minority_score,
):
    return {
        "iteration": iteration,
        "coverage_entropy": compute_entropy(entropy),
        "tail_divergence": compute_tail_divergence(tail_divergence),
        "minority_representation": compute_minority_representation(
            minority_score
        ),
    }


def run_model_collapse_pipeline():
    iterations = [
        simulate_generation_iteration(1, 0.95, 0.05, 0.94),
        simulate_generation_iteration(2, 0.91, 0.08, 0.90),
        simulate_generation_iteration(3, 0.87, 0.12, 0.85),
        simulate_generation_iteration(4, 0.82, 0.17, 0.79),
        simulate_generation_iteration(5, 0.78, 0.22, 0.72),
    ]

    return iterations