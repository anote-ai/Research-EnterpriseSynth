from model_collapse.pipeline import run_model_collapse_pipeline
from model_collapse.mitigation import (
    apply_real_data_anchoring,
    apply_diversity_reward_sampling,
)


def evaluate_model_collapse():
    iterations = run_model_collapse_pipeline()

    results = []

    for item in iterations:
        anchored_entropy = apply_real_data_anchoring(
            item["coverage_entropy"]
        )

        diversity_entropy = apply_diversity_reward_sampling(
            item["coverage_entropy"]
        )

        results.append(
            {
                "iteration": item["iteration"],
                "base_entropy": item["coverage_entropy"],
                "tail_divergence": item["tail_divergence"],
                "minority_representation": item[
                    "minority_representation"
                ],
                "anchored_entropy": anchored_entropy,
                "diversity_reward_entropy": diversity_entropy,
            }
        )

    return results