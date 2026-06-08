def apply_real_data_anchoring(score):
    """
    Simulates real-data anchoring mitigation.
    """
    return min(score + 0.08, 1.0)


def apply_diversity_reward_sampling(score):
    """
    Simulates diversity-rewarded sampling.
    """
    return min(score + 0.05, 1.0)