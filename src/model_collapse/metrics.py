def compute_entropy(entropy_score):
    """
    Diversity coverage entropy.
    Higher is better.
    """
    return entropy_score


def compute_tail_divergence(divergence):
    """
    Statistical tail divergence.
    Lower is better.
    """
    return divergence


def compute_minority_representation(score):
    """
    Minority class retention score.
    """
    return score