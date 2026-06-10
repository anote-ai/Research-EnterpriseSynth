def compute_privacy_score(auc):
    """
    Lower AUC means stronger privacy.
    """
    return 1 - auc


def compute_utility_score(tstr_score):
    """
    TSTR utility score.
    """
    return tstr_score


def compute_fidelity_score(fidelity):
    """
    Statistical fidelity score.
    """
    return fidelity