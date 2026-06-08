from tstr_eval.metrics import (
    compute_bertscore,
    compute_mauve,
)

from tstr_eval.ner import (
    compute_ner_consistency,
)


def evaluate_document_category(
    category,
    bertscore,
    mauve,
    ner_score,
    tstr_f1,
):
    return {
        "category": category,
        "bertscore": compute_bertscore(bertscore),
        "mauve": compute_mauve(mauve),
        "ner_consistency": compute_ner_consistency(ner_score),
        "tstr_f1": tstr_f1,
    }