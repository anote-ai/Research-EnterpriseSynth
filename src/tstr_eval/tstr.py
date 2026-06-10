from tstr_eval.evaluator import evaluate_document_category


def run_tstr_benchmark():
    categories = [
        {
            "category": "contracts",
            "bertscore": 0.91,
            "mauve": 0.88,
            "ner_score": 0.93,
            "tstr_f1": 0.89,
        },
        {
            "category": "support_tickets",
            "bertscore": 0.87,
            "mauve": 0.84,
            "ner_score": 0.85,
            "tstr_f1": 0.83,
        },
        {
            "category": "compliance_reports",
            "bertscore": 0.90,
            "mauve": 0.86,
            "ner_score": 0.91,
            "tstr_f1": 0.88,
        },
        {
            "category": "hr_memos",
            "bertscore": 0.88,
            "mauve": 0.82,
            "ner_score": 0.87,
            "tstr_f1": 0.84,
        },
    ]

    results = []

    for item in categories:
        evaluated = evaluate_document_category(
            category=item["category"],
            bertscore=item["bertscore"],
            mauve=item["mauve"],
            ner_score=item["ner_score"],
            tstr_f1=item["tstr_f1"],
        )

        results.append(evaluated)

    return results