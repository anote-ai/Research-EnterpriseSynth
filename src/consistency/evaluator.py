from consistency.rules import (
    hire_date_before_termination,
    salary_positive,
    age_birthyear_consistency,
)

RULES = {
    "hire_date_before_termination": hire_date_before_termination,
    "salary_positive": salary_positive,
    "age_birthyear_consistency": age_birthyear_consistency,
}


def evaluate_row(row, schema):
    violations = []

    for rule_name in schema["constraints"]:
        rule_fn = RULES[rule_name]

        if not rule_fn(row):
            violations.append(rule_name)

    return violations


def evaluate_dataset(rows, schema):
    total_rows = len(rows)
    violating_rows = 0

    all_violations = []

    for row in rows:
        violations = evaluate_row(row, schema)

        if violations:
            violating_rows += 1

        all_violations.append(
            {
                "row": row,
                "violations": violations,
            }
        )

    violation_rate = violating_rows / total_rows

    return {
        "total_rows": total_rows,
        "violating_rows": violating_rows,
        "violation_rate": violation_rate,
        "details": all_violations,
    }