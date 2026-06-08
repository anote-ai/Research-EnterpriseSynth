HR_SCHEMA = {
    "name": "hr_records",
    "constraints": [
        "hire_date_before_termination",
        "salary_positive",
        "age_birthyear_consistency"
    ]
}