from datetime import datetime


def hire_date_before_termination(row):
    """
    Ensure hire date is before termination date.
    """
    if not row.get("termination_date"):
        return True

    hire = datetime.strptime(row["hire_date"], "%Y-%m-%d")
    termination = datetime.strptime(row["termination_date"], "%Y-%m-%d")

    return hire <= termination


def salary_positive(row):
    """
    Salary must be positive.
    """
    return row["salary"] >= 0


def age_birthyear_consistency(row):
    """
    Age should roughly match birth year.
    """
    current_year = datetime.now().year
    expected_age = current_year - row["birth_year"]

    return abs(expected_age - row["age"]) <= 1