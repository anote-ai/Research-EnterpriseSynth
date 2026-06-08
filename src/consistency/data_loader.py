import csv


def load_csv(filepath):
    rows = []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed_row = {
                "hire_date": row["hire_date"],
                "termination_date": row["termination_date"],
                "salary": float(row["salary"]),
                "birth_year": int(row["birth_year"]),
                "age": int(row["age"]),
            }

            rows.append(parsed_row)

    return rows