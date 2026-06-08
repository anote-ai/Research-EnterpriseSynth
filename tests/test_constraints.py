import sys
import os

sys.path.append(os.path.abspath("src"))

from consistency.data_loader import load_csv
from consistency.evaluator import evaluate_dataset
from consistency.schemas import HR_SCHEMA

rows = load_csv("examples/hr_sample.csv")

results = evaluate_dataset(rows, HR_SCHEMA)

print("\n=== DATASET EVALUATION ===")
print(f"Total Rows: {results['total_rows']}")
print(f"Violating Rows: {results['violating_rows']}")
print(f"Violation Rate: {results['violation_rate']:.2%}")

print("\n=== DETAILS ===")

for item in results["details"]:
    print(item)