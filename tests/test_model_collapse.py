import sys
import os

sys.path.append(os.path.abspath("src"))

from model_collapse.evaluator import (
    evaluate_model_collapse,
)


results = evaluate_model_collapse()

print("\n=== MODEL COLLAPSE RESULTS ===")

for result in results:
    print(result)