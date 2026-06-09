import sys
import os

sys.path.append(os.path.abspath("src"))

from tstr_eval.tstr import run_tstr_benchmark


results = run_tstr_benchmark()

print("\n=== TSTR EVALUATION RESULTS ===")

for result in results:
    print(result)