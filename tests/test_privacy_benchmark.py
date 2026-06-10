import sys
import os

sys.path.append(os.path.abspath("src"))

from privacy_benchmark.pareto import generate_pareto_frontier


sample_configs = [
    {
        "epsilon": 0.1,
        "auc": 0.52,
        "tstr_score": 0.75,
        "fidelity": 0.80,
    },
    {
        "epsilon": 1.0,
        "auc": 0.60,
        "tstr_score": 0.88,
        "fidelity": 0.91,
    },
    {
        "epsilon": 10.0,
        "auc": 0.72,
        "tstr_score": 0.95,
        "fidelity": 0.97,
    },
]

results = generate_pareto_frontier(sample_configs)

print("\n=== PRIVACY BENCHMARK RESULTS ===")

for result in results:
    print(result)