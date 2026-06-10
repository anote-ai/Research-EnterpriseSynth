FROM python:3.11-slim

WORKDIR /workspace

# System deps: git for pip-install-from-git; no GPU required for CPU-only repro
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install (editable so scripts can import src/)
COPY . /workspace/
RUN pip install --no-cache-dir -e ".[dev]"

# Pre-generate synthetic samples at multiple epsilon values so reviewers
# can run TSTR/utility evaluations without re-running DP training.
# Results land in /workspace/results/
RUN mkdir -p results paper/figures && \
    python scripts/run_epsilon_sweep.py --seeds 3 --json > results/epsilon_sweep_prebuilt.json && \
    python scripts/run_collapse_study.py --generations 5 --json > results/collapse_study_prebuilt.json && \
    python scripts/run_dp_mechanism_comparison.py --json > results/dp_mechanisms_prebuilt.json && \
    echo "Pre-built synthetic samples ready."

# Default: run the full reproduction suite
CMD ["bash", "run_all.sh"]
