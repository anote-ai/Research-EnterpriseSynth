# EnterpriseSynth

[![CI](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml/badge.svg)](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Research Status

> See [RESEARCH_STATUS.md](RESEARCH_STATUS.md) for a full breakdown of which results are directly measured on real data vs. produced by calibrated simulation models.

**EnterpriseSynth** is a benchmark for differentially private synthetic tabular data generation in regulated enterprise domains. It measures the Privacy × Utility × Fidelity tradeoff across six DP budgets (ε ∈ {0.1, 0.5, 1, 2, 5, 10}, δ=1e-5) and six regulated data domains (HR, Healthcare EHR, Financial, IoT, E-commerce, Legal/DevOps).

**Key findings:**

- At ε=2, δ=1e-5, DP synthetic data retains 79–81% of real-data oracle F1 across tabular enterprise domains
- Financial time-series degrades fastest under DP (needs ε=5 to match HR utility at ε=2)
- Model collapse drops tail-record diversity 51% by generation 5 without mitigation; diversity-rewarded sampling keeps it within 10% of generation-0

**Note on scope:** The `src/enterprisesynth/` module includes an OpenAPI/SFT trace generation prototype from an earlier phase of the project. The primary research contribution — the DP benchmark — lives in `src/privacy_benchmark/`, `src/consistency/`, and `src/tstr_eval/`. See [DESIGN_DOC.md](DESIGN_DOC.md) and [paper/draft.md](paper/draft.md) for the full research framing.

## Research Significance

EnterpriseSynth is the first benchmark to simultaneously measure Privacy × Utility × Fidelity for differentially private synthetic data across six regulated enterprise domains. Prior work (SDGym, CTAB-GAN+) evaluates on generic tabular datasets without enterprise constraint verification; EnterpriseSynth adds constraint violation rate, domain-specific sensitivity multipliers, and a multi-generation model collapse study.

The framework is especially relevant for:

- Enterprise data teams evaluating DP synthesizers for GDPR/HIPAA/SOX compliance
- ML researchers studying the privacy-utility tradeoff in structured tabular data
- Compliance engineers mapping ε budgets to regulatory tiers
- Practitioners building synthetic data pipelines that require iterative retraining safety

## Pipeline

```
OpenAPI Schema (YAML/JSON)
        |
        v
  [SchemaParser]
        |
        v
   APISchema
  (Endpoints)
        |
        v
 [TraceGenerator]
  (seeded random)
        |
        v
  SFT Traces
  (instruction +
   tool_call +
   response +
   intent_spec)
        |
        v
  [verify_trace]
        |
        +---> verified=True  --> SFT Dataset
        |
        +---> verified=False --> Filter / Retry
        |
        v
 [dual_output_stats]
 [cold_start_score]
```

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/run_synth.py
```

```python
from enterprisesynth.core import SchemaParser, TraceGenerator
from enterprisesynth.data import crm_spec
from enterprisesynth.evaluate import verify_trace, dual_output_stats

schema = SchemaParser().parse_openapi(crm_spec())
gen = TraceGenerator(seed=42)
traces = gen.generate_traces(schema, n=10)

for trace in traces:
    trace.verified = verify_trace(trace, schema)

print(dual_output_stats(traces))
# {'verified_count': 10, 'total': 10, 'verified_rate': 1.0, 'unique_endpoints': 3}
```

## OpenAPI Snippet

```yaml
info:
  title: CRM API
  version: 1.0.0
paths:
  /contacts:
    post:
      operationId: createContact
      summary: Create a new contact
      parameters:
        - name: name
          type: string
          required: true
        - name: email
          type: string
          required: true
```

## Output Format

| Field | Type | Description |
|---|---|---|
| `trace_id` | `str` | Auto-generated 8-char UUID prefix |
| `instruction` | `str` | Natural language task description |
| `tool_call` | `dict` | `{name, arguments}` matching the OpenAPI operationId |
| `response` | `dict` | Synthetic `{status, data}` response |
| `intent_spec` | `str` | Human-readable intent statement |
| `verified` | `bool` | Whether all required params are present |

## Target Venues

- MLinPL 2026 (Machine Learning in Poland Conference)
- AAAI 2027 Workshop on Enterprise AI Evaluation

## Citation

```bibtex
@misc{enterprisesynth2026,
  title        = {EnterpriseSynth: Cold-Start SFT Trace Generation from Enterprise OpenAPI Schemas},
  author       = {Anote AI Research},
  year         = {2026},
  howpublished = {\url{https://github.com/anote-ai/research-enterprisesynth}},
  note         = {Preprint}
}
```
## Logical Consistency Benchmarking

This module evaluates inter-column logical consistency in synthetic enterprise tabular data.

Current features:
- HR schema constraint validation
- Dataset-level violation metrics
- CSV-based evaluation pipeline
- Constraint violation tracing

Example constraints:
- hire_date <= termination_date
- salary >= 0
- age consistency with birth_year

Example usage:

```bash
python tests/test_constraints.py
```
