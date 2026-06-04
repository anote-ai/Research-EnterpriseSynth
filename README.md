# EnterpriseSynth

[![CI](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml/badge.svg)](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Cold-Start Motivation

Fine-tuning LLM agents for enterprise tool-use requires thousands of high-quality (instruction, tool_call, response) traces — data that enterprises rarely have at day zero. EnterpriseSynth solves the cold-start problem by automatically generating verified SFT traces directly from OpenAPI schemas, without requiring any real user interactions or API calls.

By grounding generation in the schema's operation semantics, EnterpriseSynth produces traces that are structurally valid by construction. A lightweight verification layer checks parameter completeness against the schema, yielding a `cold_start_score` that reflects both verification rate and endpoint coverage diversity — a proxy for how much of the API surface the synthetic traces explore.

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
