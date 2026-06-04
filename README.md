# EnterpriseSynth

[![CI](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml/badge.svg)](https://github.com/anote-ai/research-enterprisesynth/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **EnterpriseSynth — Agentic SFT + Eval Data from API Schemas Without Live Execution**

## The Cold-Start Problem

Fine-tuning or evaluating tool-calling agents on enterprise APIs requires paired (instruction, tool_call, response) traces. Collecting these traces in production requires live API access, test environments, and human annotation — a prohibitive cold-start cost for most enterprise deployments.

**EnterpriseSynth** solves this by ingesting OpenAPI/Swagger specifications and generating verified SFT traces + eval records *without ever calling the real API*. An LLM synthesises realistic (instruction, tool_call, response) triples; a schema-based verifier filters out hallucinated parameter names; surviving traces are dual-output as SFT training data and eval records annotated with intent specifications.

## Pipeline

```
 OpenAPI / Swagger Spec (YAML / JSON)
           │
    ┌──────▼──────┐
    │ SchemaParser │   parse_openapi()
    └──────┬──────┘
           │  APISchema (endpoints, parameters, response schemas)
    ┌──────▼──────────┐
    │ TraceGenerator   │   generate_traces(schema, n)
    │  (LLM-powered)   │
    └──────┬───────────┘
           │  list[SFTTrace]  (unverified)
    ┌──────▼──────────┐
    │ verify_trace()   │   schema-based param validation
    └──────┬───────────┘
           │
     ┌─────┴──────┐
     │            │
  Verified     Unverified
  SFT Data     Eval Records
```

## OpenAPI Example

```yaml
openapi: "3.0.0"
info:
  title: Petstore
  version: "1.0.0"
paths:
  /pets:
    get:
      operationId: listPets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
      responses:
        "200":
          description: A list of pets
```

`SchemaParser.parse_openapi()` will convert the above into an `APISchema` with one `Endpoint` whose `parameters` list contains `{"name": "limit", "in": "query"}`.

## Output Format

| Field | SFT Trace | Eval Record |
|-------|-----------|-------------|
| `instruction` | Natural-language task | Same |
| `tool_call` | LLM-generated | Same |
| `response` | Simulated | Same |
| `intent_spec` | Structured intent JSON | Same |
| `verified` | `True` (schema-valid) | `False` (may contain errors) |

## Quickstart

```bash
git clone https://github.com/anote-ai/research-enterprisesynth
cd research-enterprisesynth
pip install -e ".[dev]"
pytest tests/ -v
```

## Citation

```bibtex
@misc{anoteai2025enterprisesynth,
  title        = {EnterpriseSynth: Agentic SFT + Eval Data from API Schemas Without Live Execution},
  author       = {Anote AI Research},
  year         = {2025},
  howpublished = {\url{https://github.com/anote-ai/research-enterprisesynth}},
}
```

## License

MIT
