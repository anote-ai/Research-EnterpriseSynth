#!/usr/bin/env python3
"""Demo: parse CRM spec, generate traces, verify, print stats."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser, TraceGenerator
from enterprisesynth.data import crm_spec
from enterprisesynth.evaluate import verify_trace, dual_output_stats, cold_start_score


def main():
    parser = SchemaParser()
    schema = parser.parse_openapi(crm_spec())
    print(f"Parsed schema: {schema.title} v{schema.version} ({len(schema.endpoints)} endpoints)")

    generator = TraceGenerator(seed=42)
    traces = generator.generate_traces(schema, n=10)
    print(f"Generated {len(traces)} traces")

    # Verify each trace against schema
    for trace in traces:
        trace.verified = verify_trace(trace, schema)

    stats = dual_output_stats(traces)
    score = cold_start_score(traces)
    print(f"Stats: {stats}")
    print(f"Cold-start score: {score:.3f}")

    for i, trace in enumerate(traces[:3]):
        print(f"  [{i}] {trace.instruction[:60]} | verified={trace.verified}")


if __name__ == "__main__":
    main()
