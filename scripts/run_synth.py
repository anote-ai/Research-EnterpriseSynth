#!/usr/bin/env python3
"""Demo: parse enterprise API specs, generate traces, verify, and print stats."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser, TraceGenerator
from enterprisesynth.data import crm_spec, finance_spec, devops_spec
from enterprisesynth.evaluate import (
    verify_trace,
    dual_output_stats,
    cold_start_score,
    trace_diversity_score,
    schema_coverage_score,
)


def run_demo(spec_fn, label: str, seed: int = 42, n: int = 10) -> None:
    parser = SchemaParser()
    schema = parser.parse_openapi(spec_fn())
    print(f"\n{'=' * 60}")
    print(f"Schema: {schema.title} v{schema.version} ({len(schema.endpoints)} endpoints)")

    generator = TraceGenerator(seed=seed)
    traces = generator.generate_traces(schema, n=n)

    for trace in traces:
        trace.verified = verify_trace(trace, schema)

    stats = dual_output_stats(traces)
    diversity = trace_diversity_score(traces)
    coverage = schema_coverage_score(traces, schema)
    score = cold_start_score(traces)

    print(f"  Traces generated : {stats['total']}")
    print(f"  Verified         : {stats['verified_count']} ({stats['verified_rate']:.0%})")
    print(f"  Unique endpoints : {stats['unique_endpoints']}")
    print(f"  Schema coverage  : {coverage:.2f}")
    print(f"  Diversity score  : {diversity:.3f}")
    print(f"  Cold-start score : {score:.3f}")

    print("\n  Sample traces:")
    for i, trace in enumerate(traces[:3]):
        print(f"    [{i}] {trace.instruction[:60]} | verified={trace.verified}")

    print("\n  Multi-step trace:")
    multi = generator.generate_multi_step_trace(schema, n_steps=3)
    print(f"    steps={len(multi.steps)} | {multi.instruction[:70]}")

    print("\n  Error-injected trace:")
    errored = generator.inject_error(traces[0])
    print(f"    error={errored.injected_error} | args={list(errored.tool_call.get('arguments', {}).keys())}")


def main() -> None:
    for spec_fn, label, seed in [
        (crm_spec, "CRM", 42),
        (finance_spec, "Finance", 7),
        (devops_spec, "DevOps", 13),
    ]:
        run_demo(spec_fn, label, seed=seed)


if __name__ == "__main__":
    main()
