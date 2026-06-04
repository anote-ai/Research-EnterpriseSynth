import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser, TraceGenerator, SFTTrace
from enterprisesynth.data import crm_spec, healthcare_spec, legal_spec
from enterprisesynth.evaluate import (
    verify_trace,
    dual_output_stats,
    cold_start_score,
    trace_diversity_score,
    schema_coverage_score,
)


def _setup():
    schema = SchemaParser().parse_openapi(crm_spec())
    gen = TraceGenerator(seed=42)
    traces = gen.generate_traces(schema, n=6)
    return schema, traces


def test_verify_trace_valid():
    schema, traces = _setup()
    result = verify_trace(traces[0], schema)
    assert isinstance(result, bool)
    assert result is True


def test_verify_trace_missing_param():
    schema, _ = _setup()
    bad_trace = SFTTrace(
        instruction="test",
        tool_call={"name": "createContact", "arguments": {}},
        response={},
        intent_spec="test",
    )
    assert verify_trace(bad_trace, schema) is False


def test_dual_output_stats_verified_rate_in_range():
    _, traces = _setup()
    for t in traces[:3]:
        t.verified = True
    stats = dual_output_stats(traces)
    assert 0.0 <= stats["verified_rate"] <= 1.0


def test_cold_start_score_in_range():
    schema, traces = _setup()
    for t in traces:
        t.verified = verify_trace(t, schema)
    score = cold_start_score(traces)
    assert 0.0 <= score <= 1.0


def test_dual_output_stats_total_count():
    _, traces = _setup()
    stats = dual_output_stats(traces)
    assert stats["total"] == 6


def test_verified_rate_all_verified():
    traces = [
        SFTTrace(
            instruction="t",
            tool_call={"name": "a"},
            response={},
            intent_spec="t",
            verified=True,
        )
        for _ in range(4)
    ]
    stats = dual_output_stats(traces)
    assert stats["verified_rate"] == 1.0


# --- trace_diversity_score ---

def test_trace_diversity_score_zero_for_single():
    _, traces = _setup()
    score = trace_diversity_score(traces[:1])
    assert score == 0.0


def test_trace_diversity_score_range():
    _, traces = _setup()
    score = trace_diversity_score(traces)
    assert 0.0 <= score <= 1.0


def test_trace_diversity_score_higher_with_varied_args():
    """Traces with distinct argument values should score higher than identical ones."""
    schema = SchemaParser().parse_openapi(crm_spec())
    gen_varied = TraceGenerator(seed=1)
    gen_same = TraceGenerator(seed=99)
    varied = gen_varied.generate_traces(schema, n=10)
    same = gen_same.generate_traces(schema, n=10)
    # Overwrite all args with identical values to simulate zero diversity
    for t in same:
        args = t.tool_call.get("arguments", {})
        for k in args:
            args[k] = "fixed_value"
    score_varied = trace_diversity_score(varied)
    score_same = trace_diversity_score(same)
    assert score_varied >= score_same


# --- schema_coverage_score ---

def test_schema_coverage_full():
    schema = SchemaParser().parse_openapi(crm_spec())
    gen = TraceGenerator(seed=42)
    # Generate enough traces to hit every endpoint
    traces = gen.generate_traces(schema, n=len(schema.endpoints))
    score = schema_coverage_score(traces, schema)
    assert score == 1.0


def test_schema_coverage_partial():
    schema = SchemaParser().parse_openapi(crm_spec())
    # Only one endpoint covered
    traces = [
        SFTTrace(
            instruction="t",
            tool_call={"name": "createContact", "arguments": {"name": "Alice", "email": "a@b.com"}},
            response={},
            intent_spec="t",
        )
    ]
    score = schema_coverage_score(traces, schema)
    assert 0.0 < score < 1.0


def test_schema_coverage_zero_for_unknown_tools():
    schema = SchemaParser().parse_openapi(crm_spec())
    traces = [
        SFTTrace(
            instruction="t",
            tool_call={"name": "unknownTool", "arguments": {}},
            response={},
            intent_spec="t",
        )
    ]
    assert schema_coverage_score(traces, schema) == 0.0


# --- multi-step trace generation ---

def test_generate_multi_step_trace_has_steps():
    schema = SchemaParser().parse_openapi(crm_spec())
    gen = TraceGenerator(seed=42)
    trace = gen.generate_multi_step_trace(schema, n_steps=3)
    assert len(trace.steps) == 3


def test_generate_multi_step_trace_instruction_combined():
    schema = SchemaParser().parse_openapi(crm_spec())
    gen = TraceGenerator(seed=42)
    trace = gen.generate_multi_step_trace(schema, n_steps=2)
    assert "->" in trace.instruction


# --- error injection ---

def test_inject_error_sets_injected_error_field():
    schema, traces = _setup()
    gen = TraceGenerator(seed=42)
    errored = gen.inject_error(traces[0])
    assert errored.injected_error is not None
    assert errored.verified is False


def test_inject_error_does_not_mutate_original():
    schema, traces = _setup()
    original_args = dict(traces[0].tool_call.get("arguments", {}))
    gen = TraceGenerator(seed=42)
    gen.inject_error(traces[0], error_type="extra_param")
    assert traces[0].tool_call.get("arguments") == original_args


def test_inject_error_extra_param():
    schema, traces = _setup()
    gen = TraceGenerator(seed=42)
    errored = gen.inject_error(traces[0], error_type="extra_param")
    assert "__extra_unexpected_param" in errored.tool_call.get("arguments", {})


# --- healthcare and legal specs ---

def test_healthcare_spec_parses():
    schema = SchemaParser().parse_openapi(healthcare_spec())
    assert schema.title == "Healthcare EHR API"
    assert len(schema.endpoints) == 4


def test_legal_spec_parses():
    schema = SchemaParser().parse_openapi(legal_spec())
    assert schema.title == "Legal DMS API"
    assert len(schema.endpoints) == 4


def test_healthcare_traces_generated():
    schema = SchemaParser().parse_openapi(healthcare_spec())
    gen = TraceGenerator(seed=7)
    traces = gen.generate_traces(schema, n=8)
    assert len(traces) == 8
    assert schema_coverage_score(traces, schema) == 1.0


def test_legal_traces_generated():
    schema = SchemaParser().parse_openapi(legal_spec())
    gen = TraceGenerator(seed=11)
    traces = gen.generate_traces(schema, n=8)
    assert len(traces) == 8
    assert schema_coverage_score(traces, schema) == 1.0
