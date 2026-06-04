import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser, TraceGenerator, SFTTrace
from enterprisesynth.data import crm_spec
from enterprisesynth.evaluate import verify_trace, dual_output_stats, cold_start_score


def _setup():
    schema = SchemaParser().parse_openapi(crm_spec())
    gen = TraceGenerator(seed=42)
    traces = gen.generate_traces(schema, n=6)
    return schema, traces


def test_verify_trace_valid():
    schema, traces = _setup()
    # First trace should have required params satisfied by generator
    result = verify_trace(traces[0], schema)
    assert isinstance(result, bool)
    # Generator fills all params so should be True for endpoints with required params
    assert result is True


def test_verify_trace_missing_param():
    schema, _ = _setup()
    # Create a trace with missing required args
    bad_trace = SFTTrace(
        instruction="test",
        tool_call={"name": "createContact", "arguments": {}},  # missing name, email
        response={},
        intent_spec="test",
    )
    assert verify_trace(bad_trace, schema) is False


def test_dual_output_stats_verified_rate_in_range():
    _, traces = _setup()
    # Mark some as verified
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
        SFTTrace(instruction="t", tool_call={"name": "a"}, response={}, intent_spec="t", verified=True)
        for _ in range(4)
    ]
    stats = dual_output_stats(traces)
    assert stats["verified_rate"] == 1.0
