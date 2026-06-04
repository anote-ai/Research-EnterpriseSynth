import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enterprisesynth.core import SchemaParser, TraceGenerator, SFTTrace, APISchema, Endpoint
from enterprisesynth.data import crm_spec


def _schema():
    return SchemaParser().parse_openapi(crm_spec())


def test_schema_parser_parse_openapi():
    schema = _schema()
    assert isinstance(schema, APISchema)
    assert schema.title == "CRM API"


def test_api_schema_endpoints_length():
    schema = _schema()
    assert len(schema.endpoints) == 3


def test_endpoint_fields():
    schema = _schema()
    ep = schema.endpoints[0]
    assert isinstance(ep, Endpoint)
    assert ep.path
    assert ep.method
    assert ep.operation_id


def test_trace_generator_generate_traces_count():
    schema = _schema()
    gen = TraceGenerator()
    traces = gen.generate_traces(schema, n=5)
    assert len(traces) == 5


def test_trace_has_required_fields():
    schema = _schema()
    gen = TraceGenerator()
    trace = gen.generate_traces(schema, n=1)[0]
    assert trace.instruction
    assert trace.tool_call
    assert trace.response
    assert trace.intent_spec


def test_sft_trace_trace_id_auto_generated():
    trace = SFTTrace(
        instruction="test", tool_call={}, response={}, intent_spec="test"
    )
    assert trace.trace_id
    assert len(trace.trace_id) == 8


def test_tool_call_has_name():
    schema = _schema()
    gen = TraceGenerator()
    trace = gen.generate_traces(schema, n=1)[0]
    assert "name" in trace.tool_call


def test_generate_traces_cycles_when_n_exceeds_endpoints():
    schema = _schema()
    gen = TraceGenerator()
    traces = gen.generate_traces(schema, n=9)
    assert len(traces) == 9
    names = [t.tool_call["name"] for t in traces]
    assert names[:3] == names[3:6]


def test_trace_generator_seed_reproducibility():
    schema = _schema()
    gen1 = TraceGenerator(seed=0)
    gen2 = TraceGenerator(seed=0)
    t1 = gen1.generate_traces(schema, n=5)
    t2 = gen2.generate_traces(schema, n=5)
    assert [t.tool_call for t in t1] == [t.tool_call for t in t2]


def test_sft_trace_verified_false_by_default():
    trace = SFTTrace(
        instruction="test", tool_call={}, response={}, intent_spec="test"
    )
    assert trace.verified is False
