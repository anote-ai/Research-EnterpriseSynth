"""Tests for enterprisesynth.core."""

import pytest

from enterprisesynth.core import APISchema, Endpoint, SFTTrace, SchemaParser, TraceGenerator


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def test_endpoint_creation() -> None:
    ep = Endpoint(path="/users", method="GET")
    assert ep.path == "/users"
    assert ep.method == "GET"
    assert ep.parameters == []


def test_endpoint_with_parameters() -> None:
    ep = Endpoint(
        path="/users/{id}",
        method="GET",
        parameters=[{"name": "id", "in": "path", "required": True}],
    )
    assert len(ep.parameters) == 1
    assert ep.parameters[0]["name"] == "id"


# ---------------------------------------------------------------------------
# APISchema
# ---------------------------------------------------------------------------

def test_api_schema_creation() -> None:
    schema = APISchema(title="Petstore", version="1.0.0")
    assert schema.title == "Petstore"
    assert schema.endpoints == []


def test_api_schema_with_endpoints() -> None:
    endpoints = [
        Endpoint(path="/pets", method="GET"),
        Endpoint(path="/pets", method="POST"),
    ]
    schema = APISchema(title="Petstore", version="1.0.0", endpoints=endpoints)
    assert len(schema.endpoints) == 2


# ---------------------------------------------------------------------------
# SFTTrace
# ---------------------------------------------------------------------------

def test_sft_trace_creation() -> None:
    trace = SFTTrace(
        instruction="List all pets",
        tool_call={"tool_name": "list_pets", "arguments": {}},
        response={"pets": []},
    )
    assert trace.instruction == "List all pets"
    assert trace.verified is False


def test_sft_trace_verified_flag() -> None:
    trace = SFTTrace(
        instruction="Get user",
        tool_call={"tool_name": "get_user", "arguments": {"id": "123"}},
        response={"id": "123", "name": "Alice"},
        verified=True,
    )
    assert trace.verified is True


# ---------------------------------------------------------------------------
# SchemaParser
# ---------------------------------------------------------------------------

def test_schema_parser_instantiation() -> None:
    parser = SchemaParser()
    assert parser is not None


def test_schema_parser_parse_openapi_raises() -> None:
    parser = SchemaParser()
    with pytest.raises(NotImplementedError):
        parser.parse_openapi({})


# ---------------------------------------------------------------------------
# TraceGenerator
# ---------------------------------------------------------------------------

def test_trace_generator_instantiation_default() -> None:
    gen = TraceGenerator()
    assert gen.model == "gpt-4o-mini"


def test_trace_generator_instantiation_custom_model() -> None:
    gen = TraceGenerator(model="claude-3-haiku")
    assert gen.model == "claude-3-haiku"


def test_trace_generator_generate_traces_raises() -> None:
    gen = TraceGenerator()
    schema = APISchema(title="Test", version="0.1")
    with pytest.raises(NotImplementedError):
        gen.generate_traces(schema, n=5)
