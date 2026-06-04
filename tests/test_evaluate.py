"""Tests for enterprisesynth.evaluate."""

from enterprisesynth.core import APISchema, Endpoint, SFTTrace
from enterprisesynth.evaluate import cold_start_score, dual_output_stats, verify_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema() -> APISchema:
    return APISchema(
        title="TestAPI",
        version="1.0",
        endpoints=[
            Endpoint(
                path="/users/{id}",
                method="GET",
                parameters=[
                    {"name": "id", "in": "path"},
                    {"name": "fields", "in": "query"},
                ],
            )
        ],
    )


def _make_trace(args: dict, verified: bool = False) -> SFTTrace:
    return SFTTrace(
        instruction="Get user by id",
        tool_call={"tool_name": "get_user", "arguments": args},
        response={},
        verified=verified,
    )


# ---------------------------------------------------------------------------
# verify_trace
# ---------------------------------------------------------------------------

def test_verify_trace_valid() -> None:
    schema = _make_schema()
    trace = _make_trace({"id": "42"})
    assert verify_trace(trace, schema) is True


def test_verify_trace_valid_multiple_params() -> None:
    schema = _make_schema()
    trace = _make_trace({"id": "42", "fields": "name"})
    assert verify_trace(trace, schema) is True


def test_verify_trace_invalid_extra_key() -> None:
    schema = _make_schema()
    trace = _make_trace({"id": "42", "unknown_param": "x"})
    assert verify_trace(trace, schema) is False


def test_verify_trace_empty_args_valid() -> None:
    """Empty arguments are a subset of any endpoint's params."""
    schema = _make_schema()
    trace = _make_trace({})
    assert verify_trace(trace, schema) is True


def test_verify_trace_no_endpoints() -> None:
    schema = APISchema(title="Empty", version="1.0")
    trace = _make_trace({"id": "1"})
    assert verify_trace(trace, schema) is False


# ---------------------------------------------------------------------------
# dual_output_stats
# ---------------------------------------------------------------------------

def test_dual_output_stats_mixed() -> None:
    traces = [
        _make_trace({}, verified=True),
        _make_trace({}, verified=True),
        _make_trace({}, verified=False),
    ]
    stats = dual_output_stats(traces)
    assert stats["sft_count"] == 3
    assert stats["eval_count"] == 1
    assert abs(stats["verified_rate"] - 2 / 3) < 1e-9


def test_dual_output_stats_empty() -> None:
    stats = dual_output_stats([])
    assert stats["sft_count"] == 0
    assert stats["verified_rate"] == 0.0


def test_dual_output_stats_all_verified() -> None:
    traces = [_make_trace({}, verified=True) for _ in range(5)]
    stats = dual_output_stats(traces)
    assert stats["verified_rate"] == 1.0
    assert stats["eval_count"] == 0


# ---------------------------------------------------------------------------
# cold_start_score
# ---------------------------------------------------------------------------

def test_cold_start_score_empty() -> None:
    assert cold_start_score([]) == 0.0


def test_cold_start_score_all_verified() -> None:
    traces = [_make_trace({}, verified=True) for _ in range(4)]
    assert cold_start_score(traces) == 1.0


def test_cold_start_score_partial() -> None:
    traces = [
        _make_trace({}, verified=True),
        _make_trace({}, verified=False),
    ]
    assert cold_start_score(traces) == 0.5
