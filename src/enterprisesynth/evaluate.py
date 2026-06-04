"""Evaluation and verification utilities for EnterpriseSynth traces."""

from __future__ import annotations

from .core import APISchema, SFTTrace


def verify_trace(trace: SFTTrace, schema: APISchema) -> bool:
    """Verify that a trace's tool call keys match any endpoint's parameter names.

    A trace is considered valid if its ``tool_call['arguments']`` keys are a
    subset of the parameter names defined for at least one endpoint in the
    schema.

    Args:
        trace: The SFT trace to verify.
        schema: The API schema providing ground-truth parameter names.

    Returns:
        True if the trace is consistent with the schema, False otherwise.
    """
    call_keys = set(trace.tool_call.get("arguments", {}).keys())
    for endpoint in schema.endpoints:
        endpoint_param_names = {p.get("name", "") for p in endpoint.parameters}
        if call_keys <= endpoint_param_names:
            return True
    return False


def dual_output_stats(traces: list[SFTTrace]) -> dict[str, object]:
    """Compute statistics for a mixed SFT + eval trace set.

    Args:
        traces: List of :class:`SFTTrace` instances, some verified and some not.

    Returns:
        Dict with keys:

        - ``sft_count`` — total number of traces.
        - ``eval_count`` — number of unverified traces (usable as eval data).
        - ``verified_rate`` — fraction of traces that are verified.
    """
    total = len(traces)
    verified = sum(1 for t in traces if t.verified)
    unverified = total - verified
    return {
        "sft_count": total,
        "eval_count": unverified,
        "verified_rate": verified / total if total > 0 else 0.0,
    }


def cold_start_score(traces: list[SFTTrace]) -> float:
    """Compute the cold-start quality score as the ratio of verified traces.

    A higher score indicates that more synthetically generated traces pass
    schema validation, making them safe to use for SFT without live API calls.

    Args:
        traces: List of :class:`SFTTrace` instances.

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.verified) / len(traces)
