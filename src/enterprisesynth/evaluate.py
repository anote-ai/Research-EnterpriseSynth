from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SFTTrace, APISchema


def verify_trace(trace: "SFTTrace", schema: "APISchema") -> bool:
    """Find matching endpoint by operation_id; check all required params are present."""
    tool_name = trace.tool_call.get("name", "")
    endpoint = next(
        (ep for ep in schema.endpoints if ep.operation_id == tool_name), None
    )
    if endpoint is None:
        return False
    provided_keys = set(trace.tool_call.get("arguments", {}).keys())
    required_params = {
        p["name"] for p in endpoint.parameters
        if "name" in p and p.get("required", True)
    }
    return required_params.issubset(provided_keys)


def dual_output_stats(traces: list["SFTTrace"]) -> dict:
    """Summary stats over a list of SFTTrace objects."""
    total = len(traces)
    verified_count = sum(1 for t in traces if t.verified)
    unique_endpoints = len({t.tool_call.get("name", "") for t in traces})
    return {
        "verified_count": verified_count,
        "total": total,
        "verified_rate": verified_count / total if total > 0 else 0.0,
        "unique_endpoints": unique_endpoints,
    }


def cold_start_score(traces: list["SFTTrace"]) -> float:
    """Score reflecting both verification rate and endpoint coverage diversity."""
    total = len(traces)
    if total == 0:
        return 0.0
    verified_count = sum(1 for t in traces if t.verified)
    verified_rate = verified_count / total
    unique_endpoints = len({t.tool_call.get("name", "") for t in traces})
    coverage = unique_endpoints / max(1, total)
    return verified_rate * coverage
