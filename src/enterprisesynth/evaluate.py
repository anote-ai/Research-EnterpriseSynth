from __future__ import annotations
import math
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
        p["name"]
        for p in endpoint.parameters
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


def trace_diversity_score(traces: list["SFTTrace"]) -> float:
    """Measure argument-level diversity across traces using normalised entropy.

    For each argument key observed across all traces, compute the entropy of
    the distribution of its values.  The final score is the mean normalised
    entropy across all argument keys, where normalisation is against
    ``log2(total_traces)``.

    Returns a score in [0, 1] where 1 = maximally diverse.
    """
    total = len(traces)
    if total <= 1:
        return 0.0

    # Collect values per argument key
    key_values: dict[str, list[str]] = {}
    for trace in traces:
        for k, v in trace.tool_call.get("arguments", {}).items():
            key_values.setdefault(k, []).append(str(v))

    if not key_values:
        return 0.0

    max_entropy = math.log2(total)
    if max_entropy == 0:
        return 0.0

    entropies: list[float] = []
    for values in key_values.values():
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        n = len(values)
        entropy = -sum(
            (c / n) * math.log2(c / n) for c in counts.values() if c > 0
        )
        entropies.append(entropy / max_entropy)

    return sum(entropies) / len(entropies)


def schema_coverage_score(traces: list["SFTTrace"], schema: "APISchema") -> float:
    """Fraction of schema endpoints exercised by at least one trace.

    Returns a score in [0, 1] where 1 = every endpoint is covered.
    """
    if not schema.endpoints:
        return 1.0
    endpoint_ids = {ep.operation_id for ep in schema.endpoints}
    covered = {t.tool_call.get("name", "") for t in traces} & endpoint_ids
    return len(covered) / len(endpoint_ids)
