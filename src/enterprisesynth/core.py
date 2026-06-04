from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any
import random
import uuid
import itertools


class Endpoint(BaseModel):
    path: str
    method: str
    operation_id: str
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    response_schema: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class APISchema(BaseModel):
    title: str
    version: str
    endpoints: list[Endpoint] = Field(default_factory=list)


class SFTTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    instruction: str
    tool_call: dict[str, Any]
    response: dict[str, Any]
    intent_spec: str
    verified: bool = False
    # For multi-step traces: ordered list of (tool_call, response) pairs
    steps: list[dict[str, Any]] = Field(default_factory=list)
    # Error injection metadata
    injected_error: str | None = None


class SchemaParser:
    def parse_openapi(self, spec: dict) -> APISchema:
        info = spec.get("info", {})
        endpoints = []
        for path, methods in spec.get("paths", {}).items():
            for method, op in methods.items():
                params = op.get("parameters", [])
                endpoints.append(
                    Endpoint(
                        path=path,
                        method=method.upper(),
                        operation_id=op.get("operationId", f"{method}_{path}"),
                        parameters=params,
                        description=op.get("summary", ""),
                        response_schema=(
                            op.get("responses", {})
                            .get("200", {})
                            .get("schema", {})
                        ),
                    )
                )
        return APISchema(
            title=info.get("title", ""),
            version=info.get("version", "1.0"),
            endpoints=endpoints,
        )


ERROR_TYPES: list[dict[str, Any]] = [
    {"type": "missing_required_param", "description": "A required parameter is omitted"},
    {"type": "wrong_type", "description": "A parameter value has the wrong type"},
    {"type": "out_of_range", "description": "A numeric value is outside valid bounds"},
    {"type": "invalid_enum", "description": "An enum field has an invalid value"},
    {"type": "extra_param", "description": "An unexpected extra parameter is included"},
]


class TraceGenerator:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _sample_value(self, param: dict) -> Any:
        t = param.get("type", "string")
        name = param.get("name", "value")
        if t == "integer":
            return self._rng.randint(1, 100)
        if t == "boolean":
            return self._rng.choice([True, False])
        if t == "array":
            return [f"item_{i}" for i in range(self._rng.randint(1, 3))]
        return f"sample_{name}_{self._rng.randint(1, 999)}"

    def generate_traces(self, schema: APISchema, n: int = 5) -> list[SFTTrace]:
        traces = []
        for ep in itertools.islice(itertools.cycle(schema.endpoints), n):
            args = {p["name"]: self._sample_value(p) for p in ep.parameters if "name" in p}
            traces.append(
                SFTTrace(
                    instruction=f"Call {ep.operation_id}: {ep.description or ep.path}",
                    tool_call={"name": ep.operation_id, "arguments": args},
                    response={"status": "ok", "data": {k: f"result_{k}" for k in args}},
                    intent_spec=(
                        f"The user wants to {ep.description or ep.operation_id} on {ep.path}"
                    ),
                )
            )
        return traces

    def generate_multi_step_trace(
        self, schema: APISchema, n_steps: int = 3
    ) -> SFTTrace:
        """Generate a single SFTTrace that chains *n_steps* endpoint calls.

        The returned trace's ``tool_call`` and ``response`` reflect the *last*
        step; all steps are stored in ``steps`` for full trajectory inspection.
        """
        eps = [
            ep
            for ep in itertools.islice(itertools.cycle(schema.endpoints), n_steps)
        ]
        steps: list[dict[str, Any]] = []
        for ep in eps:
            args = {p["name"]: self._sample_value(p) for p in ep.parameters if "name" in p}
            step = {
                "tool_call": {"name": ep.operation_id, "arguments": args},
                "response": {"status": "ok", "data": {k: f"result_{k}" for k in args}},
                "intent": f"{ep.description or ep.operation_id} on {ep.path}",
            }
            steps.append(step)

        last = steps[-1]
        instruction_parts = [s["intent"] for s in steps]
        return SFTTrace(
            instruction=" -> ".join(instruction_parts),
            tool_call=last["tool_call"],
            response=last["response"],
            intent_spec=f"Multi-step workflow: {', '.join(instruction_parts)}",
            steps=steps,
        )

    def inject_error(
        self,
        trace: SFTTrace,
        error_type: str | None = None,
    ) -> SFTTrace:
        """Return a *new* SFTTrace with a synthetic error injected for robustness testing.

        The original trace is not modified.  ``injected_error`` records which
        error type was applied.  ``verified`` is set to False.
        """
        chosen = error_type or self._rng.choice(ERROR_TYPES)["type"]
        tool_call = dict(trace.tool_call)
        args: dict[str, Any] = dict(tool_call.get("arguments", {}))

        if chosen == "missing_required_param" and args:
            key = self._rng.choice(list(args.keys()))
            args.pop(key)
        elif chosen == "wrong_type" and args:
            key = self._rng.choice(list(args.keys()))
            args[key] = ["unexpected", "list"]  # force wrong type
        elif chosen == "out_of_range":
            args["__injected_int"] = -99999
        elif chosen == "invalid_enum":
            args["__injected_enum"] = "INVALID_ENUM_VALUE"
        elif chosen == "extra_param":
            args["__extra_unexpected_param"] = "injected"

        tool_call["arguments"] = args
        return SFTTrace(
            trace_id=trace.trace_id,
            instruction=trace.instruction,
            tool_call=tool_call,
            response={"status": "error", "error": chosen},
            intent_spec=trace.intent_spec,
            verified=False,
            steps=list(trace.steps),
            injected_error=chosen,
        )
