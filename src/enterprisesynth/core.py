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
