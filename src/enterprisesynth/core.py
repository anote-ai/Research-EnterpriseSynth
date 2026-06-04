"""Core data models and generation pipeline for EnterpriseSynth."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Endpoint(BaseModel):
    """A single API endpoint parsed from an OpenAPI spec."""

    path: str = Field(..., description="URL path, e.g. /users/{id}")
    method: str = Field(..., description="HTTP method (GET, POST, …)")
    parameters: list[dict] = Field(
        default_factory=list,
        description="List of parameter objects as parsed from the spec.",
    )
    response_schema: dict = Field(
        default_factory=dict,
        description="JSON Schema of the success response body.",
    )


class APISchema(BaseModel):
    """Top-level parsed representation of an OpenAPI/Swagger specification."""

    title: str = Field(..., description="API title from the spec info block.")
    version: str = Field(..., description="API version string.")
    endpoints: list[Endpoint] = Field(default_factory=list)


class SFTTrace(BaseModel):
    """A single supervised fine-tuning trace with optional intent spec."""

    instruction: str = Field(..., description="Natural-language instruction for the agent.")
    tool_call: dict = Field(..., description="Tool call dict with tool_name and arguments.")
    response: dict = Field(..., description="Simulated tool response.")
    intent_spec: str = Field(
        default="", description="Serialized intent specification for this trace."
    )
    verified: bool = Field(
        default=False,
        description="True if the trace passes schema validation.",
    )


class SchemaParser:
    """Parses OpenAPI/Swagger specification dicts into :class:`APISchema` objects."""

    def parse_openapi(self, spec_dict: dict) -> APISchema:
        """Parse a raw OpenAPI spec dict into an :class:`APISchema`.

        Args:
            spec_dict: A dict loaded from a YAML/JSON OpenAPI file.

        Returns:
            Populated :class:`APISchema` instance.

        Raises:
            NotImplementedError: Until implementation is complete.
        """
        # TODO: traverse spec_dict['paths'] and build Endpoint instances
        raise NotImplementedError("parse_openapi is not yet implemented")


class TraceGenerator:
    """Generates SFT traces from an :class:`APISchema` without live API execution."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def generate_traces(self, schema: APISchema, n: int = 10) -> list[SFTTrace]:
        """Generate *n* SFT traces for the given API schema (stub).

        Args:
            schema: Parsed API schema to generate traces for.
            n: Number of traces to generate per endpoint (approximately).

        Returns:
            List of :class:`SFTTrace` instances (unverified).

        Raises:
            NotImplementedError: Until implementation is complete.
        """
        # TODO: use LLM to synthesise instruction + tool_call + response triples
        raise NotImplementedError("generate_traces is not yet implemented")
