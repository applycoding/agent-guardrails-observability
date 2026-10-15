"""OpenTelemetry tracing with GenAI style attributes.

Spans can be exported to any OTLP backend. Langfuse accepts OTLP over HTTP at
`<host>/api/public/otel`, so pointing OTEL_EXPORTER_OTLP_ENDPOINT there (with the basic auth
header) sends the same traces to a self hosted Langfuse. Nothing is exported by default.
"""
from __future__ import annotations

import os

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Tracer


def make_tracer(exporter: SpanExporter | None = None, service_name: str = "agent-guardrails-observability") -> tuple[Tracer, TracerProvider]:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):  # pragma: no cover - needs a collector
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider.get_tracer("agentops"), provider
