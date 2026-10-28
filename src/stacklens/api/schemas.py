"""Request and response bodies for the HTTP layer.

These are the API's contract. They are deliberately separate from the domain
types in `stacklens.types`: changing an internal dataclass should not silently
change what clients receive.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from stacklens.types import AnswerResult


class AskRequest(BaseModel):
    """A single question, with no memory of anything before it."""

    question: str = Field(min_length=1, max_length=2000, examples=["qual o lote suplementar?"])
    trace: bool = Field(default=False, description="Inclui o rastro de raciocínio na resposta.")


class ChatRequest(AskRequest):
    """A question inside a conversation."""

    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Omita para abrir uma conversa nova; reenvie para continuar a mesma.",
    )


class ToolCallResponse(BaseModel):
    name: str
    arguments: dict


class MetricsResponse(BaseModel):
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: int
    model: str
    estimated_cost_usd: float | None


class AnswerResponse(BaseModel):
    """What every question endpoint returns."""

    answer: str
    sources: list[str]
    tools_used: list[ToolCallResponse]
    metrics: MetricsResponse | None
    run_id: str
    """Identifies this answer, so feedback can point back at it."""
    session_id: str | None = None
    trace: str | None = None

    @classmethod
    def from_result(
        cls,
        result: AnswerResult,
        *,
        sources: list[str],
        run_id: str,
        session_id: str | None = None,
        trace: str | None = None,
    ) -> AnswerResponse:
        return cls(
            answer=result.answer,
            sources=sources,
            tools_used=[
                ToolCallResponse(name=call.name, arguments=call.arguments)
                for call in result.tool_calls
            ],
            metrics=_metrics(result),
            run_id=run_id,
            session_id=session_id,
            trace=trace,
        )


class FeedbackRequest(BaseModel):
    """A verdict on one answer, tied to the run that produced it."""

    run_id: str = Field(min_length=1, max_length=64)
    trace_id: str | None = Field(
        default=None, max_length=64, description="Id do trace no Langfuse, quando houver."
    )
    useful: bool
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    recorded: bool
    run_id: str


class HealthResponse(BaseModel):
    """Liveness: whether this process is answering at all.

    Deliberately free of dependency checks. A liveness probe that fails when
    the database blinks tells the orchestrator to restart a process that was
    never broken, and a restart loop is how one database problem becomes an
    outage.
    """

    status: str


class ReadinessResponse(BaseModel):
    """Readiness: whether this process can serve a real question.

    A failure here takes the instance out of rotation and leaves it running,
    which is the correct answer to a dependency that is briefly away.
    """

    status: str
    database: str
    indexed_chunks: int
    vector_store: str


class StatusResponse(BaseModel):
    """The active configuration, for diagnosing an answer after the fact."""

    chat_model: str
    embedding_model: str
    knowledge_domain: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    vector_store: str
    indexed_chunks: int
    tracing_enabled: bool


def _metrics(result: AnswerResult) -> MetricsResponse | None:
    if result.metrics is None:
        return None

    return MetricsResponse(
        latency_seconds=round(result.metrics.latency_seconds, 3),
        input_tokens=result.metrics.input_tokens,
        output_tokens=result.metrics.output_tokens,
        total_tokens=result.metrics.total_tokens,
        tool_calls=result.metrics.tool_calls,
        model=result.metrics.model,
        estimated_cost_usd=result.metrics.estimated_cost_usd,
    )
