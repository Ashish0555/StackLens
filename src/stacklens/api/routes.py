"""HTTP endpoints.

Thin on purpose. Everything these functions do is call `agent.service` and
shape the result: the same orchestration the CLI uses, with no logic of its
own. That was the point of keeping the service layer free of terminal
concerns.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from stacklens.agent import ask, format_trace
from stacklens.api.feedback import FeedbackStore
from stacklens.api.schemas import (
    AnswerResponse,
    AskRequest,
    ChatRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ReadinessResponse,
    StatusResponse,
)
from stacklens.api.sessions import RedisSessionStore, SessionStore
from stacklens.config import Settings, get_settings
from stacklens.evaluation.metrics import extract_retrieved_sources
from stacklens.indexing import DatabaseUnavailableError, count_documents, describe_location
from stacklens.observability import flush

logger = logging.getLogger(__name__)

router = APIRouter()


def get_sessions(request: Request) -> SessionStore:
    """The session store lives on the app, not in a module global."""
    store: SessionStore = request.app.state.sessions
    return store


# Annotated rather than a default value: it keeps the dependency out of the
# function signature's defaults, which is both the current FastAPI style and
# what stops the linter flagging a call in a default argument.
def get_feedback(request: Request) -> FeedbackStore:
    """The feedback store lives on the app, like the session store."""
    store: FeedbackStore = request.app.state.feedback
    return store


SettingsDep = Annotated[Settings, Depends(get_settings)]
FeedbackDep = Annotated[FeedbackStore, Depends(get_feedback)]
SessionsDep = Annotated[SessionStore, Depends(get_sessions)]


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness. Answers 200 whenever this process is answering at all.

    It checks nothing else on purpose. A liveness probe that fails when the
    database blinks tells the orchestrator to restart a process that was never
    broken, and every replica restarting at once is how one database problem
    becomes an outage. Use /ready to decide whether to send traffic.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse, tags=["ops"])
def ready() -> ReadinessResponse:
    """Readiness. Whether this process can answer a real question.

    Two things have to be true: the database answers, and something is
    indexed. An empty index replies "não encontrei" to everything, so an
    instance in that state is not ready however healthy the process is.

    A 503 here takes the instance out of rotation and leaves it running, which
    is the right answer to a dependency that is briefly away.
    """
    try:
        indexed = count_documents()
    except DatabaseUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    if not indexed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O índice está vazio. Rode: stacklens ingest",
        )

    return ReadinessResponse(
        status="ready",
        database="ok",
        indexed_chunks=indexed,
        vector_store=describe_location(),
    )


@router.get("/status", response_model=StatusResponse, tags=["ops"])
def read_status(settings: SettingsDep) -> StatusResponse:
    """The configuration behind the answers, for diagnosing one after the fact."""
    return StatusResponse(
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        knowledge_domain=settings.knowledge_domain,
        chunk_strategy=settings.chunk_strategy.value,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retrieval_k=settings.retrieval_k,
        vector_store=describe_location(),
        indexed_chunks=_safe_count(),
        tracing_enabled=settings.tracing_enabled,
    )


@router.post("/ask", response_model=AnswerResponse, tags=["agent"])
def ask_once(payload: AskRequest) -> AnswerResponse:
    """Answer one question, with no memory of anything before it."""
    _require_index()

    result = ask(payload.question)
    flush()

    return AnswerResponse.from_result(
        result,
        sources=extract_retrieved_sources(result.messages),
        run_id=uuid4().hex,
        trace=format_trace(result.messages) if payload.trace else None,
    )


@router.post("/chat", response_model=AnswerResponse, tags=["agent"])
def chat(payload: ChatRequest, sessions: SessionsDep) -> AnswerResponse:
    """Answer inside a conversation, remembering the earlier turns.

    Omit session_id to start one; send back the id from the response to
    continue it.
    """
    _require_index()

    session_id, session = sessions.get_or_create(payload.session_id)
    result = session.send(payload.question)

    # Redis holds a copy, not the object: mutating the session here does not
    # update what the next replica will read.
    if isinstance(sessions, RedisSessionStore):
        sessions.save(session_id, session)

    flush()

    return AnswerResponse.from_result(
        result,
        sources=extract_retrieved_sources(result.messages),
        run_id=uuid4().hex,
        session_id=session_id,
        trace=format_trace(result.messages) if payload.trace else None,
    )


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["agent"])
def end_chat(session_id: str, sessions: SessionsDep) -> None:
    """Forget a conversation."""
    if not sessions.drop(session_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sessão não encontrada.")


@router.post("/feedback", response_model=FeedbackResponse, tags=["agent"])
def record_feedback(payload: FeedbackRequest, feedback: FeedbackDep) -> FeedbackResponse:
    """Record what someone thought of an answer.

    The run_id comes from the answer being judged. Nothing validates that it
    corresponds to a real run: rejecting unknown ids would mean keeping every
    answer in memory, and the cost of an occasional stray entry is lower than
    the cost of that.
    """
    feedback.record(payload.model_dump())
    return FeedbackResponse(recorded=True, run_id=payload.run_id)


def _require_index() -> None:
    """Refuse to answer against an empty index instead of answering uselessly."""
    if _safe_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O índice está vazio. Rode a ingestão antes de perguntar.",
        )


def _safe_count() -> int:
    try:
        return count_documents()
    except DatabaseUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
