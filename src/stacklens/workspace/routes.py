"""User-facing API. Ownership is derived from the session, never from request IDs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field

from stacklens.api.schemas import AnswerResponse
from stacklens.api.sessions import RedisSessionStore
from stacklens.config import get_settings
from stacklens.evaluation.metrics import extract_retrieved_sources
from stacklens.indexing.scope import collection_scope, user_collection
from stacklens.observability import flush
from stacklens.workspace import store
from stacklens.workspace.ingestion import process_job

router = APIRouter(prefix="/api/v1", tags=["workspace"])
COOKIE = "stacklens_session"
_CHAT_LOCKS = [Lock() for _ in range(256)]


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=10, max_length=128)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def csrf(request: Request):
    if request.headers.get("X-StackLens-Request") != "1":
        raise HTTPException(403, "Missing request protection header.")


def user(request: Request) -> dict:
    identity = store.get_user(request.cookies.get(COOKIE))
    if not identity:
        raise HTTPException(401, "Please sign in.")
    return identity


def require_admin_key(request: Request):
    import hmac

    expected = get_settings().api_key.get_secret_value()
    if not expected or not hmac.compare_digest(request.headers.get("X-API-Key", ""), expected):
        raise HTTPException(401, "A configured administrator X-API-Key is required.")


def sign_in(response: Response, identity: dict):
    response.set_cookie(
        COOKIE,
        store.issue_session(identity["id"]),
        httponly=True,
        secure=get_settings().cookie_secure,
        samesite="strict",
        max_age=86400 * 7,
        path="/",
    )
    return identity


@router.post("/register", dependencies=[Depends(csrf)], status_code=201)
def register(payload: Credentials, response: Response):
    if not get_settings().registration_enabled:
        raise HTTPException(403, "Registration is disabled.")
    try:
        identity = store.create_user(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return sign_in(response, identity)


@router.post("/login", dependencies=[Depends(csrf)])
def login(payload: Credentials, response: Response):
    identity = store.authenticate(payload.username, payload.password)
    if not identity:
        raise HTTPException(401, "Invalid username or password.")
    return sign_in(response, identity)


@router.post("/logout", dependencies=[Depends(csrf)])
def logout(request: Request, response: Response):
    store.revoke_session(request.cookies.get(COOKIE, ""))
    response.delete_cookie(COOKIE, path="/")
    return {"signed_out": True}


@router.get("/me")
def me(identity: Annotated[dict, Depends(user)]):
    return identity


def public_job(job: dict) -> dict:
    return {
        k: job[k] for k in ("id", "filename", "status", "chunks", "error", "created", "attempts")
    }


@router.get("/documents")
def documents(identity: Annotated[dict, Depends(user)]):
    return {"documents": [public_job(j) for j in store.list_jobs(identity["id"])]}


@router.post("/documents", dependencies=[Depends(csrf)], status_code=202)
async def upload(
    request: Request,
    tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    identity: Annotated[dict, Depends(user)],
):
    s = get_settings()
    size = request.headers.get("content-length")
    if not size:
        raise HTTPException(411, "Upload requires Content-Length.")
    if not size.isdigit() or int(size) > s.max_upload_bytes + 131072:
        raise HTTPException(413, "Upload is too large.")
    name = Path(file.filename or "").name
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100]
    if Path(name).suffix.lower() not in {".md", ".txt", ".pdf", ".rst"}:
        raise HTTPException(415, "Upload a PDF, Markdown, text or reStructuredText file.")
    contents = await file.read(s.max_upload_bytes + 1)
    await file.close()
    if not contents or len(contents) > s.max_upload_bytes:
        raise HTTPException(413, "Upload must contain between 1 byte and 5 MB.")
    if Path(name).suffix.lower() == ".pdf" and not contents.startswith(b"%PDF-"):
        raise HTTPException(422, "Invalid PDF file.")
    job_id = uuid4().hex
    folder = s.upload_dir / identity["id"] / job_id
    folder.mkdir(parents=True, mode=0o700)
    path = folder / name
    path.write_bytes(contents)
    job, created = store.new_job(
        identity["id"], name, path, hashlib.sha256(contents).hexdigest(), job_id
    )
    if not created:
        path.unlink()
        folder.rmdir()
    elif s.ingestion_mode == "local":
        tasks.add_task(process_job, job["id"])
    return public_job(job)


@router.get("/documents/{job_id}")
def document(job_id: str, identity: Annotated[dict, Depends(user)]):
    job = store.get_job(job_id, identity["id"])
    if not job:
        raise HTTPException(404, "Document not found.")
    return public_job(job)


@router.post("/documents/{job_id}/retry", dependencies=[Depends(csrf)], status_code=202)
def retry(job_id: str, tasks: BackgroundTasks, identity: Annotated[dict, Depends(user)]):
    if not store.retry_job(job_id, identity["id"]):
        raise HTTPException(404, "Failed document not found.")
    if get_settings().ingestion_mode == "local":
        tasks.add_task(process_job, job_id)
    job = store.get_job(job_id, identity["id"])
    assert job is not None
    return public_job(job)


@router.post("/chat", dependencies=[Depends(csrf)], response_model=AnswerResponse)
def chat(payload: Question, request: Request, identity: Annotated[dict, Depends(user)]):
    if not payload.question.strip():
        raise HTTPException(422, "Enter a question.")
    if not any(j["status"] == "completed" for j in store.list_jobs(identity["id"])):
        raise HTTPException(409, "Upload and finish processing a document first.")
    public_id = payload.session_id or uuid4().hex
    internal_id = identity["id"] + ":" + public_id
    sessions = request.app.state.sessions
    try:
        with (
            _CHAT_LOCKS[hash(internal_id) % len(_CHAT_LOCKS)],
            collection_scope(user_collection(identity["id"])),
        ):
            _, session = sessions.get_or_create(internal_id)
            result = session.send(payload.question)
            if isinstance(sessions, RedisSessionStore):
                sessions.save(internal_id, session)
    except Exception as exc:
        from stacklens.guardrails import GuardrailViolation

        if isinstance(exc, GuardrailViolation):
            raise
        raise HTTPException(
            503,
            "The assistant is temporarily unavailable. Check model access and the document index.",
        ) from exc
    flush()
    return AnswerResponse.from_result(
        result,
        sources=extract_retrieved_sources(result.messages),
        run_id=uuid4().hex,
        session_id=public_id,
    )


@router.get("/health")
def health():
    return {"status": "ok"}
