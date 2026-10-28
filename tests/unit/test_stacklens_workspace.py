"""Workspace authorization, durable jobs, and sensitive-data boundaries."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stacklens.config import get_settings
from stacklens.workspace import routes as workspace
from stacklens.workspace import store
from stacklens.workspace.ingestion import process_job
from stacklens.workspace.worker import handle_event, publish_pending

HEADERS = {"X-StackLens-Request": "1"}


@pytest.fixture
def workspace_app(monkeypatch, tmp_path):
    import importlib

    app_module = importlib.import_module("stacklens.api.app")
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "application.sqlite3"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("RATE_LIMIT", "")
    monkeypatch.setenv("INGESTION_MODE", "kafka")
    get_settings.cache_clear()
    monkeypatch.setattr(app_module, "count_documents", lambda: 0)
    return app_module.create_app()


def account(client, name):
    result = client.post(
        "/api/v1/register",
        headers=HEADERS,
        json={"username": name, "password": "long-test-password"},
    )
    assert result.status_code == 201
    return result.json()


def test_accounts_require_authentication_and_csrf_header(workspace_app):
    with TestClient(workspace_app) as c:
        assert c.get("/api/v1/documents").status_code == 401
        assert (
            c.post(
                "/api/v1/register", json={"username": "alice", "password": "long-test-password"}
            ).status_code
            == 403
        )
        assert c.get("/status").status_code == 401
        assert c.get("/api/v1/health").status_code == 200


def test_passwords_and_sessions_are_hashed_and_logout_revokes(workspace_app):
    with TestClient(workspace_app) as c:
        account(c, "alice")
        token = c.cookies.get(workspace.COOKIE)
        with store.connection() as db:
            row = dict(db.execute("SELECT * FROM users").fetchone())
            session = dict(db.execute("SELECT * FROM sessions").fetchone())
        assert row["password_hash"] != "long-test-password"
        assert session["token_hash"] != token
        assert c.get("/api/v1/me").status_code == 200
        assert c.post("/api/v1/logout", headers=HEADERS).status_code == 200
        assert store.get_user(token) is None
        assert c.get("/api/v1/me").status_code == 401


def test_document_list_status_and_retry_are_owner_scoped(workspace_app):
    with TestClient(workspace_app) as a, TestClient(workspace_app) as b:
        account(a, "alice")
        account(b, "bob")
        job = a.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("notes.md", b"# Retry\nRetry at most three times.", "text/markdown")},
        ).json()
        assert len(a.get("/api/v1/documents").json()["documents"]) == 1
        assert b.get("/api/v1/documents").json()["documents"] == []
        assert b.get("/api/v1/documents/" + job["id"]).status_code == 404
        assert (
            b.post("/api/v1/documents/" + job["id"] + "/retry", headers=HEADERS).status_code == 404
        )
        assert "path" not in job and "owner" not in job


def test_duplicate_uploads_reuse_a_job_and_do_not_escape_upload_root(workspace_app):
    with TestClient(workspace_app) as c:
        identity = account(c, "alice")
        first = c.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("../../notes.md", b"unique text", "text/plain")},
        )
        second = c.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("other.md", b"unique text", "text/plain")},
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["id"] == second.json()["id"]
        rows = store.list_jobs(identity["id"])
        assert len(rows) == 1
        assert Path(rows[0]["path"]).resolve().is_relative_to(get_settings().upload_dir.resolve())
        assert len(list(get_settings().upload_dir.rglob("*.md"))) == 1


@pytest.mark.parametrize(
    "name,contents,status",
    [
        ("evil.html", b"<script>bad</script>", 415),
        ("fake.pdf", b"not pdf", 422),
        ("empty.txt", b"", 413),
    ],
)
def test_upload_rejects_unsupported_or_invalid_files(workspace_app, name, contents, status):
    with TestClient(workspace_app) as c:
        account(c, "alice")
        assert (
            c.post(
                "/api/v1/documents", headers=HEADERS, files={"file": (name, contents)}
            ).status_code
            == status
        )


def test_claims_are_atomic_and_duplicate_delivery_is_noop(workspace_app, tmp_path, monkeypatch):
    identity = store.create_user("alice", "long-test-password")
    path = get_settings().upload_dir / "alice" / "job" / "test.md"
    path.parent.mkdir(parents=True)
    path.write_text("Test evidence.")
    job, _ = store.new_job(identity["id"], "test.md", path, "digest", "a" * 32)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(store.claim_job, [job["id"], job["id"]]))
    assert sum(c is not None for c in claims) == 1
    claimed = next(c for c in claims if c)
    store.finish_job(claimed, chunks=1)
    assert not process_job(job["id"])


def test_ingestion_collection_comes_from_persisted_owner(workspace_app, monkeypatch):
    from stacklens.indexing.scope import current_collection, user_collection
    from stacklens.workspace import ingestion

    identity = store.create_user("alice", "long-test-password")
    path = get_settings().upload_dir / identity["id"] / "job" / "test.md"
    path.parent.mkdir(parents=True)
    path.write_text("Retries must stop after three attempts.")
    job, _ = store.new_job(identity["id"], "test.md", path, "digest", "b" * 32)
    seen = []

    def index(chunks):
        seen.append(current_collection())
        return len(chunks)

    monkeypatch.setattr(ingestion, "index_documents", index)
    assert process_job(job["id"])
    assert seen == [user_collection(identity["id"])]
    assert store.get_job(job["id"])["status"] == "completed"


def test_failed_publish_keeps_outbox_pending(workspace_app, tmp_path):
    identity = store.create_user("alice", "long-test-password")
    store.new_job(identity["id"], "test.md", tmp_path / "test.md", "digest", "c" * 32)

    class FailedProducer:
        def produce(self, *args, **kwargs):
            kwargs["on_delivery"](RuntimeError("broker down"), None)

        def flush(self, *args):
            return 0

    with pytest.raises(RuntimeError, match="outbox"):
        publish_pending(FailedProducer())
    assert len(store.pending_jobs()) == 1


def test_malformed_kafka_event_never_becomes_a_path(workspace_app):
    assert not handle_event(b'{"job_id":"../../etc/passwd"}')
    assert not handle_event(b"invalid json")


def test_same_chat_id_is_namespaced_by_authenticated_user(workspace_app, monkeypatch):
    from stacklens.indexing.scope import current_collection, user_collection
    from stacklens.types import AnswerResult

    class FakeSession:
        def send(self, question):
            return AnswerResult(answer="Supported answer.")

    class FakeSessions:
        def __init__(self):
            self.seen = []

        def get_or_create(self, sid):
            self.seen.append((sid, current_collection()))
            return sid, FakeSession()

        def __len__(self):
            return 0

    monkeypatch.setattr(workspace.store, "list_jobs", lambda owner: [{"status": "completed"}])
    with TestClient(workspace_app) as a, TestClient(workspace_app) as b:
        alice = account(a, "alice")
        bob = account(b, "bob")
        fake = FakeSessions()
        workspace_app.state.sessions = fake
        for c in (a, b):
            assert (
                c.post(
                    "/api/v1/chat",
                    headers=HEADERS,
                    json={"question": "What is the policy?", "session_id": "d" * 32},
                ).status_code
                == 200
            )
        assert fake.seen == [
            (alice["id"] + ":" + "d" * 32, user_collection(alice["id"])),
            (bob["id"] + ":" + "d" * 32, user_collection(bob["id"])),
        ]


def test_body_limit_rejects_request_before_upload_parser(workspace_app):
    with TestClient(workspace_app) as c:
        account(c, "alice")
        result = c.post(
            "/api/v1/documents", headers={**HEADERS, "Content-Length": "9000000"}, content=b"x"
        )
        assert result.status_code == 413
        assert not get_settings().upload_dir.exists()


def test_missing_body_length_is_rejected(workspace_app):
    with TestClient(workspace_app) as c:
        result = c.post("/api/v1/register", content=iter([b"{}"]), headers=HEADERS)
        assert result.status_code == 411


def test_lease_expiry_requeues_and_old_worker_cannot_finish(workspace_app, tmp_path):
    identity = store.create_user("alice", "long-test-password")
    job, _ = store.new_job(identity["id"], "test.md", tmp_path / "test.md", "lease", "e" * 32)
    old = store.claim_job(job["id"])
    with store.connection() as db:
        db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job["id"],))
    assert store.pending_jobs()[0]["id"] == job["id"]
    new = store.claim_job(job["id"])
    store.finish_job(old, chunks=99)
    assert store.get_job(job["id"])["status"] == "processing"
    store.finish_job(new, chunks=2)
    assert store.get_job(job["id"])["chunks"] == 2
