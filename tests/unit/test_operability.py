"""What an orchestrator and a load balancer see.

Two probes with different jobs, and a ceiling per caller. None of it changes
an answer, and all of it decides whether answers keep happening.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient as BaseTestClient

from stacklens.api import routes
from stacklens.api.app import create_app
from stacklens.config import get_settings
from stacklens.indexing import DatabaseUnavailableError

REDIS_URI = "redis://localhost:6379/15"


def redis_is_up() -> bool:
    """Whether the shared store is reachable, so the test can skip rather than fail."""
    from limits.storage import storage_from_string

    try:
        return bool(storage_from_string(REDIS_URI).check())
    except Exception:
        return False


requires_redis = pytest.mark.skipif(
    not redis_is_up(),
    reason="needs a running Redis (docker compose up -d redis)",
)


class TestClient(BaseTestClient):
    __test__ = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("headers", {"X-API-Key": "test-admin-key"})
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def admin_configuration(monkeypatch):
    from stacklens.config import get_settings

    monkeypatch.setenv("API_KEY", "test-admin-key")
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(routes, "count_documents", lambda: 590)

    with TestClient(create_app(), raise_server_exceptions=False) as client:
        yield client


class TestLiveness:
    def test_it_answers_ok(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_it_stays_up_when_the_database_is_away(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A liveness probe that fails on a dependency causes a restart loop.

        The process is fine. Restarting every replica because the database
        blinked is how one database problem becomes an outage.
        """

        def unreachable() -> int:
            raise DatabaseUnavailableError("Postgres fora do ar")

        monkeypatch.setattr(routes, "count_documents", unreachable)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200

    def test_it_stays_up_with_an_empty_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(routes, "count_documents", lambda: 0)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200


class TestReadiness:
    def test_a_working_instance_is_ready(self, client: TestClient) -> None:
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json()["indexed_chunks"] == 590

    def test_an_unreachable_database_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """503 takes the instance out of rotation and leaves it running."""

        def unreachable() -> int:
            raise DatabaseUnavailableError("Não foi possível conectar ao Postgres em db:5432.")

        monkeypatch.setattr(routes, "count_documents", unreachable)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.get("/ready")

        assert response.status_code == 503
        assert "db:5432" in response.json()["detail"]

    def test_an_empty_index_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It answers "não encontrei" to everything, however healthy it is."""
        monkeypatch.setattr(routes, "count_documents", lambda: 0)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.get("/ready")

        assert response.status_code == 503
        assert "stacklens ingest" in response.json()["detail"]

    def test_it_never_leaks_the_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:s3cr3t@db:5432/rag")
        get_settings.cache_clear()
        monkeypatch.setattr(routes, "count_documents", lambda: 590)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            assert "s3cr3t" not in client.get("/ready").text


class TestRateLimit:
    @pytest.fixture
    def limited(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
        monkeypatch.setenv("RATE_LIMIT", "3/minute")
        get_settings.cache_clear()
        monkeypatch.setattr(routes, "count_documents", lambda: 590)
        monkeypatch.setattr(routes, "ask", lambda question: _answer())

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            yield client

    def test_it_refuses_past_the_ceiling(self, limited: TestClient) -> None:
        """One client in a retry loop is one client's problem, not the budget's."""
        codes = [
            limited.post("/ask", json={"question": f"pergunta {n}"}).status_code for n in range(5)
        ]

        assert codes[:3] == [200, 200, 200]
        assert codes[3:] == [429, 429]

    def test_it_says_how_much_is_left(self, limited: TestClient) -> None:
        """A client can slow down before being refused rather than after."""
        response = limited.post("/ask", json={"question": "x"})

        assert "x-ratelimit-remaining" in {name.lower() for name in response.headers}

    def test_the_probes_are_never_limited(self, limited: TestClient) -> None:
        """A balancer polling every two seconds would exhaust the budget alone."""
        codes = {limited.get("/health").status_code for _ in range(10)}

        assert codes == {200}

    def test_readiness_is_not_limited_either(self, limited: TestClient) -> None:
        codes = {limited.get("/ready").status_code for _ in range(10)}

        assert codes == {200}

    def test_an_empty_setting_disables_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`stacklens serve` on a laptop must not refuse the eleventh question."""
        monkeypatch.setenv("RATE_LIMIT", "")
        get_settings.cache_clear()
        monkeypatch.setattr(routes, "count_documents", lambda: 590)
        monkeypatch.setattr(routes, "ask", lambda question: _answer())

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            codes = {client.post("/ask", json={"question": "x"}).status_code for _ in range(8)}

        assert codes == {200}


class TestSharedCounters:
    """Where the counters live decides whether the ceiling is one ceiling.

    Four workers each keeping their own counters enforce 60/minute four times
    over, so the real ceiling is 240. The tests below use the limiter directly
    rather than through the app, because what is being checked is two
    processes sharing state and a test client is one process.
    """

    def build(self, uri: str) -> tuple[object, object, object]:
        from limits import parse
        from limits.strategies import MovingWindowRateLimiter

        from stacklens.api.limits import _build_storage

        return (
            parse("3/minute"),
            MovingWindowRateLimiter(_build_storage(uri)),
            MovingWindowRateLimiter(_build_storage(uri)),
        )

    def test_in_process_counters_multiply_the_ceiling(self) -> None:
        """The default, and the reason the setting exists."""
        from uuid import uuid4

        item, first, second = self.build("")
        caller = uuid4().hex

        allowed = [
            (first if index % 2 == 0 else second).hit(item, caller)  # type: ignore[attr-defined]
            for index in range(6)
        ]

        assert allowed == [True] * 6

    @requires_redis
    def test_shared_counters_keep_one_ceiling(self) -> None:
        from uuid import uuid4

        item, first, second = self.build(REDIS_URI)
        caller = uuid4().hex

        allowed = [
            (first if index % 2 == 0 else second).hit(item, caller)  # type: ignore[attr-defined]
            for index in range(6)
        ]

        assert allowed == [True, True, True, False, False, False]

    def test_an_unreachable_store_falls_back_instead_of_failing(self) -> None:
        """Accuracy is the right thing to lose, not availability.

        Per-worker counting is a ceiling enforced too loosely. Refusing to
        start is no ceiling at all, and no service either.
        """
        from limits.storage import MemoryStorage

        from stacklens.api.limits import _build_storage

        assert isinstance(_build_storage("redis://127.0.0.1:1/0"), MemoryStorage)

    def test_an_async_uri_is_refused_rather_than_used(self) -> None:
        """This middleware is synchronous and cannot drive an async storage."""
        from limits.storage import MemoryStorage

        from stacklens.api.limits import _build_storage

        assert isinstance(_build_storage("async+memory://"), MemoryStorage)


class TestCallerIdentity:
    def test_the_key_is_hashed_before_it_is_stored(self) -> None:
        """The identifier reaches the limiter's storage and its error messages."""
        from unittest.mock import Mock

        from stacklens.api.limits import identify

        request = Mock()
        request.headers = {"X-API-Key": "chave-secreta"}

        identifier = identify(request)

        assert "chave-secreta" not in identifier
        assert identifier.startswith("key:")

    def test_without_a_key_the_address_is_used(self) -> None:
        """A shared key still leaves one client's loop as one client's problem."""
        from unittest.mock import Mock

        from stacklens.api.limits import identify

        request = Mock()
        request.headers = {}
        request.client.host = "10.0.0.7"

        assert identify(request) == "ip:10.0.0.7"

    def test_authenticated_session_uses_account_identity(self, monkeypatch) -> None:
        from unittest.mock import Mock

        from stacklens.api.limits import identify

        request = Mock()
        request.headers = {}
        request.cookies = {"stacklens_session": "session-token"}
        request.client.host = "10.0.0.7"
        monkeypatch.setattr(
            "stacklens.api.limits.store.get_user",
            lambda token: {"id": "account-1"} if token == "session-token" else None,
        )

        identifier = identify(request)

        assert identifier.startswith("user:")
        assert identifier != "ip:10.0.0.7"


def _answer() -> object:
    from stacklens.types import AnswerResult

    return AnswerResult(answer="resposta [fonte: r160.pdf]")
