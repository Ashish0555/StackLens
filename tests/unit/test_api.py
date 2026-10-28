"""The HTTP layer. It must translate, never decide.

The agent is faked throughout: what is under test is the wrapper, not the
model. A test that spends tokens is a test nobody runs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient as BaseTestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from stacklens.api import FeedbackStore, InMemorySessionStore, RedisSessionStore, create_app
from stacklens.config import get_settings
from stacklens.types import AnswerResult, RunMetrics, ToolCall


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


SOURCE = "api-contract.pdf"


def build_result(text: str = "O limite é 15%.") -> AnswerResult:
    return AnswerResult(
        answer=text,
        tool_calls=[ToolCall(name="search_documentation", arguments={"question": "limite"})],
        messages=[
            ToolMessage(
                content=f"--- Trecho 1 [fonte: {SOURCE}, Art. 60 | distância 0.4]\nconteúdo",
                tool_call_id="1",
                name="search_documentation",
            ),
            AIMessage(content=text),
        ],
        metrics=RunMetrics(
            latency_seconds=1.5,
            input_tokens=100,
            output_tokens=20,
            tool_calls=1,
            model="gpt-4o-mini",
            estimated_cost_usd=0.00003,
        ),
    )


class FakeSession:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def send(self, question: str) -> AnswerResult:
        self.questions.append(question)
        return build_result(f"resposta {len(self.questions)}")


@pytest.fixture
def indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A store that answers, so endpoints are not blocked by an empty index."""
    from stacklens.api import routes

    monkeypatch.setattr(routes, "count_documents", lambda: 590)
    monkeypatch.setattr(
        routes, "describe_location", lambda: "postgresql+psycopg://stacklens:***@db:5432/rag"
    )
    monkeypatch.setattr(routes, "ask", lambda question: build_result())
    monkeypatch.setattr(routes, "flush", lambda: None)


@pytest.fixture
def empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from stacklens.api import routes

    monkeypatch.setattr(routes, "count_documents", lambda: 0)
    monkeypatch.setattr(
        routes, "describe_location", lambda: "postgresql+psycopg://stacklens:***@db:5432/rag"
    )


@pytest.fixture
def client(indexed: None) -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as test_client:
        # Replace the store built at startup with one that fakes the agent.
        test_client.app.state.sessions = InMemorySessionStore(factory=FakeSession)  # type: ignore[attr-defined,arg-type]
        yield test_client


class TestProbes:
    """The two probes and what each one is allowed to know.

    They were one endpoint once, and it checked the database. That is
    readiness wearing a liveness name: a dependency blinking would have told
    the orchestrator to restart every replica. The split is covered in full by
    test_operability.py; what is here is that both exist and answer.
    """

    def test_liveness_says_only_that_the_process_answers(self, client: TestClient) -> None:
        body = client.get("/health").json()

        assert body == {"status": "ok"}

    def test_readiness_reports_the_index(self, client: TestClient) -> None:
        body = client.get("/ready").json()

        assert body["status"] == "ready"
        assert body["indexed_chunks"] == 590

    def test_an_empty_index_is_live_but_not_ready(self, empty_index: None) -> None:
        """It answers "não encontrei" to everything, however healthy it is."""
        with TestClient(create_app(), raise_server_exceptions=False) as unindexed:
            assert unindexed.get("/health").status_code == 200
            assert unindexed.get("/ready").status_code == 503


class TestStatus:
    def test_exposes_the_active_configuration(self, client: TestClient) -> None:
        body = client.get("/status").json()

        assert body["chat_model"]
        assert body["chunk_strategy"] in {"characters", "articles"}
        assert body["retrieval_k"] > 0


class TestAsk:
    def test_answers_a_question(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "qual o limite?"}).json()

        assert body["answer"] == "O limite é 15%."

    def test_reports_the_sources_it_retrieved(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "qual o limite?"}).json()

        assert body["sources"] == [SOURCE]

    def test_reports_the_tools_and_the_cost(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "qual o limite?"}).json()

        assert body["tools_used"][0]["name"] == "search_documentation"
        assert body["metrics"]["total_tokens"] == 120
        assert body["metrics"]["estimated_cost_usd"] == 0.00003

    def test_omits_the_trace_unless_asked(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "q"}).json()

        assert body["trace"] is None

    def test_includes_the_trace_on_request(self, client: TestClient) -> None:
        body = client.post("/ask", json={"question": "q", "trace": True}).json()

        assert body["trace"] is not None
        assert "search_documentation" in body["trace"]

    def test_a_single_question_carries_no_session(self, client: TestClient) -> None:
        assert client.post("/ask", json={"question": "q"}).json()["session_id"] is None

    @pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 2001}])
    def test_rejects_an_invalid_body(self, client: TestClient, payload: dict[str, Any]) -> None:
        assert client.post("/ask", json=payload).status_code == 422

    def test_refuses_to_answer_against_an_empty_index(self, empty_index: None) -> None:
        with TestClient(create_app()) as unindexed:
            response = unindexed.post("/ask", json={"question": "q"})

        assert response.status_code == 503
        assert "índice está vazio" in response.json()["detail"]


class TestChat:
    def test_opens_a_session_when_none_is_given(self, client: TestClient) -> None:
        body = client.post("/chat", json={"question": "primeira"}).json()

        assert body["session_id"]

    def test_the_same_session_keeps_talking_to_the_same_conversation(
        self, client: TestClient
    ) -> None:
        first = client.post("/chat", json={"question": "primeira"}).json()
        second = client.post(
            "/chat", json={"question": "segunda", "session_id": first["session_id"]}
        ).json()

        assert second["session_id"] == first["session_id"]

    def test_different_calls_without_an_id_get_different_sessions(self, client: TestClient) -> None:
        first = client.post("/chat", json={"question": "a"}).json()
        second = client.post("/chat", json={"question": "b"}).json()

        assert first["session_id"] != second["session_id"]

    def test_a_session_can_be_ended(self, client: TestClient) -> None:
        session_id = client.post("/chat", json={"question": "a"}).json()["session_id"]

        assert client.delete(f"/chat/{session_id}").status_code == 204

    def test_ending_an_unknown_session_is_a_404(self, client: TestClient) -> None:
        assert client.delete("/chat/nao-existe").status_code == 404


class TestInMemorySessionStore:
    def test_an_unknown_id_opens_a_new_conversation(self) -> None:
        """A client holding an id from before a restart should keep working."""
        store = InMemorySessionStore(factory=FakeSession)  # type: ignore[arg-type]

        returned_id, _ = store.get_or_create("id-de-antes-do-restart")

        assert returned_id == "id-de-antes-do-restart"
        assert len(store) == 1

    def test_the_same_id_returns_the_same_conversation(self) -> None:
        store = InMemorySessionStore(factory=FakeSession)  # type: ignore[arg-type]
        _, first = store.get_or_create("s1")
        _, second = store.get_or_create("s1")

        assert first is second

    def test_the_oldest_session_is_evicted_at_the_cap(self) -> None:
        """Unbounded session storage is a memory leak with a friendly name."""
        store = InMemorySessionStore(max_sessions=2, factory=FakeSession)  # type: ignore[arg-type]
        store.get_or_create("a")
        store.get_or_create("b")
        store.get_or_create("c")

        assert len(store) == 2
        assert store.drop("a") is False
        assert store.drop("c") is True

    def test_using_a_session_keeps_it_from_being_evicted(self) -> None:
        store = InMemorySessionStore(max_sessions=2, factory=FakeSession)  # type: ignore[arg-type]
        store.get_or_create("a")
        store.get_or_create("b")
        store.get_or_create("a")
        store.get_or_create("c")

        assert store.drop("a") is True
        assert store.drop("b") is False


class TestFeedback:
    """The only source of questions the dataset's author did not think of."""

    def test_records_a_verdict(self, client: TestClient) -> None:
        run_id = client.post("/ask", json={"question": "q"}).json()["run_id"]

        response = client.post("/feedback", json={"run_id": run_id, "useful": False})

        assert response.status_code == 200
        assert response.json() == {"recorded": True, "run_id": run_id}

    def test_every_answer_carries_an_id_to_point_at(self, client: TestClient) -> None:
        first = client.post("/ask", json={"question": "a"}).json()["run_id"]
        second = client.post("/ask", json={"question": "b"}).json()["run_id"]

        assert first and second
        assert first != second

    @pytest.mark.parametrize(
        "payload",
        [{}, {"useful": True}, {"run_id": "x"}, {"run_id": "", "useful": True}],
    )
    def test_rejects_an_incomplete_body(self, client: TestClient, payload: dict[str, Any]) -> None:
        assert client.post("/feedback", json=payload).status_code == 422

    def test_it_accepts_a_comment(self, client: TestClient) -> None:
        response = client.post(
            "/feedback",
            json={"run_id": "abc", "useful": False, "comment": "citou o artigo errado"},
        )

        assert response.status_code == 200


class TestFeedbackStore:
    def test_writes_one_line_per_entry(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path)

        store.record({"run_id": "a", "useful": True})
        store.record({"run_id": "b", "useful": False})

        assert len(store.read_all()) == 2

    def test_stamps_the_moment_it_arrived(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path)

        store.record({"run_id": "a", "useful": True})

        assert store.read_all()[0]["recorded_at"]

    def test_creates_the_directory_it_needs(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "nao" / "existe")

        store.record({"run_id": "a", "useful": True})

        assert store.path.is_file()

    def test_reading_an_empty_store_is_not_an_error(self, tmp_path: Path) -> None:
        assert FeedbackStore(tmp_path).read_all() == []

    def test_it_singles_out_the_rejected_answers(self, tmp_path: Path) -> None:
        """Those are the candidates for new evaluation cases."""
        store = FeedbackStore(tmp_path)
        store.record({"run_id": "a", "useful": True})
        store.record({"run_id": "b", "useful": False})
        store.record({"run_id": "c", "useful": False})

        assert [entry["run_id"] for entry in store.unhelpful()] == ["b", "c"]

    def test_concurrent_writes_do_not_interleave(self, tmp_path: Path) -> None:
        """Uvicorn serves sync endpoints from a thread pool."""
        import threading

        store = FeedbackStore(tmp_path)
        threads = [
            threading.Thread(target=store.record, args=({"run_id": str(n), "useful": True},))
            for n in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(store.read_all()) == 20


class FakeRedis:
    """A Redis stand-in: the four calls the store makes, nothing more."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    def setex(self, key: str, ttl: int, value: str | bytes) -> None:
        self.values[key] = value.encode() if isinstance(value, str) else value
        self.ttls[key] = ttl

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def delete(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0

    def keys(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [key for key in self.values if key.startswith(prefix)]


class RestorableSession:
    """A fake session that can be reloaded from stored messages."""

    def __init__(self) -> None:
        self._messages: list[Any] = []

    @property
    def messages(self) -> list[Any]:
        return list(self._messages)

    def restore(self, messages: list[Any], session_id: str | None = None) -> None:
        self._messages = list(messages)

    def send(self, question: str) -> AnswerResult:
        self._messages.append(HumanMessage(question))
        return build_result("resposta")


class TestRedisSessionStore:
    """What Redis buys: a conversation outliving the process that started it."""

    def store(self, client: FakeRedis | None = None) -> RedisSessionStore:
        return RedisSessionStore(
            client or FakeRedis(),
            ttl_seconds=60,
            factory=RestorableSession,  # type: ignore[arg-type]
        )

    def test_a_new_conversation_is_written_immediately(self) -> None:
        client = FakeRedis()

        session_id, _ = self.store(client).get_or_create(None)

        assert f"stacklens:session:{session_id}" in client.values

    def test_a_conversation_survives_a_new_store(self) -> None:
        """A restarted process reads what the previous one wrote."""
        client = FakeRedis()
        session_id, session = self.store(client).get_or_create(None)
        session.send("primeira")
        self.store(client).save(session_id, session)

        _, reloaded = self.store(client).get_or_create(session_id)

        assert [m.content for m in reloaded.messages] == ["primeira"]

    def test_only_the_messages_are_stored(self) -> None:
        """The graph holds closures that cannot be serialised, and need not be."""
        client = FakeRedis()
        session_id, session = self.store(client).get_or_create(None)
        session.send("pergunta")
        self.store(client).save(session_id, session)

        stored = json.loads(client.values[f"stacklens:session:{session_id}"])

        assert list(stored) == ["messages"]

    def test_every_write_refreshes_the_expiry(self) -> None:
        client = FakeRedis()

        session_id, _ = self.store(client).get_or_create(None)

        assert client.ttls[f"stacklens:session:{session_id}"] == 60

    def test_an_unknown_id_opens_a_new_conversation(self) -> None:
        returned_id, _ = self.store().get_or_create("id-de-antes-do-restart")

        assert returned_id == "id-de-antes-do-restart"

    def test_unreadable_data_is_discarded_rather_than_raised(self) -> None:
        client = FakeRedis()
        client.values["stacklens:session:corrompida"] = b"nao e json"

        _, session = self.store(client).get_or_create("corrompida")

        assert session.messages == []

    def test_a_dropped_conversation_is_gone(self) -> None:
        client = FakeRedis()
        session_id, _ = self.store(client).get_or_create(None)

        assert self.store(client).drop(session_id) is True
        assert self.store(client).drop(session_id) is False


class TestAuthentication:
    """A service that spends money per request cannot be open to the port."""

    @pytest.fixture
    def secured(self, monkeypatch: pytest.MonkeyPatch, indexed: None) -> Iterator[TestClient]:
        monkeypatch.setenv("API_KEY", "chave-secreta")
        get_settings.cache_clear()
        with BaseTestClient(create_app()) as client:
            client.app.state.sessions = InMemorySessionStore(factory=FakeSession)  # type: ignore[attr-defined,arg-type]
            yield client

    def test_no_key_configured_disables_admin_routes(self, monkeypatch) -> None:
        monkeypatch.delenv("API_KEY", raising=False)
        get_settings.cache_clear()
        with BaseTestClient(create_app()) as client:
            assert client.get("/health").status_code == 401

    def test_a_request_without_the_header_is_rejected(self, secured: TestClient) -> None:
        assert secured.get("/health").status_code == 401

    def test_a_wrong_key_is_rejected(self, secured: TestClient) -> None:
        response = secured.get("/health", headers={"X-API-Key": "errada"})

        assert response.status_code == 401

    def test_the_right_key_is_accepted(self, secured: TestClient) -> None:
        response = secured.get("/health", headers={"X-API-Key": "chave-secreta"})

        assert response.status_code == 200

    def test_it_protects_the_endpoints_that_cost_money(self, secured: TestClient) -> None:
        assert secured.post("/ask", json={"question": "q"}).status_code == 401
        assert secured.post("/chat", json={"question": "q"}).status_code == 401

    def test_the_rejection_says_which_header_to_send(self, secured: TestClient) -> None:
        assert "X-API-Key" in secured.get("/health").json()["detail"]
