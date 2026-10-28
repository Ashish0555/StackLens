"""Agent service: orchestration behaviour, with the model replaced by a fake."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from stacklens.agent import service
from stacklens.agent.service import ChatSession


class FakeAgent:
    """Echoes back the conversation plus a canned reply, like the real graph."""

    def __init__(self, reply: str = "resposta", tool_calls: list[dict[str, Any]] | None = None):
        self.reply = reply
        self.tool_calls = tool_calls or []
        self.invocations: list[list[BaseMessage]] = []

    def invoke(self, state: dict[str, Any], **_: Any) -> dict[str, Any]:
        incoming = list(state["messages"])
        self.invocations.append(incoming)

        produced: list[BaseMessage] = []
        if self.tool_calls:
            produced.append(AIMessage(content="", tool_calls=self.tool_calls))
        produced.append(AIMessage(content=self.reply))

        return {"messages": incoming + produced}


@pytest.fixture
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> FakeAgent:
    agent = FakeAgent()
    monkeypatch.setattr(service, "build_agent", lambda: agent)
    return agent


def test_returns_the_final_answer(fake_agent: FakeAgent) -> None:
    assert service.ask("quanto custa?").answer == "resposta"


def test_reports_the_tools_the_agent_called(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FakeAgent(
        tool_calls=[{"name": "calculate", "args": {"expression": "890*12"}, "id": "1"}]
    )
    monkeypatch.setattr(service, "build_agent", lambda: agent)

    result = service.ask("total anual?")

    assert result.tool_names == ["calculate"]
    assert result.tool_calls[0].arguments == {"expression": "890*12"}


def test_a_single_question_carries_no_history(fake_agent: FakeAgent) -> None:
    service.ask("primeira")

    assert len(fake_agent.invocations) == 1
    assert len(fake_agent.invocations[0]) == 1


class TestChatSession:
    def test_resends_the_whole_history_on_every_turn(self, fake_agent: FakeAgent) -> None:
        session = ChatSession()
        session.send("primeira")
        session.send("segunda")

        assert len(fake_agent.invocations[0]) == 1
        assert len(fake_agent.invocations[1]) > 1

    def test_keeps_the_earlier_questions_in_memory(self, fake_agent: FakeAgent) -> None:
        session = ChatSession()
        session.send("primeira")
        session.send("segunda")

        asked = [m.content for m in session.messages if isinstance(m, HumanMessage)]
        assert asked == ["primeira", "segunda"]

    def test_exposes_a_copy_so_callers_cannot_corrupt_the_history(
        self, fake_agent: FakeAgent
    ) -> None:
        session = ChatSession()
        session.send("primeira")

        session.messages.clear()

        assert session.messages
