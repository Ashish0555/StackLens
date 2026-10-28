"""Execute the compiled graph with controlled providers and real tools."""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from stacklens.agent.workflow import (
    REFUSAL,
    GroundedAnswer,
    SupportVerdict,
    build_workflow,
    validate_evidence,
)

PASSAGE = {
    "id": "evidence-1",
    "source": "runbook.md",
    "text": "Retry at most three times before escalating to the on-call engineer.",
}


def answer(quote=None, evidence_id="evidence-1"):
    return GroundedAnswer.model_validate(
        {
            "answerable": True,
            "claims": [
                {
                    "text": "Retry at most three times.",
                    "evidence": [{"evidence_id": evidence_id, "quote": quote or PASSAGE["text"]}],
                }
            ],
        }
    )


class FakePlanner:
    def invoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_documentation",
                    "args": {"question": "retry limit"},
                    "id": "search-call",
                }
            ],
        )


class FakeModel:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.value


def tool():
    return StructuredTool.from_function(
        lambda question: json.dumps({"passages": [PASSAGE]}),
        name="search_documentation",
        description="Retrieve the relevant evidence.",
    )


def test_graph_returns_only_validated_claims():
    graph = build_workflow(
        planner=FakePlanner(),
        drafter=FakeModel(answer()),
        judge=FakeModel(SupportVerdict(supported=True, reason="Supported")),
        tools=[tool()],
    )
    state = graph.invoke({"messages": [HumanMessage(content="What is the retry limit?")]})
    assert state["messages"][-1].content == "Retry at most three times. [source: runbook.md]"
    assert state["errors"] == []


def test_unknown_citation_retries_once_then_refuses():
    drafter = FakeModel(answer(evidence_id="forged"))
    graph = build_workflow(
        planner=FakePlanner(),
        drafter=drafter,
        judge=FakeModel(SupportVerdict(supported=True, reason="irrelevant")),
        tools=[tool()],
    )
    state = graph.invoke({"messages": [HumanMessage(content="What is the retry limit?")]})
    assert state["messages"][-1].content == REFUSAL
    assert drafter.calls == 2


def test_exact_quote_must_exist_in_retrieved_passage():
    assert validate_evidence(answer(quote="Retry infinitely."), {"evidence-1": PASSAGE})


def test_semantically_unsupported_answer_is_not_published():
    graph = build_workflow(
        planner=FakePlanner(),
        drafter=FakeModel(answer()),
        judge=FakeModel(SupportVerdict(supported=False, reason="Claim not entailed")),
        tools=[tool()],
    )
    state = graph.invoke({"messages": [HumanMessage(content="What is the retry limit?")]})
    assert state["messages"][-1].content == REFUSAL


def test_judge_outage_fails_closed():
    class BrokenJudge:
        def invoke(self, messages):
            raise RuntimeError("provider unavailable")

    graph = build_workflow(
        planner=FakePlanner(), drafter=FakeModel(answer()), judge=BrokenJudge(), tools=[tool()]
    )
    state = graph.invoke({"messages": [HumanMessage(content="What is the retry limit?")]})
    assert state["messages"][-1].content == REFUSAL


def test_new_question_cannot_reuse_previous_turn_evidence():
    from langchain_core.messages import ToolMessage

    from stacklens.agent.workflow import collect_evidence

    history = [
        HumanMessage(content="old"),
        ToolMessage(
            name="search_documentation",
            tool_call_id="old",
            content=json.dumps({"passages": [PASSAGE]}),
        ),
        HumanMessage(content="new"),
    ]
    assert collect_evidence(history) == {}
