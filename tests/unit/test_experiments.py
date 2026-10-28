"""Running the evaluation on the platform instead of into a folder.

The local suite writes one report per run and leaves it in the repository. A
dataset run puts the same questions in Langfuse, where each case carries its
own trace and two runs compare side by side.
"""

from __future__ import annotations

from typing import Any

import pytest

from stacklens.evaluation import experiments
from stacklens.evaluation.dataset import EvalCase
from stacklens.evaluation.experiments import LangfuseUnavailableError, sync_dataset

SOURCE = "api-contract.pdf"


class FakeDataset:
    def __init__(self) -> None:
        self.experiments: list[dict[str, Any]] = []

    def run_experiment(self, **kwargs: Any) -> str:
        self.experiments.append(kwargs)
        return "resultado"


class FakeClient:
    def __init__(self) -> None:
        self.datasets: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.dataset = FakeDataset()

    def create_dataset(self, **kwargs: Any) -> None:
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs)

    def get_dataset(self, name: str) -> FakeDataset:
        return self.dataset


class FakeItem:
    def __init__(self, question: str, item_id: str = "c1") -> None:
        self.input = {"question": question}
        self.id = item_id


def case(**overrides: Any) -> EvalCase:
    defaults: dict[str, Any] = {
        "id": "lote-suplementar",
        "question": "qual o limite?",
        "expected_source": SOURCE,
        "expected_facts": ["15"],
        "reference_answer": "15%",
        "tags": ["percentual"],
    }
    defaults.update(overrides)
    return EvalCase(**defaults)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    fake = FakeClient()
    monkeypatch.setattr(experiments, "client_or_none", lambda: fake)
    return fake


def output(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "answer": f"O limite é 15% (fonte: {SOURCE})",
        "sources": [SOURCE],
        "groundedness": 1.0,
        "refused": False,
    }
    base.update(overrides)
    return base


def expected(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"answer": "15%", "source": SOURCE, "facts": ["15"]}
    base.update(overrides)
    return base


class TestWithoutLangfuse:
    def test_syncing_says_what_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(experiments, "client_or_none", lambda: None)

        with pytest.raises(LangfuseUnavailableError, match="LANGFUSE_PUBLIC_KEY"):
            sync_dataset([case()])

    def test_running_says_what_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(experiments, "client_or_none", lambda: None)

        with pytest.raises(LangfuseUnavailableError):
            experiments.run_experiment(name="teste")


class TestSyncDataset:
    def test_it_sends_every_case(self, client: FakeClient) -> None:
        sync_dataset([case(id="a"), case(id="b")])

        assert len(client.items) == 2

    def test_the_case_id_becomes_the_item_id(self, client: FakeClient) -> None:
        """Re-syncing then updates items in place instead of duplicating them."""
        sync_dataset([case(id="lote-suplementar")])

        assert client.items[0]["id"] == "lote-suplementar"

    def test_it_carries_what_the_metrics_need(self, client: FakeClient) -> None:
        sync_dataset([case()])
        item = client.items[0]

        assert item["input"]["question"] == "qual o limite?"
        assert item["expected_output"]["source"] == SOURCE
        assert item["expected_output"]["facts"] == ["15"]

    def test_it_marks_the_out_of_corpus_cases(self, client: FakeClient) -> None:
        """The metric that catches invention depends on knowing which they are."""
        sync_dataset([case(id="fora", expected_source=None, expected_facts=[])])

        assert client.items[0]["metadata"]["answerable"] is False


class TestRunExperiment:
    def test_it_names_the_run(self, client: FakeClient) -> None:
        experiments.run_experiment(name="k8-articles")

        assert client.dataset.experiments[0]["name"] == "k8-articles"

    def test_it_registers_every_metric(self, client: FakeClient) -> None:
        experiments.run_experiment(name="x")

        assert len(client.dataset.experiments[0]["evaluators"]) == 5

    def test_it_records_the_configuration_as_metadata(self, client: FakeClient) -> None:
        """A score without the settings behind it cannot be compared."""
        experiments.run_experiment(name="x", metadata={"retrieval_k": 8})

        assert client.dataset.experiments[0]["metadata"]["retrieval_k"] == 8


class TestTask:
    def test_it_returns_what_the_metrics_grade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from langchain_core.messages import AIMessage, ToolMessage

        from stacklens.types import AnswerResult

        result = AnswerResult(
            answer=f"O limite é 15% (fonte: {SOURCE})",
            messages=[
                ToolMessage(
                    content=f"[fonte: {SOURCE} | distância 0.4]\nnão pode ultrapassar 15%",
                    tool_call_id="1",
                    name="search_documentation",
                ),
                AIMessage(content="ok"),
            ],
        )
        monkeypatch.setattr(experiments, "ask", lambda _: result)

        produced = experiments._answer(item=FakeItem("qual o limite?"))

        assert produced["sources"] == [SOURCE]
        assert produced["groundedness"] == 1.0
        assert produced["refused"] is False

    def test_a_failing_case_is_recorded_rather_than_ending_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other results are what tell you how bad the problem is."""

        def explode(_: str) -> Any:
            msg = "Recursion limit of 10 reached"
            raise RuntimeError(msg)

        monkeypatch.setattr(experiments, "ask", explode)

        produced = experiments._answer(item=FakeItem("pergunta impossível"))

        assert produced["answer"] == ""
        assert "RuntimeError" in produced["error"]


class TestEvaluators:
    def test_retrieval_passes_on_the_right_document(self) -> None:
        assert experiments._retrieval(output=output(), expected_output=expected())[0]["value"] == 1

    def test_retrieval_fails_on_the_wrong_one(self) -> None:
        scored = experiments._retrieval(
            output=output(sources=["outro.pdf"]), expected_output=expected()
        )

        assert scored[0]["value"] == 0

    def test_facts_fails_when_the_number_is_missing(self) -> None:
        scored = experiments._facts(
            output=output(answer="O limite é 22%"), expected_output=expected()
        )

        assert scored[0]["value"] == 0
        assert "missing" in scored[0]["comment"]

    def test_refusal_passes_when_the_agent_admits_ignorance(self) -> None:
        scored = experiments._refusal(
            output=output(refused=True), expected_output=expected(source=None)
        )

        assert scored[0]["value"] == 1

    def test_refusal_fails_when_it_invents(self) -> None:
        scored = experiments._refusal(
            output=output(refused=False), expected_output=expected(source=None)
        )

        assert scored[0]["value"] == 0

    def test_groundedness_fails_on_an_unsupported_number(self) -> None:
        assert experiments._grounded(output=output(groundedness=0.5))[0]["value"] == 0

    def test_a_metric_that_does_not_apply_records_nothing(self) -> None:
        """Zero would read as a failure, and null is rejected by the schema."""
        assert experiments._retrieval(output=output(), expected_output=expected(source=None)) == []
        assert experiments._refusal(output=output(), expected_output=expected()) == []
        assert experiments._grounded(output=output(groundedness=None)) == []


class FakeEvaluation:
    def __init__(self, name: str, value: float | None) -> None:
        self.name = name
        self.value = value


class FakeItemResult:
    def __init__(self, *evaluations: FakeEvaluation) -> None:
        self.evaluations = list(evaluations)


class FakeExperimentResult:
    def __init__(self, *items: FakeItemResult) -> None:
        self.item_results = list(items)


class TestSummarise:
    """The platform keeps the detail; the terminal still gets the table.

    Having to open a browser to learn whether the suite passed is a worse
    trade than the history is worth.
    """

    def test_it_counts_passes_per_metric(self) -> None:
        result = FakeExperimentResult(
            FakeItemResult(FakeEvaluation("retrieval", 1), FakeEvaluation("correctness", 1)),
            FakeItemResult(FakeEvaluation("retrieval", 1), FakeEvaluation("correctness", 0)),
        )

        assert experiments.summarise(result) == {"retrieval": (2, 2), "correctness": (1, 2)}

    def test_a_metric_that_did_not_apply_is_counted_in_neither(self) -> None:
        """Skipped is not failed, the same as everywhere else."""
        result = FakeExperimentResult(
            FakeItemResult(FakeEvaluation("refusal", 1)),
            FakeItemResult(FakeEvaluation("refusal", None)),
        )

        assert experiments.summarise(result) == {"refusal": (1, 1)}

    def test_it_reads_evaluations_given_as_dictionaries(self) -> None:
        """The SDK hands back either shape depending on the path taken."""

        class DictItem:
            def __init__(self) -> None:
                self.evaluations = [{"name": "faithfulness", "value": 0}]

        class DictResult:
            def __init__(self) -> None:
                self.item_results = [DictItem()]

        assert experiments.summarise(DictResult()) == {"faithfulness": (0, 1)}

    def test_a_result_with_no_items_summarises_to_nothing(self) -> None:
        assert experiments.summarise(FakeExperimentResult()) == {}

    def test_it_survives_a_result_without_item_results(self) -> None:
        assert experiments.summarise(object()) == {}
