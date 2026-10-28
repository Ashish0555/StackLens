"""A second model grading what string matching cannot reach.

The deterministic metrics pass an answer that states the right figure while
inverting the condition around it. Nothing else in the suite reads the
sentence, and this does.
"""

from __future__ import annotations

from typing import Any

import pytest

from stacklens.evaluation import judge as judge_module
from stacklens.evaluation.judge import Judgement, judge_answer
from stacklens.prompts.templates import JUDGE_PROMPT_NAME, PUBLISHED_PROMPTS

PASSAGE = "Os atos relevantes PODEM, excepcionalmente, deixar de ser divulgados."
QUESTION = "Um fato relevante pode deixar de ser divulgado?"


class FakeModel:
    """Stands in for the graded model, recording what it was asked."""

    def __init__(self, verdict: Any = None, *, explodes: bool = False) -> None:
        self.verdict = verdict
        self.explodes = explodes
        self.messages: list[Any] = []

    def with_structured_output(self, schema: Any) -> FakeModel:
        self.schema = schema
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.messages = messages
        if self.explodes:
            msg = "provider is down"
            raise ConnectionError(msg)
        return self.verdict


def verdict(faithful: bool = True, complete: bool = True, reason: str = "ok") -> Any:
    """A verdict shaped like the schema the judge asks for."""
    from stacklens.evaluation.judge import _Verdict

    return _Verdict(faithful=faithful, complete=complete, reason=reason)


def install(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> FakeModel:
    monkeypatch.setattr(judge_module, "build_chat_model", lambda: model)
    return model


class TestVerdicts:
    def test_a_faithful_answer_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeModel(verdict()))

        result = judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim, excepcionalmente.")

        assert result == Judgement(faithful=True, complete=True, reason="ok")
        assert result.passed is True

    def test_an_unfaithful_answer_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The case no deterministic metric catches: no number moved."""
        install(monkeypatch, FakeModel(verdict(faithful=False, reason="inverteu a condição")))

        result = judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim, DEVEM.")

        assert result is not None
        assert result.passed is False
        assert "inverteu" in result.reason

    def test_an_incomplete_answer_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeModel(verdict(complete=False, reason="responde outra coisa")))

        result = judge_answer(question=QUESTION, passages=PASSAGE, answer="O DRI divulga.")

        assert result is not None
        assert result.passed is False

    def test_the_reason_is_trimmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeModel(verdict(reason="  com espaço  ")))

        result = judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.")

        assert result is not None
        assert result.reason == "com espaço"


class TestWhatItIsShown:
    def test_it_receives_the_three_parts_labelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = install(monkeypatch, FakeModel(verdict()))

        judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.")
        _, human = model.messages

        assert QUESTION in human[1]
        assert PASSAGE in human[1]
        assert "PERGUNTA:" in human[1]
        assert "TRECHOS RECUPERADOS:" in human[1]

    def test_no_passages_is_said_rather_than_left_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = install(monkeypatch, FakeModel(verdict()))

        judge_answer(question=QUESTION, passages="", answer="Sim.")

        assert "(nenhum)" in model.messages[1][1]

    def test_the_rubric_is_the_system_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = install(monkeypatch, FakeModel(verdict()))

        judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.")
        system, _ = model.messages

        assert system[0] == "system"
        assert "faithful" in system[1]

    def test_it_asks_for_a_structured_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """So the grade is never parsed out of prose."""
        model = install(monkeypatch, FakeModel(verdict()))

        judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.")

        assert model.schema.__name__ == "_Verdict"


class TestFailure:
    """A grader that can end the run it is grading is worse than one metric fewer."""

    def test_a_failing_model_yields_no_judgement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeModel(explodes=True))

        assert judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.") is None

    def test_an_unexpected_shape_yields_no_judgement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, FakeModel({"faithful": True}))

        assert judge_answer(question=QUESTION, passages=PASSAGE, answer="Sim.") is None


class TestRubric:
    def test_it_is_a_managed_prompt(self) -> None:
        """Tightening the rubric is a version, not a commit."""
        assert JUDGE_PROMPT_NAME in PUBLISHED_PROMPTS

    def test_it_grades_only_the_two_things_it_claims_to(self) -> None:
        rubric = PUBLISHED_PROMPTS[JUDGE_PROMPT_NAME]

        assert "faithful" in rubric
        assert "complete" in rubric
        assert "Do not grade style" in rubric


class TestOptInFromTheRunner:
    def test_the_judge_does_not_run_unless_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It spends tokens on every case and its verdict drifts."""
        from stacklens.evaluation import runner
        from tests.unit.test_evaluation import SOURCE, answer, case

        called = False

        def spy(**_: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(runner, "judge_answer", spy)

        list(
            runner.run_evaluation(
                [case()],
                ask_function=lambda _: answer(f"15% (fonte: {SOURCE})", [SOURCE]),
            )
        )

        assert called is False

    def test_asking_for_it_reaches_the_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from stacklens.evaluation import runner
        from tests.unit.test_evaluation import SOURCE, answer, case

        monkeypatch.setattr(
            runner,
            "judge_answer",
            lambda **_: Judgement(faithful=False, complete=True, reason="inverteu"),
        )

        scores = list(
            runner.run_evaluation(
                [case()],
                ask_function=lambda _: answer(f"15% (fonte: {SOURCE})", [SOURCE]),
                with_judge=True,
            )
        )

        assert scores[0].judged is False
        assert scores[0].judge_reason == "inverteu"
        assert scores[0].passed is False

    def test_a_judge_that_fails_leaves_the_other_metrics_standing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stacklens.evaluation import runner
        from tests.unit.test_evaluation import SOURCE, answer, case

        monkeypatch.setattr(runner, "judge_answer", lambda **_: None)

        scores = list(
            runner.run_evaluation(
                [case()],
                ask_function=lambda _: answer(f"15% (fonte: {SOURCE})", [SOURCE]),
                with_judge=True,
            )
        )

        assert scores[0].judged is None
        assert scores[0].passed is True
