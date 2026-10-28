"""A model grading what string matching cannot reach.

The deterministic metrics check numbers and file names. They pass an answer
that states the right figure while inverting the condition around it: the
regulation says a fact "may" be withheld, the answer says it "must" be, and no
number moved. Catching that needs something that reads.

The cost is what a judge always costs. It spends tokens on every case, and it
drifts, so the same answer can be graded differently twice. That is why it runs
only when asked for and never replaces the deterministic metrics: they say
whether the number is right, this says whether the sentence around it is.

The rubric is a managed prompt like any other, so tightening it is a version in
Langfuse rather than a commit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from stacklens.observability.tracing import fetch_prompt
from stacklens.prompts.templates import JUDGE_PROMPT_NAME, JUDGE_PROMPT_TEMPLATE
from stacklens.providers import build_chat_model

logger = logging.getLogger(__name__)


class _Verdict(BaseModel):
    """The judge's structured answer, so the grade is never parsed out of prose."""

    faithful: bool = Field(description="A resposta diz o mesmo que os trechos.")
    complete: bool = Field(description="A resposta responde o que foi perguntado.")
    reason: str = Field(description="Justificativa em no máximo duas frases.")


@dataclass(frozen=True, slots=True)
class Judgement:
    """What the judge concluded about one answer."""

    faithful: bool
    complete: bool
    reason: str

    @property
    def passed(self) -> bool:
        return self.faithful and self.complete


def judge_answer(*, question: str, passages: str, answer: str) -> Judgement | None:
    """Grade one answer against the passages it was given.

    Returns None when the judge itself fails. A grader that can take down the
    run it is grading is worse than one metric fewer, and the deterministic
    scores for that case still stand.
    """
    rubric, _ = fetch_prompt(JUDGE_PROMPT_NAME, label="production", fallback=JUDGE_PROMPT_TEMPLATE)

    try:
        model = build_chat_model().with_structured_output(_Verdict)
        verdict = model.invoke(
            [
                ("system", rubric),
                ("human", _format_case(question, passages, answer)),
            ]
        )
    except Exception:
        logger.warning("The judge failed to grade an answer.", exc_info=True)
        return None

    if not isinstance(verdict, _Verdict):
        logger.warning("The judge returned an unexpected shape: %r", type(verdict))
        return None

    return Judgement(
        faithful=verdict.faithful,
        complete=verdict.complete,
        reason=verdict.reason.strip(),
    )


def _format_case(question: str, passages: str, answer: str) -> str:
    """Lay out the three parts with labels the model cannot confuse."""
    return (
        f"PERGUNTA:\n{question}\n\n"
        f"TRECHOS RECUPERADOS:\n{passages or '(nenhum)'}\n\n"
        f"RESPOSTA DO ASSISTENTE:\n{answer}"
    )
