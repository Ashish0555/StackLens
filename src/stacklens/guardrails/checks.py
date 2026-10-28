"""The checks themselves, and the decision of what blocks and what only warns.

A question is refused before it costs anything. An answer has already been
paid for by the time it can be judged, so what happens to it is a finding
attached to the result, not an exception thrown at the caller.

The one exception to that is the token ceiling, which is about money rather
than quality: a run that blew past it is reported so the caller knows the
answer was expensive, but the answer is still returned, because throwing it
away would waste what was already spent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from stacklens.config import GuardrailScanner, get_settings

logger = logging.getLogger(__name__)


class GuardrailViolation(ValueError):
    """A question was refused before it reached the model."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class OutputFinding:
    """Something worth knowing about an answer that was already produced."""

    name: str
    detail: str


_EMPTY = "A pergunta está vazia."
_TOO_LONG = "A pergunta tem {length} caracteres; o limite é {limit}."
_REFUSED = "A varredura de segurança encontrou dado sensível na pergunta ({reason})."
_INJECTION = "Parece uma tentativa de injeção de prompt ({label})."


def check_question(question: str) -> None:
    """Refuse a question before it reaches the model, or return quietly.

    Order is deliberate. The arithmetic runs first because it is free and
    because a scanner should never be handed a megabyte of text.
    """
    settings = get_settings()

    if not settings.guardrails_enabled:
        return

    if not question.strip():
        raise GuardrailViolation("empty", _EMPTY)

    limit = settings.max_question_chars
    if len(question) > limit:
        raise GuardrailViolation("too_long", _TOO_LONG.format(length=len(question), limit=limit))

    if settings.guardrail_scanner is GuardrailScanner.NONE:
        return

    from stacklens.guardrails.injection import classify
    from stacklens.guardrails.scanners import scan_question

    result = scan_question(question)
    if not result.valid:
        # The caller is told; this is for the file, where the record lives.
        logger.info("Question refused by the scanners: %s", result.reason)
        raise GuardrailViolation("scanner", _REFUSED.format(reason=result.reason))

    verdict = classify(question)
    if verdict.detected:
        logger.info("Injection suspected: %s at %.2f", verdict.label, verdict.score)
        raise GuardrailViolation("injection", _INJECTION.format(label=verdict.label))


def check_answer(answer: str, *, total_tokens: int, retrieved: bool) -> list[OutputFinding]:
    """Judge an answer that already exists, and report rather than raise.

    **Citation is a finding, not a refusal.** A correct refusal cites nothing,
    and this corpus has questions it cannot answer on purpose. Blocking every
    uncited answer would therefore break the behaviour the evaluation suite
    was built to protect. What the metric `citation` measures offline, this
    records online.
    """
    settings = get_settings()

    if not settings.guardrails_enabled:
        return []

    findings: list[OutputFinding] = []

    ceiling = settings.max_answer_tokens
    if total_tokens > ceiling:
        findings.append(
            OutputFinding(
                name="token_ceiling",
                detail=f"A resposta consumiu {total_tokens} tokens; o teto é {ceiling}.",
            )
        )

    if retrieved and "[source:" not in answer and "[fonte:" not in answer:
        findings.append(
            OutputFinding(
                name="missing_citation",
                detail="A resposta usou trechos recuperados e não citou a fonte.",
            )
        )

    for finding in findings:
        logger.warning("Guardrail finding %s: %s", finding.name, finding.detail)

    return findings


def describe_guardrails() -> str:
    """One line for `stacklens status` and the API status response."""
    settings = get_settings()

    if not settings.guardrails_enabled:
        return "desligados"

    return (
        f"{settings.guardrail_scanner.value} · "
        f"máx {settings.max_question_chars} caracteres · "
        f"teto {settings.max_answer_tokens} tokens"
    )
