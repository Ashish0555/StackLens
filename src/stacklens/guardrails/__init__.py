"""Guardrails: what is refused before the model sees it, and what is flagged after.

Two layers, and they are separate because they fail differently.

The cheap layer is arithmetic on the string. An empty question, one long
enough to be an attack by cost alone, an answer that ran past its token
ceiling. None of it needs a model and none of it can be wrong.

The scanning layer is a classifier. Prompt injection and personal data are
judgements, not facts, so they carry a confidence and they can be wrong in
both directions. That layer is LLM Guard, and it is a hard dependency rather
than an option: a guardrail nobody installed protects nobody.

Where this runs matters. `agent/service.py` calls it, not the API and not the
CLI, so every interface is covered by construction and a new one cannot forget.
"""

from stacklens.guardrails.checks import (
    GuardrailViolation,
    OutputFinding,
    check_answer,
    check_question,
    describe_guardrails,
)
from stacklens.guardrails.injection import InjectionVerdict, classify, scan_chunks

__all__ = [
    "GuardrailViolation",
    "InjectionVerdict",
    "OutputFinding",
    "check_answer",
    "check_question",
    "classify",
    "describe_guardrails",
    "scan_chunks",
]
