"""The scanning layer, answered by LLM Guard, configured for this corpus.

LLM Guard is the market standard and it is built for English. Its own
`ALL_SUPPORTED_LANGUAGES` is `["en", "zh"]`, and its default entity list is
`US_SSN` and `US_BANK_NUMBER`. Pointing it at a Portuguese corpus without
configuring it produces a guardrail that refuses the questions it exists to
protect. Measured here, on the questions this project is built to answer:

| Question | PromptInjection | Anonymize |
|---|---|---|
| "qual o prazo máximo de suspensão?" | blocked, 1.00 | ok |
| "what is the maximum suspension period?" | ok | ok |
| "o que diz o Art. 70 da Resolução 160?" | blocked, 0.90 | blocked, 0.80 |
| CPF 529.982.247-25 | blocked | **ok** |

Three conclusions, and each one is a decision recorded below.

**PromptInjection is not used.** All three models it ships are
`protectai/deberta-v3-*-prompt-injection`, trained on English. The same
question passes in English and is refused in Portuguese at a confidence of
1.00, so no threshold separates them. A guardrail that refuses every real
question is worse than no guardrail: it gets turned off, and then nothing is
protected. If a multilingual injection classifier appears, restoring this is
one entry in the list below.

**Anonymize runs on a narrowed entity list.** `PERSON` needs named entity
recognition in the right language and there is none for Portuguese here.
`US_SSN` and `US_BANK_NUMBER` are the wrong country, and one of them is what
read "Resolução 160" as an account number. What is left are the patterns that
do not depend on language at all.

**CPF, CNPJ and API keys are added by regex**, because the library knows none
of them and they are the three that actually matter for a Brazilian financial
institution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Language-neutral entities only. Everything dropped from the library default
# either needs Portuguese NER or belongs to another country's paperwork.
ENTITY_TYPES = [
    "CREDIT_CARD",
    "CREDIT_CARD_RE",
    "CRYPTO",
    "EMAIL_ADDRESS",
    "EMAIL_ADDRESS_RE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "PHONE_NUMBER",
    "UUID",
    # The custom patterns below. entity_types filters by name, so a pattern
    # that is not listed here is built and then never consulted.
    "BR_CPF",
    "BR_CNPJ",
    "API_KEY",
]

# The identifiers the library has never heard of. Both accept the punctuated
# form and the bare digits, because people paste both.
_CPF = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"
_CNPJ = r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b|\b\d{14}\b"

# Someone debugging pastes a key into a question without thinking, and the
# question is about to be sent to a third party and stored in a trace.
_API_KEY = r"\b(?:sk|pk|rk)[-_][A-Za-z0-9_\-]{16,}\b"


@dataclass(frozen=True, slots=True)
class ScanResult:
    """What the scanners made of one question."""

    valid: bool
    failed: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return ", ".join(self.failed)


class ScannersUnavailable(RuntimeError):
    """LLM Guard is not importable, so the scanning layer cannot run."""


def _regex_patterns() -> list[dict[str, Any]]:
    """The identifiers this corpus needs and the library does not know."""
    return [
        {
            "name": "BR_CPF",
            "expressions": [_CPF],
            "examples": ["529.982.247-25", "52998224725"],
            "context": ["cpf", "documento"],
            "score": 0.9,
            "languages": ["en"],
        },
        {
            "name": "BR_CNPJ",
            "expressions": [_CNPJ],
            "examples": ["11.222.333/0001-81"],
            "context": ["cnpj", "empresa"],
            "score": 0.9,
            "languages": ["en"],
        },
        {
            "name": "API_KEY",
            "expressions": [_API_KEY],
            "examples": ["sk-proj-AbCdEf0123456789AbCdEf01"],
            "context": ["token", "chave", "key"],
            "score": 0.9,
            "languages": ["en"],
        },
    ]


@lru_cache(maxsize=1)
def _input_scanners() -> list[object]:
    """Build the scanners once.

    A failure to import is raised rather than swallowed. LLM Guard is a
    declared dependency, so its absence means a broken environment, and a
    guardrail that quietly turns itself off is worse than one that is missing
    loudly.
    """
    try:
        from llm_guard.input_scanners import Anonymize, Secrets
        from llm_guard.vault import Vault
    except ImportError as error:
        msg = (
            "llm-guard não está instalado, então a camada de varredura não roda. "
            "Instale as dependências do projeto (pip install -e .) ou defina "
            "GUARDRAIL_SCANNER=none para rodar apenas as verificações básicas."
        )
        raise ScannersUnavailable(msg) from error

    logger.info("Loading the guardrail scanners; the first question pays for this.")

    return [
        Secrets(),
        # The vault is where redacted values would be kept. Nothing reads it:
        # detection is the whole contract here, and a question that trips this
        # is refused rather than rewritten. Rewriting would silently change
        # what the user asked.
        Anonymize(
            Vault(),
            entity_types=ENTITY_TYPES,
            regex_patterns=_regex_patterns(),
        ),
    ]


def scan_question(question: str) -> ScanResult:
    """Run every input scanner and report which ones objected.

    All of them run even after the first failure. Telling someone their
    question was refused for one reason, and then again for another, is worse
    than telling them both at once.
    """
    failed: list[str] = []

    for scanner in _input_scanners():
        _, valid, _score = scanner.scan(question)  # type: ignore[attr-defined]
        if not valid:
            failed.append(type(scanner).__name__)

    return ScanResult(valid=not failed, failed=tuple(failed))


def forget_scanners() -> None:
    """Drop the loaded scanners, after the settings change."""
    _input_scanners.cache_clear()


def scanners_loaded() -> bool:
    """Whether the models are already in memory."""
    return bool(_input_scanners.cache_info().currsize)
