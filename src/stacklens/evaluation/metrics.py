"""Grading one answer against what the dataset expects.

Every metric here is deterministic: same answer, same score, no second model
involved and no cost. That matters because a grader that is itself a language
model can drift, and a suite you cannot trust is worse than no suite.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, ToolMessage

from stacklens.evaluation.dataset import EvalCase
from stacklens.types import AnswerResult

# The tool renders each passage as "[fonte: nome.pdf, Art. 12 | distância ...]".
# Only the file name is captured: the article is useful in a citation but the
# dataset grades at document level.
_SOURCE_LABEL = re.compile(r"\[(?:source|fonte):\s*([^\],|]+)")

# What the system prompt tells the agent to say when nothing was found.
_REFUSAL_MARKERS = (
    "could not find",
    "not in the documentation",
    "no relevant passages",
    "nao encontrei",
    "nao consta",
    "nenhum trecho relevante",
)

# A number, optionally followed by the scale word Portuguese writes it with:
# "15%", "10.680", "R$ 75 milhões".
# The longer scale words come first: an alternation is tried left to right, so
# "mil" listed first would match the opening of "milhões" and read 75 million
# as 75 thousand.
_NUMBER = re.compile(
    r"(\d[\d.,]*\d|\d)\s*(bilh[oõ]es|bilh[aã]o|milh[oõ]es|milh[aã]o|mil)?",
    re.IGNORECASE,
)

_SCALES = {
    "mil": 1_000,
    "milhao": 1_000_000,
    "milhoes": 1_000_000,
    "bilhao": 1_000_000_000,
    "bilhoes": 1_000_000_000,
}

# A separator grouping exactly three digits is a thousands mark, so "10.680"
# is one number. One or two digits after it is a decimal, so "0.15" is not.
_THOUSANDS = re.compile(r"(?<=\d)[.,](?=\d{3}(?!\d))")


@dataclass(frozen=True, slots=True)
class CaseScore:
    """How one case did, metric by metric."""

    case_id: str
    question: str
    answer: str
    answerable: bool
    retrieved_sources: list[str]
    retrieval_hit: bool | None
    citation_correct: bool | None
    facts_present: bool | None
    refused: bool
    refusal_correct: bool | None
    latency_seconds: float
    total_tokens: int
    cost_usd: float | None
    groundedness_ratio: float | None = None
    ungrounded_numbers: list[str] = field(default_factory=list)
    judged_faithful: bool | None = None
    judged_complete: bool | None = None
    judge_reason: str | None = None
    error: str | None = None

    @property
    def judged(self) -> bool | None:
        """Whether a second model found the answer faithful and complete.

        None when the judge did not run, which is not the same as failing it.
        """
        if self.judged_faithful is None or self.judged_complete is None:
            return None
        return self.judged_faithful and self.judged_complete

    @property
    def grounded(self) -> bool | None:
        """Whether every number stated was supported. None when none was."""
        return None if self.groundedness_ratio is None else self.groundedness_ratio == 1.0

    @property
    def passed(self) -> bool:
        """A case passes when every metric that applies to it passed."""
        if self.error is not None:
            return False

        applicable = [
            value
            for value in (
                self.retrieval_hit,
                self.citation_correct,
                self.facts_present,
                self.refusal_correct,
                self.grounded,
                self.judged,
            )
            if value is not None
        ]
        return all(applicable)


def score_case(case: EvalCase, result: AnswerResult) -> CaseScore:
    """Grade one answer against its case."""
    answer = result.answer
    sources = extract_retrieved_sources(result.messages)
    refused = is_refusal(answer)
    ratio, ungrounded = groundedness(answer, result.messages, case.question)

    if case.answerable:
        expected = case.expected_source or ""
        return CaseScore(
            case_id=case.id,
            question=case.question,
            answer=answer,
            answerable=True,
            retrieved_sources=sources,
            retrieval_hit=expected in sources,
            citation_correct=_mentions(answer, expected),
            facts_present=all(_mentions(answer, fact) for fact in case.expected_facts),
            refused=refused,
            refusal_correct=None,
            latency_seconds=_latency(result),
            total_tokens=_tokens(result),
            cost_usd=_cost(result),
            groundedness_ratio=ratio,
            ungrounded_numbers=ungrounded,
        )

    # Outside the corpus: the only thing that matters is that it admitted so.
    return CaseScore(
        case_id=case.id,
        question=case.question,
        answer=answer,
        answerable=False,
        retrieved_sources=sources,
        retrieval_hit=None,
        citation_correct=None,
        facts_present=None,
        refused=refused,
        refusal_correct=refused,
        latency_seconds=_latency(result),
        total_tokens=_tokens(result),
        cost_usd=_cost(result),
        groundedness_ratio=ratio,
        ungrounded_numbers=ungrounded,
    )


def error_score(case: EvalCase, error: Exception) -> CaseScore:
    """Record a case whose run blew up, so one failure cannot end the suite."""
    return CaseScore(
        case_id=case.id,
        question=case.question,
        answer="",
        answerable=case.answerable,
        retrieved_sources=[],
        retrieval_hit=None,
        citation_correct=None,
        facts_present=None,
        refused=False,
        refusal_correct=None,
        latency_seconds=0.0,
        total_tokens=0,
        cost_usd=None,
        error=f"{type(error).__name__}: {error}",
    )


def groundedness(
    answer: str, messages: list[BaseMessage], question: str
) -> tuple[float | None, list[str]]:
    """How much of the answer is supported by what the agent actually saw.

    Every number the answer states has to appear in the retrieved passages, in
    a tool result, or in the question itself. A number that appears nowhere in
    any of those came from the model's own memory, and the whole reason for
    retrieval is not to rely on that.

    Numbers are the claim worth checking here: in regulation they carry the
    deadlines, the percentages and the limits, and they are what a model most
    confidently invents. Returns the supported ratio and the unsupported
    numbers, or (None, []) when the answer states no number at all.
    """
    stated = _number_occurrences(answer)
    if not stated:
        return None, []

    supporting = _numbers(question)
    for message in messages:
        if isinstance(message, ToolMessage):
            supporting |= _numbers(str(message.content))

    # Each occurrence is compared as a whole, not variant by variant: "75
    # milhões" reads as {75, 75000000}, and finding either form in the source
    # means the answer did not invent it.
    unsupported = sorted(min(variants, key=len) for variants in stated if not variants & supporting)
    return (len(stated) - len(unsupported)) / len(stated), unsupported


def _number_occurrences(text: str) -> list[set[str]]:
    """Each number in the text, as the set of forms that mean the same value.

    "75 milhões" yields {"75", "75000000"} because a tool result may state
    either. Keeping the forms grouped per occurrence is what lets a match on
    any one of them count as support for that number.
    """
    occurrences: list[set[str]] = []

    for match in _NUMBER.finditer(text):
        digits, scale = match.group(1), match.group(2)
        value = _to_number(_THOUSANDS.sub("", digits))
        if value is None:
            continue

        variants = {_render(value)}
        if scale:
            variants.add(_render(value * _SCALES[_fold(scale)]))
        occurrences.append(variants)

    return occurrences


def _numbers(text: str) -> set[str]:
    """Every form of every number in the text, flattened.

    Being lenient here is deliberate: a false alarm on a correct answer would
    make the metric worse than useless.
    """
    return {form for occurrence in _number_occurrences(text) for form in occurrence}


def _to_number(token: str) -> float | None:
    """Read a Portuguese-written number, where the comma is the decimal mark."""
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _render(value: float) -> str:
    """A canonical spelling, so 15 and 15.0 are the same number."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _fold(word: str) -> str:
    """Strip the accents from a scale word so "milhões" reaches the table."""
    return _normalise(word)


def retrieved_passages(messages: list[BaseMessage]) -> str:
    """Everything the search tool returned, as the judge needs to read it."""
    return "\n\n".join(
        str(message.content) for message in messages if isinstance(message, ToolMessage)
    )


def extract_retrieved_sources(messages: list[BaseMessage]) -> list[str]:
    """Every document the search tool actually returned, in order, deduplicated.

    Read from the tool output rather than by re-running the search, so this
    reflects what the agent really saw -- including a second query with
    different terms.
    """
    found: list[str] = []

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        try:
            payload = json.loads(str(message.content))
            for passage in payload.get("passages", []):
                source = passage.get("source")
                if isinstance(source, str) and source not in found:
                    found.append(source)
        except (ValueError, TypeError, AttributeError):
            pass
        for match in _SOURCE_LABEL.findall(str(message.content)):
            source = match.strip()
            if source and source not in found:
                found.append(source)

    return found


def is_refusal(answer: str) -> bool:
    """Whether the answer admits it found nothing."""
    normalised = _normalise(answer)
    return any(marker in normalised for marker in _REFUSAL_MARKERS)


def _mentions(answer: str, needle: str) -> bool:
    return _normalise(needle) in _normalise(answer)


def _normalise(text: str) -> str:
    """Fold case and accents so "não" matches "nao" and "NÃO"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _latency(result: AnswerResult) -> float:
    return result.metrics.latency_seconds if result.metrics else 0.0


def _tokens(result: AnswerResult) -> int:
    return result.metrics.total_tokens if result.metrics else 0


def _cost(result: AnswerResult) -> float | None:
    return result.metrics.estimated_cost_usd if result.metrics else None
