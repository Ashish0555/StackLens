"""Running the evaluation suite and aggregating the result."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stacklens.agent.service import ask
from stacklens.evaluation.configuration import RunConfiguration, capture_configuration
from stacklens.evaluation.dataset import EvalCase
from stacklens.evaluation.judge import judge_answer
from stacklens.evaluation.metrics import (
    CaseScore,
    error_score,
    retrieved_passages,
    score_case,
)

logger = logging.getLogger(__name__)

AskFunction = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class Rate:
    """A metric as passed-over-applicable, so 0 of 0 stays distinguishable."""

    passed: int
    total: int

    @property
    def ratio(self) -> float | None:
        return self.passed / self.total if self.total else None

    @property
    def percent(self) -> str:
        return "n/a" if self.ratio is None else f"{self.ratio * 100:.0f}%"


@dataclass
class EvalReport:
    """Everything one run of the suite produced."""

    scores: list[CaseScore] = field(default_factory=list)
    started_at: str = ""
    configuration: RunConfiguration | None = None

    @property
    def model(self) -> str:
        return self.configuration.model if self.configuration else ""

    @property
    def retrieval_k(self) -> int:
        return self.configuration.retrieval_k if self.configuration else 0

    @property
    def retrieval_accuracy(self) -> Rate:
        return self._rate(lambda score: score.retrieval_hit)

    @property
    def citation_accuracy(self) -> Rate:
        return self._rate(lambda score: score.citation_correct)

    @property
    def factual_accuracy(self) -> Rate:
        return self._rate(lambda score: score.facts_present)

    @property
    def refusal_accuracy(self) -> Rate:
        return self._rate(lambda score: score.refusal_correct)

    @property
    def judged(self) -> Rate:
        return self._rate(lambda score: score.judged)

    @property
    def groundedness(self) -> Rate:
        return self._rate(lambda score: score.grounded)

    @property
    def ungrounded(self) -> list[CaseScore]:
        return [score for score in self.scores if score.grounded is False]

    @property
    def overall(self) -> Rate:
        return Rate(sum(score.passed for score in self.scores), len(self.scores))

    @property
    def failures(self) -> list[CaseScore]:
        return [score for score in self.scores if not score.passed]

    @property
    def total_cost_usd(self) -> float:
        return sum(score.cost_usd or 0.0 for score in self.scores)

    @property
    def total_tokens(self) -> int:
        return sum(score.total_tokens for score in self.scores)

    @property
    def median_latency(self) -> float:
        if not self.scores:
            return 0.0
        ordered = sorted(score.latency_seconds for score in self.scores)
        return ordered[len(ordered) // 2]

    def _rate(self, pick: Callable[[CaseScore], bool | None]) -> Rate:
        applicable = [value for value in map(pick, self.scores) if value is not None]
        return Rate(sum(applicable), len(applicable))

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "configuration": self.configuration.to_dict() if self.configuration else {},
            "summary": {
                "cases": len(self.scores),
                "overall": self.overall.percent,
                "retrieval_accuracy": self.retrieval_accuracy.percent,
                "citation_accuracy": self.citation_accuracy.percent,
                "factual_accuracy": self.factual_accuracy.percent,
                "refusal_accuracy": self.refusal_accuracy.percent,
                "groundedness": self.groundedness.percent,
                "judged": self.judged.percent,
                "median_latency_seconds": round(self.median_latency, 2),
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 5),
            },
            "cases": [
                {
                    "id": score.case_id,
                    "passed": score.passed,
                    "question": score.question,
                    "answer": score.answer,
                    "retrieved_sources": score.retrieved_sources,
                    "retrieval_hit": score.retrieval_hit,
                    "citation_correct": score.citation_correct,
                    "facts_present": score.facts_present,
                    "refused": score.refused,
                    "refusal_correct": score.refusal_correct,
                    "grounded": score.grounded,
                    "groundedness_ratio": score.groundedness_ratio,
                    "ungrounded_numbers": score.ungrounded_numbers,
                    "judged": score.judged,
                    "judge_reason": score.judge_reason,
                    "latency_seconds": round(score.latency_seconds, 2),
                    "total_tokens": score.total_tokens,
                    "cost_usd": score.cost_usd,
                    "error": score.error,
                }
                for score in self.scores
            ],
        }


def run_evaluation(
    cases: list[EvalCase],
    *,
    ask_function: AskFunction | None = None,
    with_judge: bool = False,
) -> Iterator[CaseScore]:
    """Grade each case in turn, yielding as it goes so callers can show progress.

    The judge is opt-in: it spends tokens on every case and its verdict drifts
    between runs, so it complements the deterministic metrics rather than
    joining them by default.
    """
    answer_for = ask_function or ask

    for case in cases:
        logger.info("Evaluating %s", case.id)
        try:
            result = answer_for(case.question)
        except Exception as error:
            # One pathological question must not end the suite: the other 27
            # results are exactly what tells you how bad the problem is.
            logger.warning("Case %s raised: %s", case.id, error)
            yield error_score(case, error)
            continue

        score = score_case(case, result)  # type: ignore[arg-type]
        yield _with_judgement(score, result) if with_judge else score


def _with_judgement(score: CaseScore, result: Any) -> CaseScore:
    """Ask a second model whether the sentence around the number holds up."""
    verdict = judge_answer(
        question=score.question,
        passages=retrieved_passages(result.messages),
        answer=score.answer,
    )
    if verdict is None:
        return score

    return replace(
        score,
        judged_faithful=verdict.faithful,
        judged_complete=verdict.complete,
        judge_reason=verdict.reason,
    )


def build_report(scores: list[CaseScore]) -> EvalReport:
    """Wrap the scores with the configuration that produced them.

    A number without the settings behind it cannot be compared to the next
    run: 80% on gpt-4o-mini and 80% on gpt-4o are not the same result, and
    neither are two runs whose chunking differed.
    """
    return EvalReport(
        scores=scores,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        configuration=capture_configuration(),
    )


def save_report(report: EvalReport, directory: Path) -> Path:
    """Write the report as JSON and return where it landed."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}.json"

    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
