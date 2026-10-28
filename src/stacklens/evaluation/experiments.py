"""Running the evaluation as a Langfuse experiment.

The local suite writes a report per run and leaves it in the repository. That
works until the directory grows past what anyone opens, and it never gives a
comparison beyond the one command that reads two files.

A dataset run puts the same questions on the platform: each case gets its own
trace, the scores hang off it, and two runs sit side by side in a UI built for
that. The dataset file stays in git, because a dataset versioned with the code
is what makes a score reproducible.

The metrics are the same ones the local suite uses. They are deterministic and
cost nothing, and reusing them means the two paths cannot disagree.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from stacklens.agent.service import ask
from stacklens.evaluation.dataset import EvalCase
from stacklens.evaluation.judge import judge_answer
from stacklens.evaluation.metrics import extract_retrieved_sources as sources_of
from stacklens.evaluation.metrics import groundedness, is_refusal, retrieved_passages
from stacklens.observability.tracing import client_or_none

logger = logging.getLogger(__name__)

DATASET_NAME = "stacklens-engineering"
DATASET_DESCRIPTION = (
    "Perguntas verificadas contra o texto das resoluções indexadas. "
    "As de fora do corpus medem se o agente admite não saber."
)


class LangfuseUnavailableError(RuntimeError):
    """Langfuse is not configured, so there is no platform to run against."""


def sync_dataset(cases: list[EvalCase], *, name: str = DATASET_NAME) -> int:
    """Publish the local dataset to Langfuse. Returns how many items were sent.

    The case id becomes the item id, so re-running this updates the items in
    place instead of duplicating them, the same way ingestion is idempotent.
    """
    client = _require_client()

    client.create_dataset(name=name, description=DATASET_DESCRIPTION)

    for case in cases:
        client.create_dataset_item(
            dataset_name=name,
            id=case.id,
            input={"question": case.question},
            expected_output={
                "answer": case.reference_answer,
                "source": case.expected_source,
                "facts": case.expected_facts,
            },
            metadata={
                "answerable": case.answerable,
                "tags": case.tags,
            },
        )

    logger.info("Synced %d item(s) to the dataset %r", len(cases), name)
    return len(cases)


def run_experiment(
    *,
    name: str,
    description: str = "",
    dataset_name: str = DATASET_NAME,
    metadata: dict[str, Any] | None = None,
    with_judge: bool = False,
) -> Any:
    """Run every dataset item through the agent, as a tracked experiment.

    Langfuse creates the run, links each item to the trace that answered it,
    and attaches the scores. Comparing this run with the previous one is then
    a page in the UI rather than a command reading two files.
    """
    client = _require_client()
    dataset = client.get_dataset(dataset_name)

    # Annotated explicitly: the evaluators differ in which keyword arguments
    # they read, and inference would pin the list to the first one.
    evaluators: list[Callable[..., list[dict[str, Any]]]] = [
        _retrieval,
        _citation,
        _facts,
        _refusal,
        _grounded,
    ]
    if with_judge:
        evaluators.append(_judged)

    return dataset.run_experiment(
        name=name,
        description=description,
        task=_answer,
        evaluators=evaluators,
        metadata=metadata or {},
    )


def summarise(result: Any) -> dict[str, tuple[int, int]]:
    """Count passes per metric, so a platform run reports like a local one.

    Langfuse keeps the detail in its UI and hands back a result object. Reading
    it here means the same table appears at the end of the run either way, and
    nobody has to open a browser to learn whether the suite passed.

    Returns, per metric, how many passed out of how many applied. A metric that
    did not apply is counted in neither, the same as everywhere else.
    """
    counts: dict[str, list[int]] = {}

    for item in getattr(result, "item_results", []):
        for evaluation in getattr(item, "evaluations", []) or []:
            name = _name_of(evaluation)
            value = _value_of(evaluation)
            if name is None or value is None:
                continue
            tally = counts.setdefault(name, [0, 0])
            tally[1] += 1
            tally[0] += 1 if value >= 1 else 0

    return {name: (passed, total) for name, (passed, total) in counts.items()}


def _name_of(evaluation: Any) -> str | None:
    if isinstance(evaluation, dict):
        return evaluation.get("name")
    return getattr(evaluation, "name", None)


def _value_of(evaluation: Any) -> float | None:
    value = (
        evaluation.get("value")
        if isinstance(evaluation, dict)
        else getattr(evaluation, "value", None)
    )
    return None if value is None else float(value)


def _answer(*, item: Any, **_: Any) -> dict[str, Any]:
    """The task under evaluation: one question, one answer.

    Returns the retrieved sources alongside the text, because the evaluators
    grade both, and re-running the search to find out would measure a
    different retrieval than the one that produced the answer.

    A question the agent cannot satisfy can exhaust its step limit. That is
    recorded as a failed item rather than allowed to end the run, for the same
    reason the local suite isolates one case from the rest: the other results
    are what tell you how bad the problem is.
    """
    question = item.input["question"]

    try:
        result = ask(question)
    except Exception as error:
        logger.warning("Case %s raised: %s", getattr(item, "id", "?"), error)
        return {
            "answer": "",
            "sources": [],
            "passages": "",
            "groundedness": None,
            "refused": False,
            "error": f"{type(error).__name__}: {error}",
        }

    return {
        "answer": result.answer,
        "sources": sources_of(result.messages),
        # Carried so the judge grades against what the agent actually read,
        # rather than a fresh search that might return something else.
        "passages": retrieved_passages(result.messages),
        "groundedness": groundedness(result.answer, result.messages, item.input["question"])[0],
        "refused": is_refusal(result.answer),
    }


def _retrieval(*, output: Any, expected_output: Any, **_: Any) -> list[dict[str, Any]]:
    """Whether the search returned the document the answer lives in."""
    expected = (expected_output or {}).get("source")
    if not expected:
        return _skip("retrieval", "outside the corpus")

    hit = expected in (output or {}).get("sources", [])
    return _score("retrieval", hit, f"expected {expected}")


def _citation(*, output: Any, expected_output: Any, **_: Any) -> list[dict[str, Any]]:
    """Whether the answer names the right source."""
    expected = (expected_output or {}).get("source")
    if not expected:
        return _skip("citation", "outside the corpus")

    return _score("citation", expected in (output or {}).get("answer", ""), f"expected {expected}")


def _facts(*, output: Any, expected_output: Any, **_: Any) -> list[dict[str, Any]]:
    """Whether the expected number or term appears in the answer."""
    expected = (expected_output or {}).get("facts") or []
    if not expected:
        return _skip("correctness", "no expected fact")

    answer = (output or {}).get("answer", "")
    missing = [fact for fact in expected if fact not in answer]
    return _score("correctness", not missing, f"missing {missing}" if missing else "")


def _refusal(*, output: Any, expected_output: Any, **_: Any) -> list[dict[str, Any]]:
    """Outside the corpus, whether the agent admitted it did not know.

    The metric that catches invention, and the only one the answerable cases
    cannot exercise.
    """
    if (expected_output or {}).get("source"):
        return _skip("refusal", "answerable question")

    return _score("refusal", bool((output or {}).get("refused")), "")


def _grounded(*, output: Any, **_: Any) -> list[dict[str, Any]]:
    """Whether every number stated came from what the agent read."""
    ratio = (output or {}).get("groundedness")
    if ratio is None:
        return _skip("groundedness", "no number stated")

    return _score("groundedness", ratio == 1.0, f"{ratio:.0%} of the numbers supported")


def _judged(*, input: Any, output: Any, **_: Any) -> list[dict[str, Any]]:
    """Whether a second model finds the answer faithful to the passages.

    The one evaluator here that is not deterministic, and the only one that
    reads the sentence rather than matching a string.
    """
    answer = (output or {}).get("answer")
    if not answer:
        return _skip("faithfulness", "no answer")

    verdict = judge_answer(
        question=(input or {}).get("question", ""),
        passages=(output or {}).get("passages", ""),
        answer=answer,
    )
    if verdict is None:
        return _skip("faithfulness", "the judge failed")

    return _score("faithfulness", verdict.passed, verdict.reason)


def _score(name: str, passed: bool, comment: str) -> list[dict[str, Any]]:
    return [{"name": name, "value": 1 if passed else 0, "comment": comment}]


def _skip(name: str, reason: str) -> list[dict[str, Any]]:
    """A metric that does not apply records nothing at all.

    Langfuse takes a list of evaluations, so an empty one is the honest way to
    say the metric did not apply. Sending a zero would read as a failure, and
    sending a null is rejected by the schema.
    """
    logger.debug("Metric %s skipped: %s", name, reason)
    return []


def _require_client() -> Any:
    client = client_or_none()
    if client is None:
        msg = (
            "Langfuse não configurado. Defina LANGFUSE_PUBLIC_KEY e "
            "LANGFUSE_SECRET_KEY para usar datasets e experimentos."
        )
        raise LangfuseUnavailableError(msg)
    return client
