"""The evaluation dataset: questions whose answers are known."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from stacklens.config import PROJECT_ROOT

DEFAULT_DATASET = PROJECT_ROOT / "evals" / "dataset.json"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question with everything needed to grade the answer.

    A case with no expected source is deliberately outside the corpus: the
    right behaviour there is refusing to answer, and measuring that is the
    only way to catch a model that invents rather than admits.
    """

    id: str
    question: str
    expected_source: str | None
    expected_facts: list[str]
    reference_answer: str
    tags: list[str]

    @property
    def answerable(self) -> bool:
        return self.expected_source is not None


def load_dataset(path: Path | None = None) -> list[EvalCase]:
    """Read the dataset from disk, failing loudly on a malformed file."""
    source = path or DEFAULT_DATASET

    if not source.is_file():
        msg = f"Dataset de avaliação não encontrado: {source}"
        raise FileNotFoundError(msg)

    raw = json.loads(source.read_text(encoding="utf-8"))
    cases = [
        EvalCase(
            id=case["id"],
            question=case["question"],
            expected_source=case.get("expected_source"),
            expected_facts=list(case.get("expected_facts", [])),
            reference_answer=case.get("reference_answer", ""),
            tags=list(case.get("tags", [])),
        )
        for case in raw["cases"]
    ]

    _reject_duplicate_ids(cases)
    return cases


def _reject_duplicate_ids(cases: list[EvalCase]) -> None:
    """Duplicate ids would silently overwrite results in the report."""
    counts = Counter(case.id for case in cases)
    duplicates = [case_id for case_id, count in counts.items() if count > 1]

    if duplicates:
        msg = f"Ids repetidos no dataset: {sorted(duplicates)}"
        raise ValueError(msg)
