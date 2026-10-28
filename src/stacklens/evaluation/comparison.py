"""Comparing two evaluation runs.

A directory of reports records what happened. It does not say what changed,
and reading two JSON files side by side to find out is how a history stops
being used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from stacklens.evaluation.configuration import RunConfiguration, configuration_from_dict
from stacklens.evaluation.runner import EvalReport


class Change(StrEnum):
    """What happened to one case between two runs."""

    FIXED = "corrigido"
    BROKEN = "quebrado"
    STILL_FAILING = "ainda falha"
    ADDED = "novo"
    REMOVED = "removido"


@dataclass(frozen=True, slots=True)
class CaseChange:
    case_id: str
    change: Change

    @property
    def is_regression(self) -> bool:
        return self.change is Change.BROKEN


@dataclass(frozen=True, slots=True)
class Comparison:
    """What moved between a baseline run and the current one."""

    metrics: dict[str, tuple[str, str]]
    cases: list[CaseChange]
    settings: dict[str, tuple[Any, Any]]
    baseline_unknown_configuration: bool

    @property
    def regressions(self) -> list[CaseChange]:
        return [case for case in self.cases if case.is_regression]

    @property
    def fixes(self) -> list[CaseChange]:
        return [case for case in self.cases if case.change is Change.FIXED]

    @property
    def changed_metrics(self) -> dict[str, tuple[str, str]]:
        return {name: pair for name, pair in self.metrics.items() if pair[0] != pair[1]}


def load_report(path: Path) -> dict[str, Any]:
    """Read a saved report, failing with the path rather than a decode error."""
    if not path.is_file():
        msg = f"Relatório não encontrado: {path}"
        raise FileNotFoundError(msg)

    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline: dict[str, Any], current: EvalReport) -> Comparison:
    """Diff a saved report against the run that just finished."""
    current_data = current.to_dict()

    baseline_config = configuration_from_dict(baseline.get("configuration", {}))
    settings = _settings_diff(baseline_config, current.configuration)

    return Comparison(
        metrics=_metric_diff(baseline.get("summary", {}), current_data["summary"]),
        cases=_case_diff(baseline.get("cases", []), current_data["cases"]),
        settings=settings,
        baseline_unknown_configuration=baseline_config is None,
    )


def _metric_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Every metric present in either run, as (before, after)."""
    names = [
        "overall",
        "retrieval_accuracy",
        "citation_accuracy",
        "factual_accuracy",
        "refusal_accuracy",
        "groundedness",
    ]
    return {
        name: (str(before.get(name, "—")), str(after.get(name, "—")))
        for name in names
        if name in before or name in after
    }


def _case_diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[CaseChange]:
    """Which cases changed verdict, ordered so regressions read first."""
    was = {case["id"]: bool(case.get("passed")) for case in before}
    now = {case["id"]: bool(case.get("passed")) for case in after}

    changes: list[CaseChange] = []

    for case_id, passing in now.items():
        if case_id not in was:
            changes.append(CaseChange(case_id, Change.ADDED))
        elif was[case_id] and not passing:
            changes.append(CaseChange(case_id, Change.BROKEN))
        elif not was[case_id] and passing:
            changes.append(CaseChange(case_id, Change.FIXED))
        elif not passing:
            changes.append(CaseChange(case_id, Change.STILL_FAILING))

    changes.extend(CaseChange(case_id, Change.REMOVED) for case_id in was if case_id not in now)

    order = {
        Change.BROKEN: 0,
        Change.FIXED: 1,
        Change.STILL_FAILING: 2,
        Change.ADDED: 3,
        Change.REMOVED: 4,
    }
    return sorted(changes, key=lambda change: (order[change.change], change.case_id))


def _settings_diff(
    baseline: RunConfiguration | None,
    current: RunConfiguration | None,
) -> dict[str, tuple[Any, Any]]:
    if baseline is None or current is None:
        return {}
    return current.differences_from(baseline)
