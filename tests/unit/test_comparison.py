"""Comparing two evaluation runs, and recording the settings behind each.

The gap this closes: reports used to record the model and `retrieval_k` and
nothing else, so a score that moved because the chunking changed looked
identical to one that moved for no reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stacklens.config import get_settings
from stacklens.evaluation import (
    Change,
    build_report,
    capture_configuration,
    compare,
    hash_prompt,
    load_report,
    score_case,
)
from stacklens.evaluation.configuration import configuration_from_dict
from tests.unit.test_evaluation import SOURCE, answer, case


def report_with(**verdicts: bool) -> dict[str, Any]:
    """A saved report where each named case passed or failed."""
    scores = [
        score_case(
            case(id=case_id, expected_facts=["15"] if passed else ["99"]),
            answer(f"O limite é 15% (fonte: {SOURCE})", [SOURCE]),
        )
        for case_id, passed in verdicts.items()
    ]
    return build_report(scores).to_dict()


class TestConfigurationCapture:
    def test_records_every_setting_that_moves_the_answer(self) -> None:
        recorded = capture_configuration().to_dict()

        for field in (
            "model",
            "embedding_model",
            "temperature",
            "chunk_strategy",
            "chunk_size",
            "chunk_overlap",
            "article_max_chars",
            "retrieval_k",
            "knowledge_domain",
        ):
            assert field in recorded

    def test_records_the_prompt_and_its_hash(self) -> None:
        recorded = capture_configuration()

        assert recorded.prompt
        assert recorded.prompt_hash == hash_prompt(recorded.prompt)

    def test_a_changed_prompt_changes_the_hash(self) -> None:
        assert hash_prompt("um prompt") != hash_prompt("outro prompt")

    def test_the_hash_is_short_enough_to_read(self) -> None:
        assert len(hash_prompt("qualquer")) == 8

    def test_the_report_carries_the_configuration(self) -> None:
        written = build_report([]).to_dict()

        assert written["configuration"]["chunk_strategy"] == get_settings().chunk_strategy.value

    def test_a_report_without_a_configuration_reads_as_unknown(self) -> None:
        """Older reports predate this. Inventing defaults would be worse."""
        assert configuration_from_dict({"model": "gpt-4o-mini"}) is None


class TestSettingsDiff:
    def test_reports_what_changed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = capture_configuration()

        monkeypatch.setenv("RETRIEVAL_K", "4")
        get_settings.cache_clear()
        after = capture_configuration()

        assert after.differences_from(before)["retrieval_k"] == (before.retrieval_k, 4)

    def test_an_unchanged_run_reports_nothing(self) -> None:
        configuration = capture_configuration()

        assert configuration.differences_from(configuration) == {}

    def test_the_prompt_body_is_not_diffed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two prompts side by side are noise; the hash is the fact."""
        before = capture_configuration()

        monkeypatch.setenv("KNOWLEDGE_DOMAIN", "outro assunto qualquer")
        get_settings.cache_clear()
        differences = capture_configuration().differences_from(before)

        assert "prompt" not in differences
        assert "prompt_hash" in differences


class TestComparison:
    def test_a_case_that_started_failing_is_a_regression(self) -> None:
        baseline = report_with(a=True, b=True)
        current = build_report(
            [
                score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE])),
                score_case(case(id="b", expected_facts=["99"]), answer("resposta", [])),
            ]
        )

        diff = compare(baseline, current)

        assert [c.case_id for c in diff.regressions] == ["b"]

    def test_a_case_that_started_passing_is_a_fix(self) -> None:
        baseline = report_with(a=False)
        current = build_report(
            [score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE]))]
        )

        assert [c.case_id for c in compare(baseline, current).fixes] == ["a"]

    def test_a_case_failing_in_both_runs_is_not_a_regression(self) -> None:
        baseline = report_with(a=False)
        current = build_report([score_case(case(id="a", expected_facts=["99"]), answer("x", []))])

        diff = compare(baseline, current)

        assert diff.regressions == []
        assert diff.cases[0].change is Change.STILL_FAILING

    def test_a_new_case_is_marked_as_added(self) -> None:
        baseline = report_with(a=True)
        current = build_report(
            [
                score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE])),
                score_case(case(id="novo"), answer(f"15% (fonte: {SOURCE})", [SOURCE])),
            ]
        )

        added = [c.case_id for c in compare(baseline, current).cases if c.change is Change.ADDED]
        assert added == ["novo"]

    def test_a_removed_case_is_reported(self) -> None:
        baseline = report_with(a=True, removida=True)
        current = build_report(
            [score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE]))]
        )

        removed = [
            c.case_id for c in compare(baseline, current).cases if c.change is Change.REMOVED
        ]
        assert removed == ["removida"]

    def test_regressions_are_listed_before_anything_else(self) -> None:
        baseline = report_with(quebra=True, conserta=False)
        current = build_report(
            [
                score_case(case(id="quebra", expected_facts=["99"]), answer("x", [])),
                score_case(case(id="conserta"), answer(f"15% (fonte: {SOURCE})", [SOURCE])),
            ]
        )

        assert compare(baseline, current).cases[0].change is Change.BROKEN

    def test_it_shows_which_setting_moved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = report_with(a=True)

        monkeypatch.setenv("RETRIEVAL_K", "4")
        get_settings.cache_clear()
        current = build_report(
            [score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE]))]
        )

        assert "retrieval_k" in compare(baseline, current).settings

    def test_an_unchanged_metric_is_not_listed(self) -> None:
        baseline = report_with(a=True)
        current = build_report(
            [score_case(case(id="a"), answer(f"15% (fonte: {SOURCE})", [SOURCE]))]
        )

        assert compare(baseline, current).changed_metrics == {}

    def test_a_baseline_without_a_configuration_says_so(self) -> None:
        diff = compare({"summary": {}, "cases": []}, build_report([]))

        assert diff.baseline_unknown_configuration is True
        assert diff.settings == {}


class TestLoadReport:
    def test_reads_a_saved_report(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        path.write_text(json.dumps(report_with(a=True)), encoding="utf-8")

        assert load_report(path)["summary"]["cases"] == 1

    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nao-existe"):
            load_report(tmp_path / "nao-existe.json")
