"""Prompts as versioned assets rather than strings compiled into the binary.

What this buys: changing how the agent behaves becomes a label move in a UI
instead of a commit, a build and a deploy. What it must never cost: the agent
answering. An unreachable prompt store falls back to the text in code.
"""

from __future__ import annotations

from typing import Any

import pytest

from stacklens.config import get_settings
from stacklens.observability import tracing as observability
from stacklens.prompts import (
    build_search_tool_description,
    build_system_prompt,
    describe_source,
)
from stacklens.prompts.templates import (
    PUBLISHED_PROMPTS,
    SEARCH_TOOL_PROMPT_NAME,
    SYSTEM_PROMPT_NAME,
    SYSTEM_PROMPT_TEMPLATE,
)


class FakePrompt:
    def __init__(self, template: str, version: int = 3, is_fallback: bool = False) -> None:
        self.prompt = template
        self.version = version
        self.is_fallback = is_fallback


class FakeLangfuse:
    """Records what was asked of it, and answers with a canned prompt."""

    def __init__(self, template: str | None = None, *, explodes: bool = False) -> None:
        self.template = template
        self.explodes = explodes
        self.requested: list[dict[str, Any]] = []
        self.published: list[dict[str, Any]] = []

    def auth_check(self) -> bool:
        return True

    def get_prompt(self, name: str, **kwargs: Any) -> FakePrompt:
        self.requested.append({"name": name, **kwargs})
        if self.explodes:
            msg = "langfuse is down"
            raise ConnectionError(msg)
        if self.template is None:
            # What the SDK does when it cannot reach the platform.
            return FakePrompt(kwargs["fallback"], version=0, is_fallback=True)
        return FakePrompt(self.template)

    def create_prompt(self, **kwargs: Any) -> FakePrompt:
        self.published.append(kwargs)
        return FakePrompt(kwargs["prompt"], version=len(self.published))


@pytest.fixture(autouse=True)
def _fresh_client() -> None:
    observability.reset()


@pytest.fixture
def langfuse(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuse:
    """Tracing configured, with the platform replaced by a fake."""
    import sys
    import types

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    get_settings.cache_clear()

    client = FakeLangfuse("Você é um assistente especializado em {{domain}}. Regra nova.")
    module = types.ModuleType("langfuse")
    module.Langfuse = lambda **_: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    return client


class TestWithoutLangfuse:
    """The default path: no platform configured, prompt lives in code."""

    def test_the_prompt_comes_from_the_code(self) -> None:
        assert describe_source() == ("local", None)

    def test_it_still_renders(self) -> None:
        assert get_settings().knowledge_domain in build_system_prompt()

    def test_no_placeholder_survives(self) -> None:
        assert "{{" not in build_system_prompt()
        assert "{{" not in build_search_tool_description()


class TestWithLangfuse:
    def test_the_published_version_wins(self, langfuse: FakeLangfuse) -> None:
        assert "Regra nova." in build_system_prompt()

    def test_it_reports_the_version_in_force(self, langfuse: FakeLangfuse) -> None:
        assert describe_source() == ("langfuse", 3)

    def test_it_asks_for_the_configured_label(self, langfuse: FakeLangfuse) -> None:
        """Moving the label in the UI is how a version is deployed."""
        build_system_prompt()

        assert langfuse.requested[0]["label"] == get_settings().prompt_label

    def test_it_sends_the_local_text_as_the_fallback(self, langfuse: FakeLangfuse) -> None:
        build_system_prompt()

        assert langfuse.requested[0]["fallback"] == SYSTEM_PROMPT_TEMPLATE

    def test_the_domain_is_still_interpolated(self, langfuse: FakeLangfuse) -> None:
        rendered = build_system_prompt()

        assert get_settings().knowledge_domain in rendered
        assert "{{domain}}" not in rendered

    def test_both_prompts_are_fetched_by_name(self, langfuse: FakeLangfuse) -> None:
        build_system_prompt()
        build_search_tool_description()

        assert {call["name"] for call in langfuse.requested} == {
            SYSTEM_PROMPT_NAME,
            SEARCH_TOOL_PROMPT_NAME,
        }


class TestFallback:
    """A prompt store that can stop the agent answering is worse than none."""

    def test_an_unreachable_platform_falls_back_to_the_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        get_settings.cache_clear()

        module = types.ModuleType("langfuse")
        module.Langfuse = lambda **_: FakeLangfuse(explodes=True)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", module)

        assert describe_source() == ("local", None)
        assert get_settings().knowledge_domain in build_system_prompt()

    def test_the_sdk_returning_its_own_fallback_reads_as_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truthy answer is not proof the platform replied."""
        import sys
        import types

        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        get_settings.cache_clear()

        module = types.ModuleType("langfuse")
        module.Langfuse = lambda **_: FakeLangfuse(None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", module)

        assert describe_source() == ("local", None)


class TestPublishing:
    def test_it_publishes_every_prompt_under_the_label(self, langfuse: FakeLangfuse) -> None:
        for name, template in PUBLISHED_PROMPTS.items():
            observability.publish_prompt(name, template, label="production", commit_message="teste")

        assert [call["name"] for call in langfuse.published] == list(PUBLISHED_PROMPTS)
        assert all(call["labels"] == ["production"] for call in langfuse.published)

    def test_it_publishes_the_template_verbatim(self, langfuse: FakeLangfuse) -> None:
        """One placeholder syntax on both paths keeps the text unchanged."""
        observability.publish_prompt(
            SYSTEM_PROMPT_NAME, SYSTEM_PROMPT_TEMPLATE, label="production", commit_message="t"
        )

        assert langfuse.published[0]["prompt"] == SYSTEM_PROMPT_TEMPLATE
        assert "{{domain}}" in langfuse.published[0]["prompt"]

    def test_publishing_without_langfuse_reports_nothing(self) -> None:
        assert (
            observability.publish_prompt("x", "y", label="production", commit_message="t") is None
        )


class TestRecordedInTheReport:
    def test_the_evaluation_report_records_where_the_prompt_came_from(
        self, langfuse: FakeLangfuse
    ) -> None:
        """A run graded against one version cannot be compared to another."""
        from stacklens.evaluation import capture_configuration

        recorded = capture_configuration()

        assert recorded.prompt_source == "langfuse"
        assert recorded.prompt_version == 3

    def test_without_langfuse_it_records_local(self) -> None:
        from stacklens.evaluation import capture_configuration

        recorded = capture_configuration()

        assert recorded.prompt_source == "local"
        assert recorded.prompt_version is None
