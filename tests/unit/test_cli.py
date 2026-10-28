"""The command line, exercised through Typer's own runner.

`CliRunner` invokes a command in this process, which is what makes covering
nine commands affordable: no subprocess, no interpreter start, no real
terminal. Everything below the presentation layer is faked, because what is
being tested here is presentation: what a person sees, and what exit code a
script gets.

The exit codes are the contract. A script that pipes `stacklens eval` into a deploy
gate depends on them more than on anything printed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from stacklens.cli import app
from stacklens.evaluation import CaseScore
from stacklens.indexing import IndexedSource
from stacklens.types import AnswerResult, RunMetrics, ToolCall

runner = CliRunner()


def patch(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    """Replace a name in every cli submodule that imported it.

    The commands live in one module each and import what they need by name, so
    a fake has to land wherever the name was bound. Patching by search rather
    than by path means a command moving between modules does not silently stop
    being faked.
    """
    import sys

    import stacklens.cli  # noqa: F401  imports every submodule

    replaced = [
        module
        for path, module in list(sys.modules.items())
        if path.startswith("stacklens.cli") and hasattr(module, name)
    ]
    for module in replaced:
        monkeypatch.setattr(module, name, value)

    assert replaced, f"{name} is not imported by any cli module"


@pytest.fixture(autouse=True)
def no_logging_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command configures logging; none of them should write a file here."""
    patch(monkeypatch, "setup_logging", lambda **_: None)
    patch(monkeypatch, "flush_traces", lambda: None)


@pytest.fixture
def indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    patch(monkeypatch, "count_documents", lambda: 590)


@pytest.fixture
def empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    patch(monkeypatch, "count_documents", lambda: 0)


def answer(text: str = "O prazo é de 30 dias. [fonte: r160.pdf]") -> AnswerResult:
    return AnswerResult(
        answer=text,
        tool_calls=[ToolCall(name="search_documentation", arguments={"question": "prazo"})],
        messages=[],
        metrics=RunMetrics(
            latency_seconds=1.5,
            input_tokens=800,
            output_tokens=120,
            tool_calls=1,
            model="gpt-4o-mini",
            estimated_cost_usd=0.00042,
        ),
    )


class TestEmptyIndex:
    """Every command that reads the index must stop before answering nothing."""

    @pytest.mark.parametrize(
        "command",
        [["ask", "qual o prazo?"], ["chat"], ["eval"]],
    )
    def test_it_names_the_command_that_fills_the_index(
        self, command: list[str], empty_index: None
    ) -> None:
        result = runner.invoke(app, command)

        assert result.exit_code == 1
        assert "stacklens ingest" in result.stdout

    def test_an_unreachable_database_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver's own error names a port and no remedy."""
        from stacklens.indexing import DatabaseUnavailableError

        def unreachable() -> int:
            raise DatabaseUnavailableError("Não foi possível conectar ao Postgres em db:5432.")

        patch(monkeypatch, "count_documents", unreachable)

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 1
        assert "db:5432" in result.stdout
        assert "Traceback" not in result.stdout


class TestStatus:
    def test_it_reports_the_configuration_and_the_count(self, indexed: None) -> None:
        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "590" in result.stdout
        assert "gpt-4o-mini" in result.stdout

    def test_it_names_every_layer_a_reader_would_ask_about(self, indexed: None) -> None:
        """One screen that answers "what is this thing set to right now"."""
        output = runner.invoke(app, ["status"]).stdout

        for label in ("modelo", "chunk", "índice", "busca", "rerank", "guardrails"):
            assert label in output

    def test_the_password_never_reaches_the_screen(
        self, indexed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from stacklens.config import get_settings

        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://stacklens:s3cr3t@db:5432/rag")
        get_settings.cache_clear()

        assert "s3cr3t" not in runner.invoke(app, ["status"]).stdout


class TestAsk:
    def test_it_prints_the_answer(self, indexed: None, monkeypatch: pytest.MonkeyPatch) -> None:
        patch(monkeypatch, "ask", lambda question: answer())

        result = runner.invoke(app, ["ask", "qual o prazo?"])

        assert result.exit_code == 0
        assert "30 dias" in result.stdout

    def test_it_reports_the_tools_and_the_cost(
        self, indexed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Latency, tokens and cost on every answer, without signing up for anything."""
        patch(monkeypatch, "ask", lambda question: answer())

        output = runner.invoke(app, ["ask", "qual o prazo?"]).stdout

        assert "search_documentation" in output
        assert "920 tokens" in output
        assert "US$" in output

    def test_an_unpriced_model_reports_no_cost(
        self, indexed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Showing nothing beats showing a number that looks authoritative."""
        unpriced = AnswerResult(
            answer="resposta",
            metrics=RunMetrics(
                latency_seconds=1.0,
                input_tokens=10,
                output_tokens=5,
                tool_calls=0,
                model="modelo-desconhecido",
                estimated_cost_usd=None,
            ),
        )
        patch(monkeypatch, "ask", lambda question: unpriced)

        assert "US$" not in runner.invoke(app, ["ask", "x"]).stdout

    def test_the_trace_is_off_unless_asked_for(
        self, indexed: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(monkeypatch, "ask", lambda question: answer())
        patch(monkeypatch, "format_trace", lambda messages: "PENSAMENTO INTERNO")

        assert "PENSAMENTO INTERNO" not in runner.invoke(app, ["ask", "x"]).stdout
        assert "PENSAMENTO INTERNO" in runner.invoke(app, ["ask", "x", "--trace"]).stdout


class TestChat:
    @pytest.fixture
    def session(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        asked: list[str] = []

        class Session:
            def send(self, question: str) -> AnswerResult:
                asked.append(question)
                return answer(f"resposta para {question}")

        patch(monkeypatch, "ChatSession", Session)
        return asked

    def test_it_remembers_across_turns(self, indexed: None, session: list[str]) -> None:
        runner.invoke(app, ["chat"], input="primeira\nsegunda\nsair\n")

        assert session == ["primeira", "segunda"]

    @pytest.mark.parametrize("word", ["sair", "exit", "quit"])
    def test_every_exit_word_works(self, word: str, indexed: None, session: list[str]) -> None:
        result = runner.invoke(app, ["chat"], input=f"pergunta\n{word}\n")

        assert result.exit_code == 0
        assert session == ["pergunta"]

    def test_a_blank_line_is_not_a_question(self, indexed: None, session: list[str]) -> None:
        runner.invoke(app, ["chat"], input="\n\npergunta\nsair\n")

        assert session == ["pergunta"]

    def test_end_of_input_ends_the_conversation(self, indexed: None, session: list[str]) -> None:
        """Ctrl+D should not look like a crash."""
        assert runner.invoke(app, ["chat"], input="pergunta\n").exit_code == 0


class TestIngest:
    @pytest.fixture
    def corpus(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        seen: dict[str, Any] = {"reset": False, "indexed": 0}

        def reset() -> None:
            seen["reset"] = True

        def index(chunks: list[Any]) -> int:
            seen["indexed"] = len(chunks)
            return len(chunks)

        patch(monkeypatch, "load_documents", lambda path: ["doc"])
        patch(monkeypatch, "split_documents", lambda documents, **_: ["a", "b", "c"])
        patch(monkeypatch, "reset_index", reset)
        patch(monkeypatch, "index_documents", index)
        return seen

    def test_it_indexes_what_it_split(self, corpus: dict[str, Any]) -> None:
        result = runner.invoke(app, ["ingest"])

        assert result.exit_code == 0
        assert corpus["indexed"] == 3

    def test_it_does_not_clear_the_index_unasked(self, corpus: dict[str, Any]) -> None:
        runner.invoke(app, ["ingest"])

        assert corpus["reset"] is False

    def test_reset_clears_first(self, corpus: dict[str, Any]) -> None:
        """Required after a chunking change: new text means new ids."""
        runner.invoke(app, ["ingest", "--reset"])

        assert corpus["reset"] is True

    def test_an_empty_data_folder_fails_and_names_the_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch(monkeypatch, "load_documents", lambda path: [])

        result = runner.invoke(app, ["ingest"])

        assert result.exit_code == 1
        assert "data" in result.stdout


class TestSources:
    def test_it_lists_what_is_indexed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch(monkeypatch, "list_sources", lambda: [IndexedSource(name="r160.pdf", chunks=416)])

        output = runner.invoke(app, ["sources"]).stdout

        assert "r160.pdf" in output
        assert "416" in output

    def test_removing_names_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch(monkeypatch, "list_sources", lambda: [IndexedSource(name="r160.pdf", chunks=416)])
        patch(monkeypatch, "delete_source", lambda source: 416)

        output = runner.invoke(app, ["sources", "--remove", "r160.pdf", "--yes"]).stdout

        assert "416" in output


class TestEval:
    @pytest.fixture
    def local_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No Langfuse, so the local path runs and a report is built."""
        from stacklens.config import get_settings

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        get_settings.cache_clear()

        patch(monkeypatch, "load_dataset", lambda path: ["caso"])
        patch(monkeypatch, "run_evaluation", lambda cases, **_: iter([_score()]))
        patch(monkeypatch, "save_report", lambda report, folder: Path("relatorio.json"))

    def test_it_prints_the_table_and_passes(self, indexed: None, local_run: None) -> None:
        result = runner.invoke(app, ["eval", "--min-score", "0.0"])

        assert result.exit_code == 0
        assert "retrieval" in result.stdout

    def test_below_the_threshold_it_fails(self, indexed: None, local_run: None) -> None:
        """This exit code is what a deploy gate reads."""
        result = runner.invoke(app, ["eval", "--min-score", "1.0"])

        assert result.exit_code == 1
        assert "abaixo do limiar" in result.stdout

    def test_no_save_writes_nothing(
        self, indexed: None, local_run: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written: list[Path] = []
        patch(
            monkeypatch, "save_report", lambda report, folder: written.append(folder) or Path("x")
        )

        runner.invoke(app, ["eval", "--min-score", "0.0", "--no-save"])

        assert written == []

    def test_the_cost_ceiling_fails_the_run(
        self, indexed: None, local_run: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A verbose prompt shows up here rather than on the invoice."""
        result = runner.invoke(app, ["eval", "--min-score", "0.0", "--max-cost", "0.0001"])

        assert result.exit_code == 1
        assert "teto de custo" in result.stdout


class TestPrompt:
    def test_push_without_langfuse_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from stacklens.config import get_settings

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        get_settings.cache_clear()

        result = runner.invoke(app, ["prompt", "push"])

        assert result.exit_code == 1
        assert "Langfuse" in result.stdout

    def test_show_reports_where_the_prompt_came_from(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch(monkeypatch, "describe_source", lambda: ("local", None))

        output = runner.invoke(app, ["prompt", "show"]).stdout

        assert "código local" in output
        assert "stacklens prompt push" in output


def _score() -> CaseScore:
    """One case that passes most metrics and fails one, so both paths render."""
    return CaseScore(
        case_id="caso-1",
        question="qual o prazo?",
        answer="30 dias [fonte: r160.pdf]",
        answerable=True,
        retrieved_sources=["r160.pdf"],
        retrieval_hit=True,
        citation_correct=True,
        facts_present=False,
        refused=False,
        refusal_correct=None,
        latency_seconds=1.0,
        total_tokens=120,
        cost_usd=0.0005,
        groundedness_ratio=1.0,
    )
