"""The terminal itself, and the checks every command runs before working.

Nothing here decides anything about the domain. It reads a line, prints a
panel, or stops early with a message that names the command to run next.
"""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from stacklens.agent import ask, format_trace
from stacklens.evaluation import LangfuseUnavailableError
from stacklens.guardrails import GuardrailViolation
from stacklens.indexing import DatabaseUnavailableError, count_documents
from stacklens.types import AnswerResult, RunMetrics

console = Console()


_EMPTY_INDEX_HINT = "[yellow]O índice está vazio. Rode primeiro:[/] stacklens ingest"


_EXIT_WORDS = frozenset({"sair", "exit", "quit"})


def _require_index() -> None:
    """Stop early when there is nothing to search: the answer would be useless."""
    if _count_or_exit() == 0:
        console.print(_EMPTY_INDEX_HINT)
        raise typer.Exit(code=1)


def _count_or_exit() -> int:
    """Count indexed chunks, turning an unreachable server into a clean exit."""
    try:
        return count_documents()
    except DatabaseUnavailableError as error:
        console.print(f"[red]{error}")
        raise typer.Exit(code=1) from error


def _read_question() -> str | None:
    """Read one line from the user. None means the user asked to leave."""
    try:
        return console.input("\n[bold cyan]você[/] > ").strip()
    except (KeyboardInterrupt, EOFError):
        return None


def _ask_or_exit(question: str) -> AnswerResult:
    """Ask, and turn a refusal into a message instead of a stack trace."""
    try:
        return ask(question)
    except GuardrailViolation as refusal:
        _print_refusal(refusal)
        raise typer.Exit(code=2) from refusal


def _print_refusal(refusal: GuardrailViolation) -> None:
    """Say what was refused and why, without the traceback.

    A guardrail firing is the system working, not failing. Printing a stack
    trace for it teaches the reader that guardrails are bugs.
    """
    console.print(f"[yellow]Recusado:[/] {escape(refusal.detail)}")


def _render(result: AnswerResult, *, show_trace: bool) -> None:
    if show_trace:
        console.print(
            Panel(
                escape(format_trace(result.messages)),
                title="raciocínio",
                border_style="dim",
            )
        )

    console.print(Panel(escape(result.answer), title="resposta", border_style="green"))

    if result.used_tools:
        console.print(f"[dim]ferramentas usadas: {', '.join(result.tool_names)}")

    if result.metrics:
        console.print(f"[dim]{_format_metrics(result.metrics)}")


def _format_metrics(metrics: RunMetrics) -> str:
    """One line of usage: what the run took and what it cost."""
    parts = [
        f"{metrics.latency_seconds:.2f}s",
        f"{metrics.total_tokens} tokens ({metrics.input_tokens} in / {metrics.output_tokens} out)",
        f"{metrics.tool_calls} tool call(s)",
    ]
    # An unlisted model yields no estimate; showing nothing beats showing a
    # number that looks authoritative and is wrong.
    if metrics.estimated_cost_usd is not None:
        parts.append(f"~US$ {metrics.estimated_cost_usd:.5f}")
    return " · ".join(parts)


def _with_langfuse[T](action: Callable[[], T]) -> T:
    """Run something that needs the platform, or exit saying it is missing."""
    try:
        return action()
    except LangfuseUnavailableError as error:
        console.print(f"[yellow]{error}")
        raise typer.Exit(code=1) from error
