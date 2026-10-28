"""Putting a question to the agent, once or in a conversation.

`stacklens ask`, `stacklens chat`.
"""

from __future__ import annotations

import typer
from rich.markup import escape
from rich.panel import Panel

from stacklens.agent import ChatSession
from stacklens.cli.console import (
    _EXIT_WORDS,
    _ask_or_exit,
    _format_metrics,
    _print_refusal,
    _read_question,
    _render,
    _require_index,
    console,
)
from stacklens.cli.options import TraceOption, VerboseOption
from stacklens.guardrails import GuardrailViolation
from stacklens.observability import flush as flush_traces
from stacklens.observability import setup_logging

commands = typer.Typer()


@commands.command(name="ask")
def ask_command(
    question: str = typer.Argument(..., help="A pergunta em linguagem natural."),
    trace: bool = TraceOption,
    verbose: bool = VerboseOption,
) -> None:
    """Pergunta ao agente, que decide sozinho quais ferramentas usar."""
    setup_logging(verbose=verbose)
    _require_index()

    with console.status("[cyan]Pensando..."):
        result = _ask_or_exit(question)

    _render(result, show_trace=trace)
    flush_traces()


@commands.command()
def chat(trace: bool = TraceOption, verbose: bool = VerboseOption) -> None:
    """Conversa contínua com o agente, com memória entre as perguntas."""
    setup_logging(verbose=verbose)
    _require_index()

    session = ChatSession()
    console.print(
        Panel(
            "Converse com o agente. Ele lembra do que já foi dito.\n"
            "Digite [bold]sair[/] para encerrar.",
            title="modo conversa",
            border_style="cyan",
        )
    )

    while True:
        question = _read_question()
        if question is None:
            console.print("\n[dim]até mais.")
            break
        if not question:
            continue
        if question.lower() in _EXIT_WORDS:
            console.print("[dim]até mais.")
            flush_traces()
            break

        try:
            with console.status("[cyan]pensando..."):
                result = session.send(question)
        except GuardrailViolation as refusal:
            # A conversation survives a refused question: the guardrail runs
            # before the history grows, so the next turn is unaffected.
            _print_refusal(refusal)
            continue

        console.print(f"[bold green]agente[/] > {escape(result.answer)}")
        if trace:
            console.print(f"[dim]ferramentas: {', '.join(result.tool_names) or 'nenhuma'}")
            if result.metrics:
                console.print(f"[dim]{_format_metrics(result.metrics)}")
