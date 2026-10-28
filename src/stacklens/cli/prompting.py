"""What the agent is told, and the questions it is graded on.

`stacklens prompt show`, `stacklens prompt push`, `stacklens dataset push`. Both subcommand
groups talk to Langfuse and to nothing else, which is why they sit together.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.markup import escape
from rich.panel import Panel

from stacklens.cli.console import _with_langfuse, console
from stacklens.cli.options import CommitMessageOption, DatasetOption, VerboseOption
from stacklens.config import get_settings
from stacklens.evaluation import DATASET_NAME, load_dataset, sync_dataset
from stacklens.observability import publish_prompt, setup_logging
from stacklens.prompts import build_search_tool_description, build_system_prompt, describe_source
from stacklens.prompts.templates import PUBLISHED_PROMPTS

prompt_app = typer.Typer(
    add_completion=False,
    help="Gerencia os prompts publicados no Langfuse.",
)


dataset_app = typer.Typer(
    add_completion=False,
    help="Publica o dataset de avaliação no Langfuse.",
)


@prompt_app.command("show")
def prompt_show(verbose: bool = VerboseOption) -> None:
    """Mostra o prompt em vigor e de onde ele veio."""
    setup_logging(verbose=verbose)
    settings = get_settings()
    source, version = describe_source()

    origem = f"Langfuse v{version}" if version is not None else "código local"
    console.print(f"[bold]origem      [/] {origem}")
    console.print(f"[bold]label       [/] {settings.prompt_label}")
    console.print(f"[bold]domínio     [/] {settings.knowledge_domain}")

    console.print(Panel(escape(build_system_prompt()), title="system", border_style="cyan"))
    console.print(
        Panel(
            escape(build_search_tool_description()),
            title="search_documentation",
            border_style="dim",
        )
    )

    if source == "local":
        console.print(
            "[dim]Publique com [/]stacklens prompt push[dim] para poder versionar e "
            "trocar sem novo deploy."
        )


@prompt_app.command("push")
def prompt_push(
    message: str = CommitMessageOption,
    verbose: bool = VerboseOption,
) -> None:
    """Publica os prompts locais no Langfuse, sob o label configurado."""
    setup_logging(verbose=verbose)
    settings = get_settings()

    if not settings.tracing_enabled:
        console.print("[yellow]Langfuse não configurado. Nada a publicar.")
        raise typer.Exit(code=1)

    for name, template in PUBLISHED_PROMPTS.items():
        version = publish_prompt(
            name, template, label=settings.prompt_label, commit_message=message
        )
        if version is None:
            console.print(f"[red]falhou[/] {name}")
            raise typer.Exit(code=1)
        console.print(f"[green]OK[/] {name} v{version} ({settings.prompt_label})")

    console.print(
        f"[dim]Edite no Langfuse e mova o label {settings.prompt_label} para trocar de versão."
    )


@dataset_app.command("push")
def dataset_push(
    dataset: Path = DatasetOption,
    verbose: bool = VerboseOption,
) -> None:
    """Envia o dataset local para o Langfuse, criando ou atualizando os itens."""
    setup_logging(verbose=verbose)
    cases = load_dataset(dataset)

    with console.status(f"[cyan]Enviando {len(cases)} caso(s)..."):
        total = _with_langfuse(lambda: sync_dataset(cases))

    console.print(f"[green]OK[/] {total} item(ns) em [bold]{DATASET_NAME}[/]")
    console.print("[dim]O arquivo continua no git: dataset versionado com o código é o padrão.")
