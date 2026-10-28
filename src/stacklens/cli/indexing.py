"""Filling the index and looking at what is in it.

`stacklens ingest`, `stacklens sources`, `stacklens status`.
"""

from __future__ import annotations

import typer
from rich.table import Table

from stacklens.cli.console import _count_or_exit, console
from stacklens.cli.options import RemoveOption, ResetOption, VerboseOption, YesOption
from stacklens.config import get_settings
from stacklens.guardrails import describe_guardrails
from stacklens.indexing import (
    delete_source,
    describe_location,
    index_documents,
    list_sources,
    load_documents,
    reranking_enabled,
    reset_index,
    split_documents,
)
from stacklens.observability import setup_logging

commands = typer.Typer()


@commands.command()
def ingest(reset: bool = ResetOption, verbose: bool = VerboseOption) -> None:
    """Lê os documentos de data/, quebra em pedaços e indexa no banco vetorial."""
    setup_logging(verbose=verbose)
    settings = get_settings()

    if reset:
        with console.status("[cyan]Limpando o índice..."):
            reset_index()
        console.print("[green]OK[/] índice limpo")

    with console.status("[cyan]Carregando documentos..."):
        documents = load_documents(settings.data_dir)

    if not documents:
        console.print(f"[yellow]Nenhum documento encontrado em {settings.data_dir}")
        raise typer.Exit(code=1)

    console.print(f"[green]OK[/] {len(documents)} documento(s) carregado(s)")

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        strategy=settings.chunk_strategy,
        article_max_chars=settings.article_max_chars,
    )
    console.print(f"[green]OK[/] {len(chunks)} pedaço(s) gerado(s)")

    with console.status("[cyan]Gerando embeddings e indexando..."):
        total = index_documents(chunks)

    console.print(f"[green]OK[/] {total} pedaço(s) indexado(s) em {describe_location()}")


@commands.command()
def sources(source: str = RemoveOption, yes: bool = YesOption) -> None:
    """Lista os documentos indexados, ou remove um deles do índice."""
    if source is None:
        _print_sources()
        return

    _remove_source(source, confirmed=yes)


def _print_sources() -> None:
    """Show what is indexed, grouped by the document it came from."""
    indexed = list_sources()

    if not indexed:
        console.print("[yellow]O índice está vazio. Rode: stacklens ingest[/]")
        return

    table = Table(title="documentos indexados", border_style="cyan")
    table.add_column("documento", style="bold")
    table.add_column("pedaços", justify="right")

    for entry in indexed:
        table.add_row(entry.name or "[sem origem]", str(entry.chunks))

    console.print(table)
    console.print(
        f"\n{len(indexed)} documento(s), {sum(entry.chunks for entry in indexed)} pedaço(s)"
    )


def _remove_source(source: str, *, confirmed: bool) -> None:
    """Remove one document, after saying exactly what will disappear.

    Re-ingesting overwrites a chunk whose text is unchanged, so it cannot undo
    a removal by itself: a document taken out of the index comes back only if
    its file is still in data/ and ingestion runs again.
    """
    known = {entry.name: entry.chunks for entry in list_sources()}

    if source not in known:
        console.print(f"[red]'{source}' não está indexado.[/]")
        if known:
            console.print("\nIndexados no momento:")
            for name in known:
                console.print(f"  {name}")
        raise typer.Exit(code=1)

    if not confirmed:
        confirmed = typer.confirm(f"Remover {known[source]} pedaço(s) de '{source}'?")

    if not confirmed:
        console.print("Cancelado.")
        return

    removed = delete_source(source)
    console.print(f"[green]Removido:[/] {removed} pedaço(s) de '{source}'")


@commands.command()
def status() -> None:
    """Mostra a configuração ativa e quantos pedaços estão indexados."""
    settings = get_settings()
    total = _count_or_exit()

    console.print(f"[bold]modelo      [/] {settings.chat_model}")
    console.print(f"[bold]embeddings  [/] {settings.embedding_model}")
    console.print(
        f"[bold]chunk       [/] {settings.chunk_strategy.value} · "
        f"{settings.chunk_size} / overlap {settings.chunk_overlap}"
    )
    console.print(f"[bold]domínio     [/] {settings.knowledge_domain}")
    console.print(f"[bold]documentos  [/] {settings.data_dir}")
    console.print(f"[bold]índice      [/] postgres · {describe_location()}")
    console.print(f"[bold]busca       [/] {settings.search_strategy.value}")
    console.print(
        f"[bold]rerank      [/] {settings.rerank_strategy.value}"
        + (
            f" · {settings.rerank_model} · {settings.rerank_candidates} candidatos"
            if reranking_enabled()
            else ""
        )
    )
    console.print(f"[bold]guardrails  [/] {describe_guardrails()}")
    console.print(f"[bold]indexado    [/] [{'green' if total else 'yellow'}]{total} pedaço(s)")
