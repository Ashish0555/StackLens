"""Serving the same agent over HTTP.

`stacklens serve`.
"""

from __future__ import annotations

import typer

from stacklens.cli.console import console
from stacklens.cli.options import (
    HostOption,
    PortOption,
    ReloadOption,
    VerboseOption,
    WorkersOption,
)
from stacklens.config import get_settings
from stacklens.observability import setup_logging

commands = typer.Typer()


@commands.command()
def serve(
    host: str = HostOption,
    port: int = PortOption,
    workers: int = WorkersOption,
    reload: bool = ReloadOption,
    verbose: bool = VerboseOption,
) -> None:
    """Sobe a API HTTP. Documentação interativa em /docs."""
    setup_logging(verbose=verbose)

    import uvicorn

    # One process serialises requests: a question that takes eight seconds
    # blocks every other question for those eight seconds. Reload needs a
    # single process to re-import into, so the two are mutually exclusive and
    # saying so beats silently ignoring one of them.
    count = 1 if reload else (workers or get_settings().api_workers)
    if reload and workers:
        console.print("[yellow]--reload usa um processo só; --workers ignorado.")

    console.print(
        f"[green]API em[/] http://{host}:{port}  ·  docs em http://{host}:{port}/docs"
        f"  ·  {count} worker(s)"
    )
    # Passed as an import string rather than the object, because both --reload
    # and --workers re-import the module in fresh processes.
    uvicorn.run("stacklens.api.app:app", host=host, port=port, reload=reload, workers=count)
