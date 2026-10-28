"""Command line interface: presentation only, no domain logic.

One module per group of commands, and one Typer application assembled from
them here. It was a single 742-line module, which is what a file becomes when
every new command is appended to the end of the last one.

    stacklens ingest              build the index from data/
    stacklens sources             list what is indexed, or drop one document
    stacklens ask "question"      ask the agent once
    stacklens chat                talk to the agent with memory
    stacklens status              inspect the current configuration and index
    stacklens eval                grade the agent against known answers
    stacklens serve               start the HTTP API
    stacklens prompt show|push    read and publish the prompts
    stacklens dataset push        upload the evaluation dataset

The commands are registered rather than re-declared: each module owns a Typer
instance holding its own, and `add_typer` folds them into one application. A
new group is a new module and one line here.
"""

from __future__ import annotations

import typer

from stacklens.cli import asking, evaluating, indexing, prompting, reviewing, serving
from stacklens.cli.console import console

app = typer.Typer(
    add_completion=False,
    help="Agente RAG sobre a sua própria base de documentos.",
)

for group in (indexing, asking, evaluating, reviewing, serving):
    app.registered_commands.extend(group.commands.registered_commands)

app.add_typer(prompting.prompt_app, name="prompt")
app.add_typer(prompting.dataset_app, name="dataset")

__all__ = ["app", "console"]


if __name__ == "__main__":
    app()
