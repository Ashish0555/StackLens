"""Local repository diff review command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from stacklens.review import review_diff
from stacklens.review.github import GitHubAPIError, review_pull_request

commands = typer.Typer()
console = Console()
DIFF_FILE = typer.Argument(..., exists=True, readable=True, help="Unified diff file.")
REPOSITORY_ROOT = typer.Option(
    Path(),
    "--repo-root",
    exists=True,
    file_okay=False,
    readable=True,
    help="Repository containing the changed files.",
)


@commands.command(name="review-diff")
def review_diff_command(
    diff_file: Path = DIFF_FILE,
    repository_root: Path = REPOSITORY_ROOT,
) -> None:
    """Review a local unified diff against repository context."""
    report = review_diff(diff_file.read_text(encoding="utf-8"), repository_root)
    console.print(f"[bold]Summary:[/] {report.summary}")
    if not report.findings:
        console.print("[green]No actionable findings supported by the supplied context.[/]")
        return
    for finding in report.findings:
        console.print(
            f"\n[bold]{finding.severity.value.upper()}[/] "
            f"{finding.path}:{finding.line} — {finding.title}\n"
            f"{finding.explanation}\n[dim]Suggested fix: {finding.suggested_fix} "
            f"(confidence {finding.confidence:.0%})[/]"
        )


@commands.command(name="review-pr")
def review_pr_command(
    repository: str = typer.Argument(..., help="GitHub repository in OWNER/REPOSITORY format."),
    number: int = typer.Argument(..., min=1, help="Pull-request number."),
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Publish inline comments after the review. Dry-run is the default.",
    ),
) -> None:
    """Review a GitHub pull request, optionally publishing a review."""
    try:
        pull_request, report = review_pull_request(repository, number, publish=publish)
    except GitHubAPIError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"[bold]PR #{pull_request.number}: {pull_request.title}[/]")
    console.print(f"[bold]Summary:[/] {report.summary}")
    if not report.findings:
        console.print("[green]No actionable findings supported by the supplied context.[/]")
    for finding in report.findings:
        console.print(
            f"\n[bold]{finding.severity.value.upper()}[/] "
            f"{finding.path}:{finding.line} — {finding.title}\n"
            f"{finding.explanation}\n[dim]Suggested fix: {finding.suggested_fix} "
            f"(confidence {finding.confidence:.0%})[/]"
        )
    if publish:
        console.print("[green]Review published to GitHub.[/]")
