"""Grading the agent, and printing what the grade means.

`stacklens eval`, locally or on the platform, and the tables both paths print.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from stacklens.cli.console import _require_index, _with_langfuse, console
from stacklens.cli.options import (
    CompareOption,
    DatasetOption,
    JudgeOption,
    LimitOption,
    MaxCostOption,
    MinScoreOption,
    RunNameOption,
    SaveOption,
    VerboseOption,
)
from stacklens.config import PROJECT_ROOT, get_settings
from stacklens.evaluation import (
    CaseScore,
    Comparison,
    EvalReport,
    build_report,
    capture_configuration,
    compare,
    load_dataset,
    load_report,
    run_evaluation,
    run_experiment,
    save_report,
    summarise,
)
from stacklens.observability import setup_logging

commands = typer.Typer()

# One description per metric, so the local table and the platform table say
# the same thing about the same number.
# Names follow the RAG evaluation literature, so a reader who knows RAGAS or
# LangSmith recognises what each one measures. The second column says what the
# name means in this project, since the same word is used loosely elsewhere.
#
# The first four compare the answer against the dataset's expected output.
# The last two compare it against what the agent retrieved, which is why they
# also work on production traffic, where no expected answer exists.
_METRIC_LABELS = {
    "retrieval": "the right document came back",
    "citation": "the answer names the right source",
    "correctness": "the expected number or term is present",
    "refusal": "admitted not knowing, outside the corpus",
    "groundedness": "every number came from what it read",
    "faithfulness": "the sentence matches the passage (model-graded)",
}


def _today() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%d-%H%M")


@commands.command(name="eval")
def eval_command(
    dataset: Path = DatasetOption,
    limit: int = LimitOption,
    save: bool = SaveOption,
    compare_with: Path | None = CompareOption,
    min_score: float = MinScoreOption,
    max_cost: float = MaxCostOption,
    judge: bool = JudgeOption,
    name: str | None = RunNameOption,
    verbose: bool = VerboseOption,
) -> None:
    """Mede o agente contra perguntas cuja resposta é conhecida.

    Com o Langfuse configurado, as perguntas vêm do dataset publicado lá e as
    notas voltam para ele. Sem, o arquivo local é lido e um relatório é
    gravado. O agente e as métricas rodam nesta máquina nos dois casos: nenhuma
    plataforma executa a sua aplicação.
    """
    setup_logging(verbose=verbose)
    _require_index()

    cases = load_dataset(dataset)
    if limit > 0:
        cases = cases[:limit]

    # No flag chooses the destination. Configured Langfuse means the questions
    # come from the dataset there and the scores go back to it; without it the
    # file is read and a report is written. One command, one behaviour, and the
    # decision belongs to the environment rather than to whoever types it.
    if get_settings().tracing_enabled:
        _evaluate_on_langfuse(name=name, with_judge=judge, min_score=min_score)
        return

    scores: list[CaseScore] = []
    with console.status(f"[cyan]Avaliando {len(cases)} caso(s)...") as spinner:
        for score in run_evaluation(cases, with_judge=judge):
            scores.append(score)
            spinner.update(f"[cyan]Avaliando... {len(scores)}/{len(cases)} · {score.case_id}")
            console.print(
                f"  {'[green]PASS[/]' if score.passed else '[red]FAIL[/]'} {score.case_id}"
            )

    report = build_report(scores)
    _render_report(report)

    if compare_with is not None:
        _render_comparison(compare(load_report(compare_with), report), compare_with)

    if save:
        path = save_report(report, PROJECT_ROOT / "evals" / "results")
        console.print(f"[dim]relatório salvo em {path}")

    # A verbose prompt or a larger k shows up here as a number rather than as a
    # surprise on the invoice.
    if max_cost and report.total_cost_usd > max_cost:
        console.print(
            f"\n[red]acima do teto de custo:[/] "
            f"US$ {report.total_cost_usd:.4f} > US$ {max_cost:.4f}"
        )
        raise typer.Exit(code=1)

    # A threshold rather than "every case must pass": one known failure should
    # not block a release, while a real regression should.
    overall = report.overall.ratio or 0.0
    if overall < min_score:
        # One decimal, because 26 of 29 rounds to 90% and reading
        # "90% < 90%" as the reason for a failure helps nobody.
        console.print(f"\n[red]abaixo do limiar:[/] {overall:.1%} < {min_score:.1%}")
        raise typer.Exit(code=1)


def _evaluate_on_langfuse(*, name: str | None, with_judge: bool, min_score: float) -> None:
    """Run the same suite on the platform, and report it the same way here.

    Langfuse keeps the detail and the comparison; the terminal still gets the
    table, because having to open a browser to learn whether the suite passed
    is a worse trade than the history is worth.
    """
    settings = get_settings()
    run_name = name or f"{settings.chat_model}-k{settings.retrieval_k}-{_today()}"
    configuration = capture_configuration().to_dict()
    configuration.pop("prompt")

    with console.status(f"[cyan]Avaliando no Langfuse: {run_name}..."):
        result = _with_langfuse(
            lambda: run_experiment(
                name=run_name,
                description=f"chunking {settings.chunk_strategy.value}, k={settings.retrieval_k}",
                metadata=configuration,
                with_judge=with_judge,
            )
        )

    counts = summarise(result)
    _render_counts(counts, run_name)

    url = getattr(result, "dataset_run_url", None)
    if url:
        console.print(f"[dim]{url}")

    overall = _overall_of(counts)
    if overall < min_score:
        console.print(f"\n[red]abaixo do limiar:[/] {overall:.1%} < {min_score:.1%}")
        raise typer.Exit(code=1)


def _render_counts(counts: dict[str, tuple[int, int]], run_name: str) -> None:
    """The same table the local run prints, built from the platform's result."""
    table = Table(title=f"avaliação · {run_name}", border_style="cyan")
    table.add_column("metric")
    table.add_column("score", justify="right")
    table.add_column("what it measures", style="dim")

    for metric, description in _METRIC_LABELS.items():
        if metric not in counts:
            continue
        passed, total = counts[metric]
        table.add_row(metric, f"{passed / total:.0%}" if total else "n/a", description)

    passed, total = _totals(counts)
    table.add_row("overall", f"{passed / total:.0%}" if total else "n/a", "positive scores overall")
    console.print(table)


def _overall_of(counts: dict[str, tuple[int, int]]) -> float:
    passed, total = _totals(counts)
    return passed / total if total else 0.0


def _totals(counts: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return (
        sum(passed for passed, _ in counts.values()),
        sum(total for _, total in counts.values()),
    )


def _render_report(report: EvalReport) -> None:
    """Print the summary, then the failures, because failures are the point."""
    table = Table(title="avaliação", border_style="cyan")
    table.add_column("metric")
    table.add_column("score", justify="right")
    table.add_column("what it measures", style="dim")

    table.add_row("retrieval", report.retrieval_accuracy.percent, "trouxe o documento certo")
    table.add_row("citação", report.citation_accuracy.percent, "citou a fonte certa")
    table.add_row("fato", report.factual_accuracy.percent, "o número ou termo esperado apareceu")
    table.add_row("recusa", report.refusal_accuracy.percent, "admitiu não saber, fora do corpus")
    table.add_row("fundamentação", report.groundedness.percent, "todo número saiu do que ele leu")
    table.add_row("geral", report.overall.percent, "passou em tudo que se aplicava")
    console.print(table)

    console.print(
        f"[dim]{report.model} · k={report.retrieval_k} · "
        f"mediana {report.median_latency:.2f}s · "
        f"{report.total_tokens} tokens · ~US$ {report.total_cost_usd:.4f}"
    )

    if not report.failures:
        return

    console.print(f"\n[red]{len(report.failures)} falha(s):")
    for score in report.failures:
        console.print(f"\n[bold]{score.case_id}[/] — {escape(score.question)}")
        if score.error:
            console.print(f"  [red]erro:[/] {escape(score.error[:160])}")
            continue
        console.print(f"  [dim]recuperou:[/] {', '.join(score.retrieved_sources) or 'nada'}")
        console.print(f"  [dim]respondeu:[/] {escape(score.answer[:180])}")
        for label, value in (
            ("retrieval", score.retrieval_hit),
            ("citation", score.citation_correct),
            ("correctness", score.facts_present),
            ("refusal", score.refusal_correct),
            ("groundedness", score.grounded),
            ("faithfulness", score.judged),
        ):
            if value is False:
                console.print(f"  [red]failed on {label}")
        if score.judge_reason and score.judged is False:
            console.print(f"  [red]judge:[/] {escape(score.judge_reason)}")
        if score.ungrounded_numbers:
            console.print(
                f"  [red]números sem apoio na fonte:[/] {', '.join(score.ungrounded_numbers)}"
            )


def _render_comparison(diff: Comparison, source: Path) -> None:
    """Show what moved, and what setting moved with it."""
    console.print(f"\n[bold]comparação com[/] {source.name}")

    if diff.settings:
        for name, (before, after) in diff.settings.items():
            console.print(f"  [cyan]{name}[/] {before} → {after}")
    elif diff.baseline_unknown_configuration:
        # Reports written before the configuration was recorded cannot be
        # compared setting by setting. Saying so beats implying nothing moved.
        console.print("  [dim]o relatório antigo não gravou a configuração")

    for name, (before, after) in diff.changed_metrics.items():
        console.print(f"  {name:<20} {before} → {after}")

    if not diff.changed_metrics:
        console.print("  [dim]nenhuma métrica mudou")

    for case in diff.cases:
        if case.change.value in {"corrigido", "quebrado"}:
            cor = "red" if case.is_regression else "green"
            console.print(f"  [{cor}]{case.change.value}[/] {case.case_id}")

    if diff.regressions:
        console.print(f"  [red]{len(diff.regressions)} regressão(ões)")
