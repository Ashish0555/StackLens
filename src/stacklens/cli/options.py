"""Every flag the commands share, defined once.

A flag defined next to the command that uses it is defined again next to the
next command that wants it, and the two drift. `--verbose` means the same
thing on every command precisely because there is one of it.
"""

from __future__ import annotations

import typer

from stacklens.evaluation import DEFAULT_DATASET

VerboseOption = typer.Option(False, "--verbose", "-v", help="Mostra logs detalhados.")


TraceOption = typer.Option(False, "--trace", "-t", help="Mostra o raciocínio do agente.")


DatasetOption = typer.Option(DEFAULT_DATASET, "--dataset", "-d", help="Arquivo do dataset.")


LimitOption = typer.Option(0, "--limit", "-n", help="Roda apenas os N primeiros casos.")


SaveOption = typer.Option(True, "--save/--no-save", help="Grava o relatório em evals/results.")


CompareOption = typer.Option(None, "--compare", "-c", help="Compara com um relatório anterior.")


MaxCostOption = typer.Option(
    0.0,
    "--max-cost",
    min=0.0,
    help="Custo máximo em US$ para a execução inteira. 0 desliga a checagem.",
)


JudgeOption = typer.Option(
    True,
    "--judge/--no-judge",
    help="A métrica faithfulness, julgada por um modelo. Ligada por padrão; custa tokens.",
)


MinScoreOption = typer.Option(
    1.0,
    "--min-score",
    min=0.0,
    max=1.0,
    help="Nota geral mínima para o comando sair com sucesso. 1.0 exige tudo.",
)


HostOption = typer.Option("127.0.0.1", "--host", help="Endereço de escuta.")


PortOption = typer.Option(8080, "--port", "-p", help="Porta.")


ReloadOption = typer.Option(False, "--reload", help="Reinicia ao salvar arquivo (desenvolvimento).")


WorkersOption = typer.Option(
    0, "--workers", "-w", min=0, help="Processos. 0 usa API_WORKERS. Ignorado com --reload."
)


RemoveOption = typer.Option(
    None,
    "--remove",
    help="Remove do índice todos os pedaços deste documento. Use o nome exato do arquivo.",
)


YesOption = typer.Option(False, "--yes", "-y", help="Não pede confirmação.")


ResetOption = typer.Option(
    False, "--reset", help="Apaga o índice antes de indexar. Use ao trocar de estratégia."
)


CommitMessageOption = typer.Option(
    "publicado pela CLI", "--message", "-m", help="Mensagem da versão."
)


RunNameOption = typer.Option(
    None, "--name", help="Nome da execução no Langfuse. O padrão é modelo, k e data."
)
