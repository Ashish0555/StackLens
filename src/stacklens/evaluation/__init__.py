"""Measuring the agent against questions whose answers are known.

Unit tests prove the code does what it was written to do. This proves the
system answers correctly -- a different question, and the one that decides
whether a change to the prompt, the chunking or the model made things better
or worse.
"""

from stacklens.evaluation.comparison import CaseChange, Change, Comparison, compare, load_report
from stacklens.evaluation.configuration import RunConfiguration, capture_configuration, hash_prompt
from stacklens.evaluation.dataset import DEFAULT_DATASET, EvalCase, load_dataset
from stacklens.evaluation.experiments import (
    DATASET_NAME,
    LangfuseUnavailableError,
    run_experiment,
    summarise,
    sync_dataset,
)
from stacklens.evaluation.metrics import (
    CaseScore,
    error_score,
    extract_retrieved_sources,
    groundedness,
    is_refusal,
    score_case,
)
from stacklens.evaluation.runner import EvalReport, Rate, build_report, run_evaluation, save_report

__all__ = [
    "DATASET_NAME",
    "DEFAULT_DATASET",
    "CaseChange",
    "CaseScore",
    "Change",
    "Comparison",
    "EvalCase",
    "EvalReport",
    "LangfuseUnavailableError",
    "Rate",
    "RunConfiguration",
    "build_report",
    "capture_configuration",
    "compare",
    "error_score",
    "extract_retrieved_sources",
    "groundedness",
    "hash_prompt",
    "is_refusal",
    "load_dataset",
    "load_report",
    "run_evaluation",
    "run_experiment",
    "save_report",
    "score_case",
    "summarise",
    "sync_dataset",
]
