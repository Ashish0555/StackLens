"""Review a local unified diff using repository-aware context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stacklens.providers import build_chat_model


class Severity(StrEnum):
    """Allowed review finding severities."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    NIT = "nit"


class ChangedFile(BaseModel):
    """A changed file and the lines that can receive findings."""

    path: str
    diff: str
    content: str
    changed_lines: list[int] = Field(min_length=1)


class ReviewFinding(BaseModel):
    """One actionable finding tied to a changed line."""

    model_config = ConfigDict(extra="forbid")

    path: str
    line: int = Field(gt=0)
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    title: str = Field(min_length=1, max_length=160)
    explanation: str = Field(min_length=1, max_length=1200)
    suggested_fix: str = Field(min_length=1, max_length=1200)


class ReviewReport(BaseModel):
    """Structured result returned by the review model."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2000)
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=50)


def parse_unified_diff(diff_text: str, repository_root: Path) -> list[ChangedFile]:
    """Parse a unified diff and load changed files from the repository."""
    files: list[ChangedFile] = []
    current_path: str | None = None
    current_lines: list[str] = []
    changed_lines: list[int] = []
    new_line = 0

    def finish() -> None:
        if current_path is None:
            return
        path = _safe_relative_path(current_path)
        file_path = (repository_root / path).resolve()
        root = repository_root.resolve()
        if not file_path.is_relative_to(root):
            raise ValueError(f"Diff path escapes repository root: {path}")
        content = file_path.read_text(encoding="utf-8", errors="replace")
        files.append(
            ChangedFile(
                path=path,
                diff="".join(current_lines),
                content=content,
                changed_lines=changed_lines or [1],
            )
        )

    for raw_line in diff_text.splitlines(keepends=True):
        if raw_line.startswith("diff --git "):
            finish()
            current_path = None
            current_lines = [raw_line]
            changed_lines = []
            continue
        if not current_lines:
            continue
        current_lines.append(raw_line)
        if raw_line.startswith("+++ b/"):
            current_path = raw_line[6:].strip()
        elif raw_line.startswith("+++ /dev/null"):
            current_path = None
        elif raw_line.startswith("@@"):
            new_line = _parse_hunk_start(raw_line)
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            changed_lines.append(new_line)
            new_line += 1
        elif not raw_line.startswith("-") and not raw_line.startswith("\\"):
            new_line += 1

    finish()
    if not files:
        raise ValueError("The diff does not contain any files.")
    return files


def review_diff(
    diff_text: str,
    repository_root: Path,
    *,
    model=None,
) -> ReviewReport:
    """Review changed code using only the supplied diff and file context."""
    files = parse_unified_diff(diff_text, repository_root)
    return review_files(files, model=model)


def review_files(files: list[ChangedFile], *, model=None) -> ReviewReport:
    """Review already-loaded files using the shared evidence-bound workflow."""
    if not files:
        raise ValueError("No changed files were provided.")
    reviewer = model or build_chat_model().with_structured_output(ReviewReport)
    context = "\n\n".join(_format_file(item) for item in files)
    prompt = (
        "Review the proposed code changes for likely correctness bugs and security issues. "
        "Use only the supplied diff and repository file contents as evidence. Do not invent "
        "missing files, runtime behavior, or requirements. Report only actionable findings. "
        "Every finding must use a path and line number from changed_lines. Do not report style "
        "preferences or speculative concerns. Return an empty findings list when no actionable "
        "issue is supported. Severity blocker means the change should not ship, major means a "
        "likely user-visible or data-integrity bug, minor means a localized correctness issue, "
        "and nit is a low-impact improvement.\n\n"
        f"{context}"
    )
    report = reviewer.invoke(prompt)
    result = ReviewReport.model_validate(report)
    _validate_findings(result, files)
    return result


def _format_file(item: ChangedFile) -> str:
    return (
        f"FILE: {item.path}\n"
        f"CHANGED_LINES: {item.changed_lines}\n"
        f"DIFF:\n{item.diff}\n"
        f"CURRENT_CONTENT:\n{item.content}"
    )


def _validate_findings(report: ReviewReport, files: list[ChangedFile]) -> None:
    changed_by_path = {item.path: set(item.changed_lines) for item in files}
    for finding in report.findings:
        if finding.path not in changed_by_path:
            raise ValueError(f"Review finding references an unchanged file: {finding.path}")
        if finding.line not in changed_by_path[finding.path]:
            raise ValueError(
                f"Review finding references a non-changed line: {finding.path}:{finding.line}"
            )


def _safe_relative_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe diff path: {path}")
    return candidate.as_posix()


def _parse_hunk_start(line: str) -> int:
    try:
        new_range = line.split(" ")[2]
        return int(new_range.split(",")[0].removeprefix("+"))
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid unified diff hunk header: {line.strip()}") from error


def changed_lines_from_patch(patch: str) -> list[int]:
    """Return added-line numbers from a GitHub file patch."""
    lines: list[int] = []
    new_line = 0
    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            new_line = _parse_hunk_start(raw_line)
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            lines.append(new_line)
            new_line += 1
        elif not raw_line.startswith("-") and not raw_line.startswith("\\"):
            new_line += 1
    return lines or [1]
