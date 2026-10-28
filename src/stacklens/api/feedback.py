"""What people thought of an answer.

The evaluation dataset is written by whoever built the system, which means it
tests the questions that person thought of. Feedback from real use is the only
source of the ones they did not.

It goes to Langfuse as a score on the trace that produced the answer, which is
where the platform already keeps everything else about that run. A local JSON
lines file is written as well, so the loop still closes when tracing is off
and so the rejected answers can be read without an account.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stacklens.observability import record_score

logger = logging.getLogger(__name__)

FEEDBACK_FILE = "feedback.jsonl"
SCORE_NAME = "user_feedback"


class FeedbackStore:
    """Records a verdict in Langfuse, and keeps a local copy."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / FEEDBACK_FILE
        # Uvicorn serves sync endpoints from a thread pool, so two requests can
        # append at the same moment. A lock is cheaper than the interleaved
        # line it would otherwise write.
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, entry: dict[str, Any]) -> bool:
        """Store one verdict. Returns whether it reached Langfuse."""
        stamped = {"recorded_at": datetime.now(UTC).isoformat(timespec="seconds"), **entry}

        traced = record_score(
            trace_id=str(entry.get("trace_id") or ""),
            name=SCORE_NAME,
            value=bool(entry.get("useful")),
            comment=entry.get("comment"),
        )
        self._append(stamped | {"sent_to_langfuse": traced})

        logger.info("Feedback recorded for run %s (langfuse=%s)", entry.get("run_id"), traced)
        return traced

    def read_all(self) -> list[dict[str, Any]]:
        """Every entry recorded so far, oldest first."""
        if not self._path.is_file():
            return []

        return [
            json.loads(line)
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def unhelpful(self) -> list[dict[str, Any]]:
        """The answers people rejected: the candidates for new test cases."""
        return [entry for entry in self.read_all() if entry.get("useful") is False]

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
