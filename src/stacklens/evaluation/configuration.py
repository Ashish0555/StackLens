"""The settings a score depends on, captured with the score.

A report that records the model but not the chunking cannot explain why one
run scored higher than the next. Everything here changes the answer, so
everything here belongs next to the number.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from stacklens.config import get_settings
from stacklens.prompts import build_system_prompt, describe_source


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """Everything that influenced the answers of one evaluation run."""

    model: str
    embedding_model: str
    temperature: float
    knowledge_domain: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    article_max_chars: int
    retrieval_k: int
    prompt_hash: str
    prompt_source: str
    prompt_version: int | None
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def differences_from(self, other: RunConfiguration) -> dict[str, tuple[Any, Any]]:
        """Which settings changed between two runs, as (before, after).

        The prompt itself is compared by hash: printing two prompts side by
        side in a diff is noise, while a changed hash is the fact that matters.
        """
        mine, theirs = self.to_dict(), other.to_dict()
        mine.pop("prompt")
        theirs.pop("prompt")

        return {key: (theirs[key], mine[key]) for key in mine if mine[key] != theirs[key]}


def capture_configuration() -> RunConfiguration:
    """Snapshot the active settings and the prompt they render."""
    settings = get_settings()
    prompt = build_system_prompt()
    source, version = describe_source()

    return RunConfiguration(
        model=settings.chat_model,
        embedding_model=settings.embedding_model,
        temperature=settings.temperature,
        knowledge_domain=settings.knowledge_domain,
        chunk_strategy=settings.chunk_strategy.value,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        article_max_chars=settings.article_max_chars,
        retrieval_k=settings.retrieval_k,
        prompt_hash=hash_prompt(prompt),
        prompt_source=source,
        prompt_version=version,
        prompt=prompt,
    )


def hash_prompt(prompt: str) -> str:
    """A short, stable fingerprint of the prompt.

    Short enough to read in a table, long enough that two different prompts do
    not collide in a project this size.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def configuration_from_dict(data: dict[str, Any]) -> RunConfiguration | None:
    """Read a configuration back from a report, tolerating older ones.

    Reports written before this existed carry only a few fields. Returning
    None for those is honest: the comparison then says the configuration is
    unknown instead of inventing defaults that were never in effect.
    """
    fields = RunConfiguration.__dataclass_fields__
    if not all(key in data for key in fields):
        return None

    return RunConfiguration(**{key: data[key] for key in fields})
