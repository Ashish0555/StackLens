"""The second pass: reordering what the search already retrieved.

The retrievers underneath compare a question to a passage through something
precomputed. An embedding is calculated at ingestion, long before the question
exists, so it compresses the passage without knowing what will be asked of it.
Keyword search compares words, which have no opinion about each other.

A cross-encoder reads the question and the passage together, in one pass, and
answers the question directly: does this passage answer that one. That is why
it is more accurate, and why it cannot be precomputed and costs a model pass
per candidate.

The two stages have different jobs. Retrieval decides what is in the pool and
is judged on recall. Reranking decides what comes out of it and is judged on
precision. A reranker cannot repair a pool the answer is not in, which is
exactly why this is off by default here: the failure this corpus had was
recall, and hybrid search is what fixed it.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from langchain_core.documents import Document

from stacklens.config import RerankStrategy, get_settings

logger = logging.getLogger(__name__)


class RerankerUnavailableError(RuntimeError):
    """Reranking is configured but the model cannot be loaded."""


class Reranker(Protocol):
    """Reorders candidates against the question that retrieved them."""

    def rerank(self, query: str, candidates: list[Document], limit: int) -> list[Document]:
        """The candidates most relevant to the query, best first."""
        ...


class PassThroughReranker:
    """Keeps the order it was given.

    Not a null object for its own sake: it is what makes reranking a
    configuration change rather than a branch at every call site.
    """

    def rerank(self, query: str, candidates: list[Document], limit: int) -> list[Document]:  # noqa: ARG002
        # The query is ignored, and the name is kept: this has to satisfy the
        # protocol so the caller never asks which reranker it is holding.
        return candidates[:limit]


class CrossEncoderReranker:
    """A local cross-encoder, scoring each candidate against the question.

    The model is loaded once and held, because loading is measured in seconds
    and scoring in milliseconds.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            msg = (
                "RERANK_STRATEGY=cross_encoder precisa da dependência opcional. "
                "Instale com: pip install -e '.[rerank]' "
                "ou volte para RERANK_STRATEGY=none."
            )
            raise RerankerUnavailableError(msg) from error

        logger.info("Loading the reranker %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[Document], limit: int) -> list[Document]:
        """Score every candidate against the query and keep the best.

        The pairs are scored in one call rather than one per candidate, since
        the cost is dominated by the forward pass and batching is what the
        model is built for.
        """
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, candidate.page_content) for candidate in candidates]
        scores = model.predict(pairs)  # type: ignore[attr-defined]

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])

        return [candidate for candidate, _ in ranked[:limit]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """The reranker described by the current settings, built once."""
    settings = get_settings()

    if settings.rerank_strategy is RerankStrategy.CROSS_ENCODER:
        return CrossEncoderReranker(settings.rerank_model)

    return PassThroughReranker()


def forget_reranker() -> None:
    """Drop the cached reranker, after the settings change."""
    get_reranker.cache_clear()


def reranking_enabled() -> bool:
    """Whether a second pass runs at all.

    The caller uses this to decide how wide to retrieve: reranking only earns
    its latency when it is given more candidates than it returns.
    """
    return get_settings().rerank_strategy is not RerankStrategy.NONE
