"""Fusing several rankings into one.

Nothing here touches a database. Two retrievers disagree, and this decides
what the disagreement means; keeping that free of I/O is what makes it
testable and what lets the retrievers underneath be replaced without
rewriting it.
"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document

# Reciprocal rank fusion. The constant damps the top of each list so one
# retriever cannot dominate the other with a single confident hit; 60 is the
# value from the paper the method comes from, and the one every implementation
# starts with.
RRF_CONSTANT = 60

# Each retriever is asked for this many times the number of passages wanted,
# and the fused list is cut back to it. Fusing two short lists rewards the
# documents both retrievers already agreed on, which are the ones a single
# retriever would have found anyway; the passages worth adding sit deeper in
# one list. Measured on this corpus, the suspension deadline reaches the top
# eight at a multiplier of five and not at three.
FUSION_POOL = 5

_WORD = re.compile(r"[a-z0-9]+")

# The ordinal marks are compatibility characters that decompose into letters,
# turning "§2º" into "2o" and hiding it from a search for "§ 2". This corpus is
# written in articles and paragraphs, so it is full of them.
_ORDINALS = str.maketrans("", "", "ºª°")


def tokenise(text: str) -> list[str]:
    """Words, lowercased and stripped of accents.

    Portuguese writes the same word with and without an accent often enough
    that folding them together is worth more than the precision it costs. The
    stored side is folded too, by the `unaccent` step in the text search
    configuration, so both sides of a comparison are reduced the same way.
    """
    stripped = text.lower().translate(_ORDINALS)
    decomposed = unicodedata.normalize("NFKD", stripped)
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))

    return _WORD.findall(folded)


def fuse(rankings: list[list[Document]], limit: int) -> list[Document]:
    """Merge several rankings into one by reciprocal rank fusion.

    Each document scores the sum of 1/(constant + rank) across the lists it
    appears in, so a document ranked well by both retrievers outranks one
    ranked brilliantly by a single retriever. Fusing on rank rather than on
    score is what makes it work at all: a cosine distance and a text search
    rank are not on the same scale and cannot be added.
    """
    scores: dict[str, float] = {}
    seen: dict[str, Document] = {}

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = identity(document)
            seen.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_CONSTANT + rank)

    best = sorted(scores, key=lambda key: -scores[key])[:limit]
    return [seen[key] for key in best]


def identity(document: Document) -> str:
    """What makes two results the same chunk, across two retrievers."""
    return f"{document.metadata.get('source', '')}::{document.page_content}"
