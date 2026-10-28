"""Bounded, collection-scoped retrieval exposed as a model tool."""

from __future__ import annotations

import hashlib
import json

from langchain_core.tools import StructuredTool

from stacklens.config import get_settings
from stacklens.indexing import search
from stacklens.indexing.scope import collection_scope, current_collection
from stacklens.prompts import build_search_tool_description
from stacklens.types import SearchHit

TOOL_NAME = "search_documentation"
_NO_RESULTS = "No relevant passages found in the documentation."
_BUDGET_SPENT = "Search budget exhausted. Answer using existing evidence or refuse."


def evidence_id(hit: SearchHit) -> str:
    return hashlib.sha256(
        f"{current_collection()}:{hit.source}:{hit.content}".encode()
    ).hexdigest()[:20]


def search_documentation(question: str) -> str:
    hits = search(question)
    return json.dumps(
        {
            "passages": [
                {
                    "id": evidence_id(h),
                    "source": h.source,
                    "text": h.content,
                    "distance": h.distance,
                }
                for h in hits
            ]
        },
        ensure_ascii=False,
    )


def build_search_tool() -> StructuredTool:
    budget = get_settings().max_searches_per_turn
    collection = current_collection()
    used = 0

    def search_with_budget(question: str) -> str:
        nonlocal used
        used += 1
        if used > budget:
            return _BUDGET_SPENT
        # Tool executors can use their own thread/context. Capture the authenticated
        # collection when building the tool, not from model-supplied arguments.
        with collection_scope(collection):
            return search_documentation(question)

    return StructuredTool.from_function(
        func=search_with_budget, name=TOOL_NAME, description=build_search_tool_description()
    )


def _format_hit(position: int, hit: SearchHit) -> str:
    return f"--- Passage {position} [source: {hit.citation} | distance {hit.distance:.3f}]\n{hit.content}"
