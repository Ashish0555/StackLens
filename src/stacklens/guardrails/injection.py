"""Prompt injection, on the way in and on the way into the index.

Two doors, and the second is the one that belongs to RAG.

**The question.** Someone types an instruction instead of a question, trying
to talk past the system prompt.

**The corpus.** A retrieved passage is pasted into the context and the model
reads it exactly the way it reads the system prompt: a retriever works in
embedding space and has no notion of "this is data" rather than "this is an
instruction". A document carrying a hidden instruction therefore attacks every
question that retrieves it. OWASP calls this indirect prompt injection, and it
is the risk specific to retrieval rather than to chat.

Scanning the corpus happens **at ingestion**, once per chunk, and not at query
time. The documents are ours and they change when someone indexes them, so
paying per question for an answer that cannot change between questions would
be latency bought for nothing.

## Why not the classifier LLM Guard ships

Measured on this corpus, eight cases, four of them real attacks:

| Model | Correct | False positives | Open |
|---|---|---|---|
| `katanemolabs/Arch-Guard` | 7/8 | 0 | yes |
| `testsavantai/prompt-injection-defender-large-v0` | 6/8 | 0 | yes |
| `jackhhao/jailbreak-classifier` | 5/8 | 0 | yes |
| `protectai/deberta-v3-base-prompt-injection-v2` | 5/8 | **3** | yes |
| `meta-llama/Llama-Prompt-Guard-2-86M` | not tested | | **gated** |

The last row is the one the market would reach for first: Meta's Prompt Guard
2 is multilingual by design and evaluated on Portuguese. It is also a gated
repository, so using it means accepting a licence and carrying a Hugging Face
token, which is a reasonable thing for a company and a poor default for a
project someone clones.

The `protectai` model is what LLM Guard uses by default, and it refused all
three legitimate Portuguese questions at a confidence of 1.00. The same
question in English passed. It is trained on English, and no threshold
separates a language from an attack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from stacklens.config import get_settings

logger = logging.getLogger(__name__)

# Labels a classifier uses for "nothing wrong here". Different models word it
# differently, and the alternative to a list is trusting label order.
_HARMLESS = frozenset({"BENIGN", "SAFE", "NORMAL", "UNHARMFUL", "NO_INJECTION", "LABEL_0"})

# Long enough for a question and for a chunk of this corpus. The model
# truncates past its own window anyway; saying so here makes it deliberate.
_MAX_TOKENS = 512


@dataclass(frozen=True, slots=True)
class InjectionVerdict:
    """What the classifier made of one piece of text."""

    detected: bool
    label: str
    score: float


@lru_cache(maxsize=1)
def _classifier() -> Any:
    """Load the injection classifier once.

    Loading costs seconds and classifying costs milliseconds, so this is held
    for the life of the process.
    """
    from transformers import pipeline

    model = get_settings().injection_model
    logger.info("Loading the injection classifier %s", model)

    return pipeline(
        "text-classification",
        model=model,
        truncation=True,
        max_length=_MAX_TOKENS,
    )


def forget_classifier() -> None:
    """Drop the loaded classifier, after the settings change."""
    _classifier.cache_clear()


def classify(text: str) -> InjectionVerdict:
    """Ask the classifier whether this text is trying to give instructions."""
    if not text.strip():
        return InjectionVerdict(detected=False, label="EMPTY", score=0.0)

    result = _classifier()(text)[0]
    label = str(result["label"]).upper()

    return InjectionVerdict(
        detected=label not in _HARMLESS,
        label=label,
        score=float(result["score"]),
    )


def scan_chunks(texts: list[str]) -> dict[int, InjectionVerdict]:
    """Find the chunks that carry instructions, by position in the list.

    Returns only the ones that tripped, because a clean corpus is the normal
    case and a dictionary of five hundred passes is not information.
    """
    if not get_settings().scan_corpus_for_injection:
        return {}

    flagged = {
        index: verdict for index, text in enumerate(texts) if (verdict := classify(text)).detected
    }

    if flagged:
        logger.warning("Injection suspected in %d of %d chunk(s)", len(flagged), len(texts))

    return flagged
