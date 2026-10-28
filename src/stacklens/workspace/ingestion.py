"""Idempotent ingestion shared by local tasks and the Kafka consumer."""

from __future__ import annotations

import logging
from pathlib import Path

from stacklens.config import get_settings
from stacklens.indexing import index_documents, load_documents, split_documents
from stacklens.indexing.scope import collection_scope, user_collection
from stacklens.workspace import store

logger = logging.getLogger(__name__)


def process_job(job_id: str) -> bool:
    job = store.claim_job(job_id)
    if job is None:
        return False
    try:
        original_path = Path(job["path"])
        path = original_path.resolve()
        root = get_settings().upload_dir.resolve()
        if not path.is_relative_to(root) or original_path.is_symlink() or not path.is_file():
            raise ValueError("Upload is not available.")
        documents = load_documents(path.parent)
        if not documents:
            raise ValueError("No extractable text. Scanned PDFs need OCR before upload.")
        if sum(len(d.page_content) for d in documents) > 500_000:
            raise ValueError("Extracted document exceeds the 500,000 character processing limit.")
        for doc in documents:
            doc.metadata["source"] = f"{job['id'][:12]}-{job['filename']}"
        with collection_scope(user_collection(job["owner"])):
            settings = get_settings()
            chunks = split_documents(
                documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                strategy=settings.chunk_strategy,
                article_max_chars=settings.article_max_chars,
            )
            count = index_documents(chunks)
        store.finish_job(job, chunks=count)
    except Exception as exc:
        # Do not persist provider error text: it can contain request data or credentials.
        logger.warning("Ingestion failed for job %s (%s)", job_id, type(exc).__name__)
        message = (
            str(exc)
            if isinstance(exc, ValueError)
            else "Ingestion failed. Check model access and database availability, then retry."
        )
        store.finish_job(job, error=message)
    return True
