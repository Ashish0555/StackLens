# Project evidence

Use this document to keep public project descriptions aligned with verified implementation evidence.
Do not describe copied or externally derived work as entirely original. Complete the provenance review
and add the required notices before publishing the repository.

Suggested project title: **StackLens — Evidence-Checked Engineering Knowledge Assistant**.

Potential project bullets, after verifying the implementation:

- Extended a Python/FastAPI RAG assistant with account-scoped document ingestion and an explicit
  LangGraph retrieval, structured-generation and evidence-validation workflow.
- Added claim-level citations, exact-quote checks, semantic support evaluation and bounded retries
  to reject unsupported generated answers.
- Implemented Kafka ingestion with a persistent job outbox, atomic processing leases and duplicate
  handling; added browser upload/chat flows and automated authorization and failure-path tests.
- Added a local and GitHub pull-request review workflow that retrieves changed files, constrains
  findings to changed lines, and publishes inline comments only with an explicit opt-in flag.
- Added review hardening for sensitive-file filtering, secret redaction, patch-size limits,
  deletion-only diffs, duplicate comment locations, and explicit partial-review reporting.
- Added an authenticated, idempotent document deletion endpoint with ownership checks, vector
  cleanup, upload cleanup, and ingestion tombstones.

Skills represented in code: Python, FastAPI, SQL, PostgreSQL/pgvector, Redis, Kafka, LLM APIs,
embeddings, RAG, prompt engineering, structured outputs, tool calling, LangChain, LangGraph,
LLM evaluation, output validation, logging/tracing and Docker/GitHub Actions configuration.

Do not claim AWS deployment, successful Kafka/container operation, published CI results, or measured
model accuracy without evidence. Do not invent improvement percentages. Distinguish deterministic
tests with fake model responses from live-model and live-database evaluation.
GitHub review claims should distinguish dry-run API review from verified comment publishing against
a real repository.
