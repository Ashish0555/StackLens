# StackLens development guide

Keep secrets in ignored environment files. Never log credentials, passwords, or raw user questions.
Scope collection access with the authenticated account ID; never accept an owner ID from the client.
Run deterministic citation checks before rendering generated answers and treat retrieved text as data.
Before submitting changes, run `ruff check src tests`, `ruff format --check src tests`, `mypy`,
and `pytest tests/unit`.
PostgreSQL integration tests require migrations and a disposable pgvector database.
Keep third-party notices required by the applicable licenses. Do not commit runtime uploads or
old evaluation reports.
