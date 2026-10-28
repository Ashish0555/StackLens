# Atlas Orders architecture — synthetic demonstration document

The HTTP API validates incoming orders and writes order records to PostgreSQL.
Redis stores short-lived session state with a 30-minute expiry. PostgreSQL remains
the source of truth. A Redis outage requires clients to authenticate again.

An outbox record is committed in the same database transaction as an order. A publisher
sends its opaque event ID to Kafka. Consumers acknowledge only after processing.
Delivery is at least once. Consumers deduplicate by event ID; exactly-once effects
are not assumed. Failed messages move to a dead-letter topic after 5 attempts.

Customer identifiers must not appear in application logs. Logs include request IDs,
latency, status and error category. Secrets are injected at runtime and never committed.
Every database query must include the authenticated account's scope.

The system has no documented billing, pricing or employee salary policy.
