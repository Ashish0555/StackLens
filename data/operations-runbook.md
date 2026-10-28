# Atlas Orders operations — synthetic demonstration document

## Service targets
The monthly availability target is 99.9 percent. The p95 response-time target is 300 ms.
These are design targets in fictional sample data, not measured StackLens performance.

## Incident handling
A sustained error rate above 5 percent for 10 minutes opens a critical incident.
The on-call engineer acknowledges a critical incident within 15 minutes.
Roll back when the new release increases p95 latency above 600 ms for 5 consecutive minutes.

## Backups and recovery
A database snapshot is taken every 6 hours. Recovery point objective is 6 hours;
recovery time objective is 60 minutes. Restore into a new database, verify integrity,
then switch the application connection. Never overwrite the only backup.

## Worker failures
Failed ingestion jobs require an explicit retry after the underlying issue is resolved.
The same document is deduplicated within its owner's collection. A worker lease lasts
15 minutes. Duplicate delivery must not create duplicate document chunks.
