# Atlas Orders API — synthetic demonstration document

This fictional API is sample content for StackLens; it is not a real service contract.

## Creating an order
POST /v1/orders requires an Idempotency-Key header. The key is retained for 24 hours.
A repeated key with the same request body returns the original response. A repeated key
with a different body returns HTTP 409. A successful creation returns HTTP 201.

## Limits and retries
Each account may send 120 requests per minute. Requests above that limit receive HTTP 429.
Clients should honor Retry-After. Retry only HTTP 429 and HTTP 503, for at most 3 attempts.
Do not retry validation failures. The maximum request body is 2 MB.

## Pagination
GET /v1/orders accepts a cursor and a limit. The default page size is 25 orders;
the maximum page size is 100 orders. A missing next_cursor means the final page.
