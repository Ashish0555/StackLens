"""Bound request bodies before multipart parsing or allocation."""

from starlette.responses import JSONResponse

from stacklens.config import get_settings


class UploadLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        upload = scope.get("path") == "/api/v1/documents"
        limit = get_settings().max_upload_bytes + 131072 if upload else 65536
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"-1"))
        except ValueError:
            length = -1
        if length < 0 or length > limit:
            response = JSONResponse(
                {"detail": "A valid, bounded Content-Length is required."},
                status_code=413 if length > limit else 411,
            )
            return await response(scope, receive, send)
        # Buffer at most the configured limit, before any route or parser runs.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > limit or len(body) > length:
                response = JSONResponse(
                    {"detail": "Request body exceeds its allowed size."}, status_code=413
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        if len(body) != length:
            return await JSONResponse({"detail": "Invalid request length."}, status_code=400)(
                scope, receive, send
            )
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
