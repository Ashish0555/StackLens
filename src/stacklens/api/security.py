"""API authentication.

A service that spends money per request cannot be open to whoever reaches the
port. One shared key in a header is the smallest thing that closes that, and
it is enough for a service behind a gateway or inside a private network.

It is deliberately not an identity system. There are no users, no scopes and
no rotation: anything beyond a single shared secret belongs to whatever issues
the tokens, not here.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from stacklens.config import get_settings

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# auto_error=False so a missing header reaches the check below, which can then
# let it through when no key is configured at all.
_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def require_api_key(presented: str | None = Security(_header)) -> None:
    """Reject the request unless it carries the configured key.

    With no key configured the API is open, which is what makes `stacklens serve`
    work out of the box on a laptop. Setting `API_KEY` closes it.
    """
    settings = get_settings()

    if not settings.auth_required:
        return

    expected = settings.api_key.get_secret_value()

    # compare_digest rather than ==: a plain comparison returns as soon as two
    # characters differ, and that timing difference is enough to guess a key
    # one character at a time.
    if presented is None or not hmac.compare_digest(presented, expected):
        logger.warning("Rejected a request with a missing or invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Informe a chave no cabeçalho {API_KEY_HEADER}.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
