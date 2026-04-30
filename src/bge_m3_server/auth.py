# src/bge_m3_server/auth.py
import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def create_api_key_dependency(expected_key: str):
    """Create a FastAPI dependency that validates Bearer token API keys.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing attacks (defense-in-depth even though we bind to loopback).
    """

    async def verify_api_key(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(security)
        ] = None,
    ) -> bool:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )
        if not hmac.compare_digest(credentials.credentials, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return True

    return Depends(verify_api_key)
