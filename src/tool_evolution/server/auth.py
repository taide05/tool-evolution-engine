import hmac
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from ..utils.config import settings

logger = logging.getLogger(__name__)


def require_api_key() -> None:
    if not settings.api_key:
        raise RuntimeError(
            "TOOLEVO_API_KEY is not set — refusing to start (fail-closed)"
        )


async def api_key_middleware(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    provided = request.headers.get("X-API-Key")
    if not provided or not hmac.compare_digest(provided, settings.api_key or ""):
        logger.warning("auth rejected: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)
