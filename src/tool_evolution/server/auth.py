"""鉴权口径声明（D2-7 修复）：

- fail-closed：未配置 TOOLEVO_API_KEY 拒绝启动（require_api_key 启动门）；
  /health 外全部路径经中间件校验，无 key/错 key → 401
- 密钥比较：hmac.compare_digest 常量时间比较（防时序侧信道）
- 存储口径：明文 env 单 key（TOOLEVO_API_KEY）——非 SHA256 存储。
  单 key 部署下明文 env 是合理选择（无密钥表/多租户需求），
  与"API Key 管理用 SHA256"的通用描述差异以此声明为准
"""

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
