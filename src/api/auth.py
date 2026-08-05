from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from src.config import config

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """
    API 密钥身份认证依赖 (Milestone 5 Authentication)。
    支持通过 HTTP Header `X-API-Key` 或 `Authorization: Bearer <token>` 传入密钥。
    """
    token = header_key
    if not token and bearer_creds:
        token = bearer_creds.credentials

    if not token or token != config.gea_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="❌ 无效或缺失的 API 密钥 (X-API-Key 或 Bearer token 无效)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
