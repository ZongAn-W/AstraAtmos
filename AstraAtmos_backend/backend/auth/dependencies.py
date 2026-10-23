"""
FastAPI 依赖项：从 Authorization header 提取并验证用户
"""

from fastapi import Depends, HTTPException, Request
from jose import JWTError
from sqlalchemy import select

from auth.security import decode_access_token
from database.engine import async_session_maker
from database.models import User


def _extract_token(request: Request) -> str | None:
    """从 Authorization: Bearer <token> 中提取 token 字符串。"""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header[7:]
    return None


async def get_current_user(request: Request) -> User:
    """
    强制认证依赖项。
    token 无效、过期或用户不存在时抛出 401。
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="未提供认证令牌")

    try:
        payload = decode_access_token(token)
        user_id: int = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="认证令牌无效或已过期")

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="用户不存在或已被禁用")

    return user


async def get_optional_user(request: Request) -> User | None:
    """
    可选认证依赖项。
    没有 token 或 token 无效时返回 None，不抛出异常。
    """
    token = _extract_token(request)
    if not token:
        return None

    try:
        payload = decode_access_token(token)
        user_id: int = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None

    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    return user if (user and user.is_active) else None


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    管理员权限依赖项。
    非 admin 角色抛出 403。
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
