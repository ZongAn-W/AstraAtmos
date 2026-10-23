"""
密码哈希 + JWT 令牌工具
"""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
import bcrypt
from config import JWT_SECRET_KEY, JWT_EXPIRE_HOURS

_ALGORITHM = "HS256"

# ─── 密码 ───

def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode('utf-8'), salt).decode('utf-8')


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))


# ─── JWT ───

def create_access_token(data: dict) -> str:
    """生成 JWT；payload 中必须包含 sub（user_id）和 role。"""
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    解码并验证 JWT。
    失败（过期/签名错误）时抛出 JWTError。
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[_ALGORITHM])
