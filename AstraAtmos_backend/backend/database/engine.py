"""
数据库引擎 — 异步 SQLite (aiosqlite + SQLAlchemy AsyncSession)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


# 异步引擎；check_same_thread=False 对 SQLite 多线程访问是必要的
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

# Session 工厂（每个请求创建一个独立 session）
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
