"""
SQLite database setup using SQLAlchemy.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import logging

logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(os.getcwd(), 'healthcare.db')}")


class Base(DeclarativeBase):
    pass


# Optimized engine configuration
engine_kwargs = {
    "future": True,
    "echo": False,  # Set True for SQL debugging
}

if DB_URL.startswith("sqlite"):
    engine_kwargs.update({
        "connect_args": {
            "check_same_thread": False,
            "timeout": 30.0,  # Increased timeout for better concurrency
        },
        "poolclass": pool.StaticPool,  # Better for SQLite
    })
else:
    engine_kwargs.update({
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,  # Test connections before using
    })

engine = create_engine(DB_URL, **engine_kwargs)

# Enable SQLite WAL mode for better concurrent access
if DB_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create database tables."""
    from backend.models import Analysis  # noqa: F401 - ensure model is imported
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


