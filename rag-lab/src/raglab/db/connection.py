"""
Database connection management — supports SQLite and Postgres.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global connection pool (lazy initialized)
_pool: Optional[Any] = None
_backend: str = "sqlite"  # "sqlite" or "postgres"


def get_pool(cfg=None):
    """
    Get or create database connection pool.
    
    Args:
        cfg: Optional DatabaseCfg with backend and connection settings
        
    Returns:
        Connection pool (sqlite3.Connection wrapper or psycopg_pool.ConnectionPool)
    """
    global _pool, _backend
    
    if _pool is not None:
        return _pool
    
    # Determine backend
    if cfg and hasattr(cfg, "backend"):
        _backend = cfg.backend
    else:
        # Check environment variable DATABASE_URL
        dsn = os.getenv("DATABASE_URL", "")
        if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
            _backend = "postgres"
        else:
            _backend = "sqlite"
    
    logger.info(f"Initializing database pool: {_backend}")
    
    if _backend == "sqlite":
        _pool = _init_sqlite_pool(cfg)
    elif _backend == "postgres":
        _pool = _init_postgres_pool(cfg)
    else:
        raise ValueError(f"Unsupported database backend: {_backend}")
    
    return _pool


def _init_sqlite_pool(cfg) -> sqlite3.Connection:
    """
    Initialize SQLite connection (simple wrapper, not true pooling).
    
    SQLite is single-threaded, so we use a simple connection wrapper.
    For production, use Postgres with psycopg_pool.
    """
    # Get database path from config or default
    if cfg and hasattr(cfg, "sqlite_path"):
        db_path = cfg.sqlite_path
    else:
        db_path = "./out/neuralbench.db"
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Allow dict-like access to rows
    
    logger.info(f"SQLite database initialized: {db_path}")
    return conn


def _init_postgres_pool(cfg):
    """
    Initialize Postgres connection pool using psycopg_pool.
    
    DSN is read from:
      1. cfg.dsn (if provided)
      2. env DATABASE_URL
      3. default: "postgresql://localhost/neuralbench"
    """
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        raise ImportError(
            "psycopg_pool required for Postgres. "
            "Install with: pip install 'psycopg[binary,pool]'"
        )
    
    # Get DSN
    if cfg and hasattr(cfg, "dsn") and cfg.dsn:
        dsn = cfg.dsn
    else:
        dsn = os.getenv("DATABASE_URL", "postgresql://localhost/neuralbench")
    
    # Pool settings
    min_size = getattr(cfg, "pool_min_size", 2) if cfg else 2
    max_size = getattr(cfg, "pool_max_size", 10) if cfg else 10
    
    pool = ConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        open=True,
    )
    
    logger.info(f"Postgres pool initialized: {dsn.split('@')[-1]} (min={min_size}, max={max_size})")
    return pool


def close_pool():
    """Close the connection pool."""
    global _pool, _backend
    
    if _pool is None:
        return
    
    logger.info(f"Closing database pool: {_backend}")
    
    if _backend == "sqlite":
        _pool.close()
    elif _backend == "postgres":
        _pool.close()
    
    _pool = None


def get_backend() -> str:
    """Get current database backend."""
    return _backend
