from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Any
import duckdb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import DEFAULT_DB_CONFIG, DEFAULT_MODEL_CONFIG


def get_duckdb_writer(DEFAULT_DB_CONFIG) -> duckdb.DuckDBPyConnection:
    """Writable DuckDB connection for file ingestion only."""
    config=DEFAULT_DB_CONFIG
    config.validate()
    if config.provider != "duckdb":
        raise ValueError("DuckDB writer requested, but DB_PROVIDER is not 'duckdb'.")
    return duckdb.connect(config.duckdb_path, read_only=False)


def get_duckdb_reader(DEFAULT_DB_CONFIG) -> duckdb.DuckDBPyConnection:
    """Read-only DuckDB connection for query tools / sidebar / agent."""
    config=DEFAULT_DB_CONFIG
    config.validate()
    if config.provider != "duckdb":
        raise ValueError("DuckDB reader requested, but DB_PROVIDER is not 'duckdb'.")
    return duckdb.connect(config.duckdb_path, read_only=True)


def get_sqlalchemy_reader(DEFAULT_DB_CONFIG) -> Engine:
    """Read-only SQLAlchemy engine for external SQL databases."""
    config=DEFAULT_DB_CONFIG
    config.validate()
    if config.provider != "mysql":
        raise ValueError("SQLAlchemy reader requested, but DB_PROVIDER is not 'mysql'.")
    return create_engine(config.mysql_uri, pool_pre_ping=True)


@contextmanager
def with_read_cursor(DEFAULT_DB_CONFIG) -> Iterator[Any]:
    """
    Unified read-only cursor/connection context for tools and sidebar.
    - DuckDB: yields DuckDB connection
    - MySQL: yields SQLAlchemy connection
    """
    config=DEFAULT_DB_CONFIG
    config.validate()

    if config.provider == "duckdb":
        conn = get_duckdb_reader(config)
        try:
            yield conn
        finally:
            conn.close()
        return

    if config.provider == "mysql":
        engine = get_sqlalchemy_reader(config)
        with engine.connect() as conn:
            yield conn
        return

    raise ValueError(f"Unsupported provider: {config.provider}")


def is_safe_readonly_sql(sql: str) -> bool:
    """
    Allow only read-only SQL.
    """
    normalized = " ".join(sql.strip().lower().split())

    blocked_prefixes = (
        "insert", "update", "delete", "drop", "alter", "create", "replace",
        "truncate", "attach", "detach", "copy", "call", "merge", "grant",
        "revoke", "begin", "commit", "rollback", "vacuum"
    )

    if not normalized:
        return False

    if normalized.startswith(blocked_prefixes):
        return False

    # optional stricter rule:
    # only allow select/show/describe/pragma/explain
    allowed_prefixes = ("select", "show", "describe", "desc", "pragma", "explain", "with")
    return normalized.startswith(allowed_prefixes)