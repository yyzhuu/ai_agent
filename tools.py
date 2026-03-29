from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Iterable, List

import duckdb
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from config import DEFAULT_DB_CONFIG
from log_utils import log_panel, red_border_style

def call_tool(tool_call,callback=None):
    tools_by_name = {tool.name: tool for tool in get_available_tools()}
    tool = tools_by_name.get(tool_call["name"])
    args=tool_call.get("args",{})
    reasoning=args.get("reasoning","")
    inputs={k:v for k,v in args.items() if k!= "reasoning"}
    if callback: 
        callback( 
            f"Using `{tool_call['name']}`\n"
            f"Reason: {reasoning}\n"
            f"Inputs: {inputs}"
        )
    if tool is None:
        response = f"Tool not found: {tool_call['name']}"
    else:
        try:
            response = tool.invoke(tool_call["args"])
        except Exception as exc:
            response = f"Tool execution failed: {exc}"

    return ToolMessage(
        content=str(response),
        tool_call_id=tool_call["id"],
    )

def _get_mysql_engine() -> Engine:
    config = DEFAULT_DB_CONFIG
    config.validate()

    if config.provider != "mysql":
        raise ValueError("MySQL engine requested, but provider is not 'mysql'.")

    return create_engine(config.mysql_uri, pool_pre_ping=True)


@contextmanager
def with_sql_cursor() -> Generator[Any, None, None]:
    """
    Open a read-only DB connection and close it safely.

    Yields:
        - duckdb.DuckDBPyConnection for DuckDB
        - sqlalchemy Connection for MySQL
    """
    config = DEFAULT_DB_CONFIG
    conn = None

    try:
        config.validate()

        if config.provider == "duckdb":
            conn = duckdb.connect(config.duckdb_path, read_only=True)
            yield conn

        elif config.provider == "mysql":
            engine = _get_mysql_engine()
            with engine.connect() as mysql_conn:
                yield mysql_conn

        else:
            raise ValueError(f"Unsupported database provider: {config.provider}")

    except Exception as exc:
        log_panel(
            title="Database Error",
            content=str(exc),
            border_style=red_border_style,
        )
        raise

    finally:
        if config.provider == "duckdb" and conn is not None:
            conn.close()


def _rows_to_string(rows: Iterable[tuple]) -> str:
    return "\n".join(str(tuple(row)) for row in rows)


def _fetch_all(conn: Any, query: str, params: dict | tuple | None = None) -> list:
    """
    Run a query for either DuckDB or MySQL and return fetchall().
    """
    config = DEFAULT_DB_CONFIG

    if config.provider == "duckdb":
        if params is None:
            return conn.execute(query).fetchall()
        return conn.execute(query, params).fetchall()

    if config.provider == "mysql":
        result = conn.execute(text(query), params or {})
        return result.fetchall()

    raise ValueError(f"Unsupported database provider: {config.provider}")


def _table_exists(conn: Any, table_name: str) -> bool:
    config = DEFAULT_DB_CONFIG

    if config.provider == "duckdb":
        query = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = ?
            LIMIT 1;
        """
        rows = conn.execute(query, [table_name]).fetchall()
        return bool(rows)

    if config.provider == "mysql":
        query = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND table_name = :table_name
            LIMIT 1;
        """
        rows = conn.execute(text(query), {"table_name": table_name}).fetchall()
        return bool(rows)

    raise ValueError(f"Unsupported database provider: {config.provider}")


@tool
def list_tables(reasoning: str) -> List[str]:
    """Return all table names in the configured database."""
    log_panel(
        title="List Table Tool",
        content=f"Reasoning: {reasoning}",
    )

    config = DEFAULT_DB_CONFIG

    if config.provider == "duckdb":
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name;
        """
    elif config.provider == "mysql":
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
            ORDER BY table_name;
        """
    else:
        raise ValueError(f"Unsupported database provider: {config.provider}")

    with with_sql_cursor() as conn:
        rows = _fetch_all(conn, query)
        log_panel(
            title="List Table Tool Result",
            content=f"Rows fetched: {len(rows)}\nRows: {rows}"
        )
        return [row[0] for row in rows]


@tool
def sample_table(reasoning: str, table_name: str, row_sample_size: int = 5) -> str:
    """Return sample rows from a table, one row per line."""
    safe_limit = max(1, min(row_sample_size, 5))

    log_panel(
        title="Sample Table Tool",
        content=(
            f"Table: {table_name}\n"
            f"Rows: {safe_limit}\n"
            f"Reasoning: {reasoning}"
        ),
    )

    with with_sql_cursor() as conn:
        if not _table_exists(conn, table_name):
            return f"Table not found: {table_name}"

        config = DEFAULT_DB_CONFIG

        if config.provider == "duckdb":
            query = f'SELECT * FROM "{table_name}" LIMIT {safe_limit};'
            rows = conn.execute(query).fetchall()
        elif config.provider == "mysql":
            query = f"SELECT * FROM `{table_name}` LIMIT {safe_limit};"
            rows = conn.execute(text(query)).fetchall()
        else:
            raise ValueError(f"Unsupported database provider: {config.provider}")

        if not rows:
            return f"No rows found in table: {table_name}"

        return _rows_to_string(rows)


@tool
def describe_table(reasoning: str, table_name: str) -> str:
    """Return schema information for a table."""
    log_panel(
        title="Describe Table Tool",
        content=f"Table: {table_name}\nReasoning: {reasoning}",
    )

    config = DEFAULT_DB_CONFIG

    with with_sql_cursor() as conn:
        if not _table_exists(conn, table_name):
            return f"Table not found: {table_name}"

        if config.provider == "duckdb":
            query = f'DESCRIBE "{table_name}";'
            rows = conn.execute(query).fetchall()

            if not rows:
                return f"No schema info found for table: {table_name}"

            lines = []
            for row in rows:
                # DuckDB DESCRIBE typically returns:
                # column_name, column_type, null, key, default, extra
                lines.append(str(tuple(row)))
            return "\n".join(lines)

        if config.provider == "mysql":
            query = """
                SELECT
                    column_name,
                    column_type,
                    is_nullable,
                    column_default,
                    column_key
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = :table_name
                ORDER BY ordinal_position;
            """
            rows = conn.execute(text(query), {"table_name": table_name}).fetchall()

            if not rows:
                return f"No schema info found for table: {table_name}"

            lines = []
            for name, col_type, is_nullable, default_value, column_key in rows:
                lines.append(
                    f"column={name}, type={col_type}, notnull={is_nullable == 'NO'}, "
                    f"default={default_value}, primary_key={column_key == 'PRI'}"
                )
            return "\n".join(lines)

        raise ValueError(f"Unsupported database provider: {config.provider}")


def _is_safe_readonly_sql(sql_query: str) -> tuple[bool, str]:
    stripped = " ".join(sql_query.strip().lower().split())

    if not stripped:
        return False, "Empty query."

    allowed_prefixes = ("select", "with", "show", "describe", "desc")
    if not stripped.startswith(allowed_prefixes):
        return False, "Only read-only SELECT/SHOW/DESCRIBE queries are allowed."

    forbidden_keywords = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " replace ",
        " truncate ",
        " attach ",
        " detach ",
        " call ",
        " merge ",
        " grant ",
        " revoke ",
        " commit ",
        " rollback ",
    ]

    padded = f" {stripped} "
    if any(keyword in padded for keyword in forbidden_keywords):
        return False, "Query blocked: only safe read-only statements are allowed."

    return True, ""


@tool
def execute_sql(reasoning: str, sql_query: str) -> List[str]:
    """Execute a read-only SQL query and return rows as strings."""
    log_panel(
        title="Execute SQL Tool",
        content=f"Query: {sql_query}\nReasoning: {reasoning}",
    )

    is_safe, message = _is_safe_readonly_sql(sql_query)
    if not is_safe:
        return [message]

    with with_sql_cursor() as conn:
        rows = _fetch_all(conn, sql_query)
        if not rows:
            return ["Query returned no rows."]
        return [str(tuple(row)) for row in rows]


def get_available_tools():
    return [list_tables, sample_table, describe_table, execute_sql]