from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()


def build_postgres_conninfo() -> str:
    username = os.getenv("PSQL_USERNAME")
    password = os.getenv("PSQL_PASSWORD")
    host = os.getenv("PSQL_HOST", "localhost")
    port = os.getenv("PSQL_PORT", "5432")
    database = os.getenv("PSQL_DATABASE")
    sslmode = os.getenv("PSQL_SSLMODE", "disable")

    if not username:
        raise EnvironmentError("PSQL_USERNAME is not set.")
    if password is None:
        raise EnvironmentError("PSQL_PASSWORD is not set.")
    if not database:
        raise EnvironmentError("PSQL_DATABASE is not set.")

    return (
        f"postgres://{username}:{password}"
        f"@{host}:{port}/{database}"
        f"?sslmode={sslmode}"
    )


@asynccontextmanager
async def get_checkpointer():
    """
    Async Postgres checkpointer for LangGraph thread persistence.
    """
    conninfo = build_postgres_conninfo()

    async with AsyncConnectionPool(
        conninfo=conninfo,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    ) as pool, pool.connection() as conn:
        saver = AsyncPostgresSaver(conn)

        # Call once safely; creates checkpoint tables if needed.
        await saver.setup()

        yield saver