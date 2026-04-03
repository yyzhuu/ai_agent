from __future__ import annotations

from memory import get_postgres_store, flush_session_memories_to_postgres


def persist_agent_session(agent_state: dict) -> int:
    """
    Flush all session memories from InMemoryStore into PostgresStore.
    """
    user_id = agent_state["user_id"]
    db_id = agent_state["db_id"]
    session_id = agent_state["session_id"]

    session_store = agent_state["session_store"]
    session_namespace = ("session_memories", user_id, db_id, session_id)

    persistent_namespace = ("long_term_memories", user_id, db_id)

    flushed = 0
    with get_postgres_store() as postgres_store:
        flushed += flush_session_memories_to_postgres(
            session_store=session_store,
            postgres_store=postgres_store,
            session_namespace=session_namespace,
            persistent_namespace=persistent_namespace,
        )

        flushed += flush_session_memories_to_postgres(
            session_store=session_store,
            postgres_store=postgres_store,
            session_namespace=("session_rules", user_id, db_id, session_id),
            persistent_namespace=("long_term_rules", user_id, db_id),
        )

        flushed += flush_session_memories_to_postgres(
            session_store=session_store,
            postgres_store=postgres_store,
            session_namespace=("session_tables", user_id, db_id, session_id),
            persistent_namespace=("long_term_tables", user_id, db_id),
        )

    return flushed