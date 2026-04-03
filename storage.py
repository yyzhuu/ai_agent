from sqlalchemy import create_engine, text
import os 
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URI =( f"postgresql+psycopg2://"
                f"{os.getenv('PSQL_USERNAME')}"
                f":{os.getenv('PSQL_PASSWORD')}@"
                f"{os.getenv('PSQL_HOST', 'localhost')}:"
                f"{os.getenv('PSQL_PORT', '5432')}/"
                f"{os.getenv('PSQL_DATABASE')}?sslmode="
                f"{os.getenv('PSQL_SSLMODE', 'disable')}"
              ) 
engine = create_engine(POSTGRES_URI)


def load_messages_from_postgres(thread_id: str) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT role, content
                FROM chat_history
                WHERE thread_id = :thread_id
                ORDER BY created_at ASC, id ASC
            """),
            {"thread_id": thread_id},
        )
        rows = result.fetchall()

    return [{"role": row[0], "content": row[1]} for row in rows]


def save_message_to_postgres(thread_id: str, role: str, content: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO chat_history (thread_id, role, content)
                VALUES (:thread_id, :role, :content)
            """),
            {
                "thread_id": thread_id,
                "role": role,
                "content": content,
            },
        )