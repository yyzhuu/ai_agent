from __future__ import annotations

import asyncio
import streamlit as st

from agents import create_runtime_cache
from chat_service import ask_with_hybrid_memory
from config import DEFAULT_DB_CONFIG
from storage import load_messages_from_postgres, save_message_to_postgres
from sqlalchemy import text
from memory import build_postgres_conninfo

def load_css() -> None:
    st.markdown(
        """
        <style>
        .stChatMessage { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_session_history():
    st.session_state.messages = []
    st.session_state.runtime_cache = {}
    # optional: reset current thread
    # st.session_state.thread_id = str(uuid.uuid4())


def clear_postgres_memory():
    engine = build_postgres_conninfo()  # your postgres engine

    with engine.begin() as conn:
        # chat history
        conn.execute(text("DELETE FROM chat_history;"))

        # optional: chat sessions too
        conn.execute(text("DELETE FROM chat_sessions;"))

        # langgraph checkpoint memory
        conn.execute(text("DELETE FROM checkpoint_blobs;"))
        conn.execute(text("DELETE FROM checkpoint_writes;"))
        conn.execute(text("DELETE FROM checkpoints;"))


def render_sidebar_controls():
    with st.sidebar:
        st.divider()
        st.subheader("Memory Controls")

        if st.button("Clear Session History", use_container_width=True):
            clear_session_history()
            st.success("Session history cleared.")
            st.rerun()

        if st.button("Clear Postgres Memory", use_container_width=True):
            clear_postgres_memory()
            st.success("Postgres memory cleared.")
            st.rerun()


def init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "user_123_db_main"

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_postgres(
            st.session_state.thread_id
        )

    if "runtime_cache" not in st.session_state:
        st.session_state.runtime_cache = create_runtime_cache()

    if "db_config" not in st.session_state:
        DEFAULT_DB_CONFIG.validate()
        st.session_state.db_config = DEFAULT_DB_CONFIG


async def run_agent(user_prompt: str) -> str:
    answer, updated_cache = await ask_with_hybrid_memory(
        user_prompt=user_prompt,
        thread_id=st.session_state.thread_id,
        runtime_cache=st.session_state.runtime_cache,
    )
    st.session_state.runtime_cache = updated_cache
    return answer

from collections import defaultdict
import streamlit as st


def render_sidebar_history() -> None:
    with st.sidebar:
        st.header("Chat History")

        messages = st.session_state.get("messages", [])
        if not messages:
            st.caption("No messages yet.")
            return

        sessions = defaultdict(list)
        for msg in messages:
            thread_id = msg.get("thread_id", "unknown_session")
            sessions[thread_id].append(msg)

        for thread_id, session_msgs in sessions.items(): # Group messages by thread_id to form sessions
            first_user_msg = next(
                (m["content"] for m in session_msgs if m["role"] == "user"),
                "Untitled chat"
            )
            preview = first_user_msg.replace("\n", " ").strip()
            preview = preview[:50] + ("..." if len(preview) > 50 else "")

            with st.expander(f"Session: {preview}", expanded=False):
                current_turn = None
                turns = []

                for msg in session_msgs:
                    if msg["role"] == "user":
                        current_turn = {"user": msg["content"], "assistant": ""}
                        turns.append(current_turn)
                    elif msg["role"] == "assistant" and current_turn is not None:
                        current_turn["assistant"] = msg["content"]

                for i, turn in enumerate(turns, start=1):
                    st.markdown(f"**Turn {i}**")
                    st.markdown("**User**")
                    st.markdown(turn["user"])
                    st.markdown("**Assistant**")
                    st.markdown(turn["assistant"] or "_No response yet_")
                    st.divider()


def render_main_chat() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="Database Agent", layout="wide")
    load_css()
    init_session()

    render_sidebar_history()

    st.title("AI Database Assistant")

    render_main_chat()

    user_prompt = st.chat_input("Ask a question about your database...")
    if not user_prompt:
        return

    thread_id = st.session_state.thread_id

    save_message_to_postgres(thread_id, "user", user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = asyncio.run(run_agent(user_prompt))
            st.markdown(answer)

    save_message_to_postgres(thread_id, "assistant", answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()