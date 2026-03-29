from __future__ import annotations

import random
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import ask, create_history
from config import DEFAULT_DB_CONFIG, DEFAULT_MODEL_CONFIG
from models import create_llm
from tools import get_available_tools, with_sql_cursor
import pandas as pd
import streamlit as st
import re 
from database import get_duckdb_writer


LOADING_MESSAGES = [
    "Inspecting the database...",
    "Reading table structure...",
    "Checking sample data...",
    "Preparing the answer...",
]


@st.cache_resource(show_spinner=False)
def get_model() -> BaseChatModel:
    llm = create_llm(DEFAULT_MODEL_CONFIG)
    llm = llm.bind_tools(get_available_tools())
    return llm


def load_css() -> None:
    st.markdown(
        """
        <style>
        .stChatMessage { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_parsed_tablename(name:str) -> str: 
    name = Path(name).stem
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not name:
        name = "uploaded_data"
    return name

def upload_csv_to_duckdb(uploaded_file, table_name: str) -> None:
    df = pd.read_csv(uploaded_file, sep=",", engine="python")

    conn = get_duckdb_writer(DEFAULT_DB_CONFIG)
    try:
        conn.register("temp_df", df)
        conn.execute(f'''
            CREATE OR REPLACE TABLE "{table_name}" AS
            SELECT * FROM temp_df
        ''')
    finally:
        conn.close()


def render_sidebar() -> None:
    st.sidebar.write("### Upload CSV to DuckDB")

    uploaded_files = st.sidebar.file_uploader("Choose CSV file", type=["csv"],accept_multiple_files=True,)

    if st.sidebar.button("Upload to DuckDB"):
        if uploaded_files is None:
            st.sidebar.error("Please upload a CSV file first.")
            return
        
        success_tables=[]
        failed_files=[]

        for uploaded_file in uploaded_files: 
            table_name = get_parsed_tablename(uploaded_file.name)

            try:
                upload_csv_to_duckdb(uploaded_file, table_name)
                success_tables.append(table_name) 
            except Exception as exc:
                failed_files.append(uploaded_file, str(exc))

        if success_tables:
            st.sidebar.success(f"Uploaded tables: {', '.join(success_tables)}")

        for filename, err in failed_files:
            st.sidebar.error(f"{filename} failed: {err}")


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="SQL Agent Chat", page_icon="🗃️", layout="wide")
    load_css()

    st.header("SQL Database Chat Assistant")
    st.subheader("Ask questions about your database using a tool-enabled LLM")

    render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = create_history()

    for message in st.session_state.messages:
        if isinstance(message, SystemMessage):
            continue

        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(str(message.content))

    prompt = st.chat_input("Ask a question about the database")
    if not prompt:
        return

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.container() 
        with st.spinner(random.choice(LOADING_MESSAGES)):

            def callback(msg):
                container.write(msg)
            response=ask(
                prompt, 
                st.session_state.messages[:-1], # remove the last msg 
                get_model(), 
                callback=callback
            )
        container.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    main()
