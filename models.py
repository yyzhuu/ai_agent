from __future__ import annotations

import os
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
import duckdb
from sqlalchemy import create_engine
from config import ModelConfig, DatabaseConfig


def create_llm(config: ModelConfig) -> BaseChatModel:
    """Create a chat model from config.

    Supports:
    - OpenAI via langchain-openai
    """
    config.validate()

    if config.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        llm = ChatOpenAI(
            model=config.openai_model_name, 
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        return cast(BaseChatModel, llm)


    raise ValueError(f"Unsupported provider: {config.provider}")


def create_database(config: DatabaseConfig):
    config.validate()

    if config.provider == "duckdb":
        con = duckdb.connect(config.duckdb_path, read_only=False)
        return con

    if config.provider == "mysql":
        engine = create_engine(config.mysql_uri)
        return engine

    raise ValueError("Unsupported database provider")